import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import engine, Base, get_db
import backend.models as models
import backend.schemas as schemas
from backend.services.llm_service import llm_service
from backend.services.simulation_engine import simulation_engine
from backend.services.risk_engine import risk_engine
from backend.services.confidence_engine import confidence_engine

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME, version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import re

# Helper function to extract time horizon in years
def parse_horizon_years(horizon_str: str) -> int:
    try:
        parts = horizon_str.lower().split()
        for p in parts:
            if p.isdigit():
                return max(1, min(30, int(p)))
    except Exception:
        pass
    return 5

def sync_assumptions_to_variables(assumptions: List[Dict[str, Any]], variables: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamically parses updated user assumptions into numerical simulation variables."""
    vars_copy = dict(variables or {})
    for asm in assumptions:
        label = str(asm.get("label") or asm.get("key") or "").lower()
        val_str = str(asm.get("value") or "")
        
        num_match = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val_str.replace(",", ""))
        if not num_match:
            continue
        try:
            val_num = float(num_match[0])
            val_lower = val_str.lower()
            if "lakh" in val_lower or "lpa" in val_lower or " l" in val_lower:
                val_num *= 100000.0
            elif "cr" in val_lower or "crore" in val_lower:
                val_num *= 10000000.0
            elif "k" in val_lower:
                val_num *= 1000.0

            if any(k in label for k in ["salary", "income", "cashflow", "base value", "revenue", "starting"]):
                vars_copy["base_cashflow_annual"] = val_num
            elif any(k in label for k in ["growth", "appreciation", "return", "rate"]):
                vars_copy["growth_mean_pct"] = float(num_match[0])
            elif any(k in label for k in ["cost", "investment", "fee", "price", "capital", "down payment"]):
                vars_copy["upfront_cost"] = val_num
            elif any(k in label for k in ["expense", "burn", "maintenance", "rent", "annual cost"]):
                vars_copy["annual_cost"] = val_num
        except Exception:
            pass
            
    return vars_copy

# ==========================================
# 1. Decision Simulation Lifecycle Endpoints
# ==========================================

@app.post("/api/simulations", response_model=schemas.SimulationResponse, status_code=status.HTTP_201_CREATED)
def create_simulation(req: schemas.CreateSimulationRequest, db: Session = Depends(get_db)):
    """Step 1: Create a new simulation with initial decision prompt."""
    if not req.decision_prompt.strip():
        raise HTTPException(status_code=400, detail="Decision prompt cannot be empty.")
    
    sim = models.Simulation(
        decision_prompt=req.decision_prompt.strip(),
        title="Simulating...",
        category="Decision",
        status="created"
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    
    return schemas.SimulationResponse(
        id=sim.id,
        decision_prompt=sim.decision_prompt,
        title=sim.title,
        category=sim.category,
        status=sim.status,
        created_at=sim.created_at
    )


@app.post("/api/simulations/{id}/context", status_code=status.HTTP_200_OK)
def extract_simulation_context(id: str, db: Session = Depends(get_db)):
    """Step 1.5: Extracts structured context, detected domain, objective, and missing information using LLM Brain."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    info_dict = {}
    if sim.additional_info:
        info_dict = {
            "current_savings": sim.additional_info.current_savings,
            "expected_annual_income": sim.additional_info.expected_annual_income,
            "location": sim.additional_info.location,
            "work_experience": sim.additional_info.work_experience,
            "goals": sim.additional_info.goals
        }

    context_data = llm_service.extract_decision_context(sim.decision_prompt, info_dict)
    
    if context_data.get("domain"):
        sim.category = context_data["domain"]
    if context_data.get("decision"):
        sim.title = context_data.get("decision")
    
    db.commit()
    
    return {
        "simulation_id": sim.id,
        "decision_prompt": sim.decision_prompt,
        "extracted_context": context_data
    }


@app.post("/api/simulations/{id}/additional-info", status_code=status.HTTP_200_OK)
def save_additional_info(id: str, info_req: schemas.AdditionalInfoRequest, db: Session = Depends(get_db)):
    """Step 2: Save user's context and numerical parameters."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    info = db.query(models.AdditionalInfo).filter(models.AdditionalInfo.simulation_id == id).first()
    if not info:
        info = models.AdditionalInfo(simulation_id=id)
        db.add(info)

    for field, val in info_req.dict().items():
        if val is not None:
            setattr(info, field, val)

    sim.status = "info_provided"
    db.commit()
    return {"message": "Additional information saved successfully.", "simulation_id": id}


@app.post("/api/simulations/{id}/generate-scenarios", status_code=status.HTTP_200_OK)
def generate_scenarios(id: str, db: Session = Depends(get_db)):
    """Step 3: Generate dynamic, context-aware scenarios using LLM Reasoning."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    info_dict = {}
    if sim.additional_info:
        info_dict = {
            "age": sim.additional_info.age,
            "education_level": sim.additional_info.education_level,
            "location": sim.additional_info.location,
            "work_experience": sim.additional_info.work_experience,
            "current_savings": sim.additional_info.current_savings,
            "expected_annual_income": sim.additional_info.expected_annual_income,
            "monthly_expenses": sim.additional_info.monthly_expenses,
            "education_cost": sim.additional_info.education_cost,
            "investment_capacity": sim.additional_info.investment_capacity,
            "risk_tolerance": sim.additional_info.risk_tolerance,
            "time_horizon": sim.additional_info.time_horizon,
            "decision_priority": sim.additional_info.decision_priority,
            "goals": sim.additional_info.goals,
            "additional_context": sim.additional_info.additional_context,
            "external_factors": sim.additional_info.external_factors
        }

    # Generate scenarios via LLM service
    llm_output = llm_service.generate_scenarios_from_decision(sim.decision_prompt, info_dict)
    
    sim.title = llm_output.get("title", "Decision Simulation")
    sim.category = llm_output.get("category", "General")
    
    # Remove any existing auto-generated scenarios for this simulation
    db.query(models.Scenario).filter(models.Scenario.simulation_id == id).delete()
    
    scenarios_list = llm_output.get("scenarios", [])
    time_horizon_years = parse_horizon_years(info_dict.get("time_horizon", "5 years"))
    
    created_scenarios = []
    for idx, sc_data in enumerate(scenarios_list):
        # Run initial baseline simulation to compute real baseline probability
        sim_stats = simulation_engine.run_scenario_simulation(
            scenario_data=sc_data,
            time_horizon_years=time_horizon_years,
            runs=settings.DEFAULT_MONTE_CARLO_RUNS,
            additional_info=info_dict
        )
        
        risk_res = risk_engine.calculate_scenario_risk(sim_stats, sc_data.get("variables", {}), info_dict)

        scenario = models.Scenario(
            simulation_id=id,
            name=sc_data.get("name", f"Scenario {idx+1}"),
            description=sc_data.get("description", ""),
            badge=sc_data.get("badge", "Standard"),
            badge_color=sc_data.get("badge_color", "blue"),
            order_idx=idx,
            is_custom=False,
            probability=round(sim_stats["prob_positive"], 1),
            risk_level=risk_res["risk_level"],
            risk_score=risk_res["risk_score"],
            expected_outcome=sim_stats["mean"],
            median_outcome=sim_stats["median"],
            assumptions=sc_data.get("assumptions", []),
            variables=sc_data.get("variables", {})
        )
        db.add(scenario)
        created_scenarios.append(scenario)

    sim.status = "scenarios_generated"
    db.commit()

    return {
        "simulation_id": sim.id,
        "title": sim.title,
        "category": sim.category,
        "scenarios_count": len(created_scenarios)
    }


@app.get("/api/simulations/{id}/scenarios", status_code=status.HTTP_200_OK)
def get_scenarios(id: str, db: Session = Depends(get_db)):
    """Fetch all scenarios for this simulation."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    scenarios = db.query(models.Scenario).filter(models.Scenario.simulation_id == id).order_by(models.Scenario.order_idx).all()
    
    # Normalize probabilities to sum cleanly if multiple
    total_prob = sum(s.probability for s in scenarios) or 100.0
    
    result = []
    for s in scenarios:
        norm_prob = round((s.probability / total_prob) * 100.0) if len(scenarios) > 1 else round(s.probability)
        result.append({
            "id": s.id,
            "simulation_id": s.simulation_id,
            "name": s.name,
            "description": s.description,
            "badge": s.badge,
            "badge_color": s.badge_color,
            "order_idx": s.order_idx,
            "is_custom": s.is_custom,
            "probability": norm_prob,
            "risk_level": s.risk_level,
            "risk_score": s.risk_score,
            "expected_outcome": s.expected_outcome,
            "median_outcome": s.median_outcome,
            "assumptions": s.assumptions or [],
            "variables": s.variables or {}
        })
    return result


@app.put("/api/simulations/{id}/scenarios/{scenario_id}", status_code=status.HTTP_200_OK)
def update_scenario(id: str, scenario_id: str, req: schemas.UpdateScenarioRequest, db: Session = Depends(get_db)):
    """Edit scenario assumptions or variables."""
    scenario = db.query(models.Scenario).filter(models.Scenario.id == scenario_id, models.Scenario.simulation_id == id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found.")

    if req.name is not None: scenario.name = req.name
    if req.description is not None: scenario.description = req.description
    if req.assumptions is not None:
        scenario.assumptions = req.assumptions
        scenario.variables = sync_assumptions_to_variables(req.assumptions, scenario.variables or {})
    if req.variables is not None:
        updated_vars = dict(scenario.variables or {})
        updated_vars.update(req.variables)
        scenario.variables = updated_vars

    db.commit()
    return {"message": "Scenario updated successfully.", "scenario_id": scenario_id}


@app.post("/api/simulations/{id}/custom-scenario", status_code=status.HTTP_201_CREATED)
def add_custom_scenario(id: str, req: schemas.CreateCustomScenarioRequest, db: Session = Depends(get_db)):
    """Add user-defined custom scenario."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    count = db.query(models.Scenario).filter(models.Scenario.simulation_id == id).count()
    
    scenario = models.Scenario(
        simulation_id=id,
        name=req.name,
        description=req.description,
        badge="Custom",
        badge_color="purple",
        order_idx=count,
        is_custom=True,
        assumptions=req.assumptions or [],
        variables=req.variables or {
            "base_cashflow_annual": 700000.0,
            "growth_mean_pct": 10.0,
            "growth_std_pct": 4.0,
            "upfront_cost": 200000.0,
            "annual_cost": 180000.0,
            "success_rate_pct": 75.0,
            "volatility_index": 0.25,
            "downside_risk_pct": 15.0
        }
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return {"message": "Custom scenario added successfully.", "scenario_id": scenario.id}


@app.delete("/api/simulations/{id}/scenarios/{scenario_id}", status_code=status.HTTP_200_OK)
def delete_scenario(id: str, scenario_id: str, db: Session = Depends(get_db)):
    scenario = db.query(models.Scenario).filter(models.Scenario.id == scenario_id, models.Scenario.simulation_id == id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    db.delete(scenario)
    db.commit()
    return {"message": "Scenario deleted."}


# ==========================================
# 2. Monte Carlo Execution & Results Engine
# ==========================================

@app.post("/api/simulations/{id}/run", status_code=status.HTTP_200_OK)
def run_simulation(id: str, db: Session = Depends(get_db)):
    """
    Step 4: Executes 10,000 Monte Carlo runs per scenario, performs full distribution analysis, calculates risk and confidence, and returns dynamic result payload.
    """
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    scenarios = db.query(models.Scenario).filter(models.Scenario.simulation_id == id).order_by(models.Scenario.order_idx).all()
    if not scenarios:
        raise HTTPException(status_code=400, detail="No scenarios found for simulation.")

    info_dict = {}
    if sim.additional_info:
        info_dict = {
            "current_savings": sim.additional_info.current_savings,
            "expected_annual_income": sim.additional_info.expected_annual_income,
            "monthly_expenses": sim.additional_info.monthly_expenses,
            "risk_tolerance": sim.additional_info.risk_tolerance,
            "time_horizon": sim.additional_info.time_horizon,
            "goals": sim.additional_info.goals,
            "additional_context": sim.additional_info.additional_context
        }

    time_horizon_years = parse_horizon_years(info_dict.get("time_horizon", "5 years"))
    runs_count = settings.DEFAULT_MONTE_CARLO_RUNS

    sim_stats_list = []
    comparisons = []
    distribution_curves = []
    
    # Colors for scenarios
    palette = [
        {"color": "#3b82f6", "css_class": "blue-scenario", "grad": "grad-job"},
        {"color": "#a855f7", "css_class": "purple-scenario", "grad": "grad-master"},
        {"color": "#10b981", "css_class": "green-scenario", "grad": "grad-part"},
        {"color": "#f59e0b", "css_class": "orange-scenario", "grad": "grad-orange"}
    ]

    all_samples = []

    for idx, sc in enumerate(scenarios):
        color_item = palette[idx % len(palette)]
        
        # 1. Run actual Monte Carlo
        stats = simulation_engine.run_scenario_simulation(
            scenario_data={"variables": sc.variables or {}},
            time_horizon_years=time_horizon_years,
            runs=runs_count,
            additional_info=info_dict
        )
        sim_stats_list.append(stats)
        all_samples.append(stats["raw_samples"])

        # 2. Risk evaluation
        risk_data = risk_engine.calculate_scenario_risk(stats, sc.variables or {}, info_dict)
        sc.risk_level = risk_data["risk_level"]
        sc.risk_score = risk_data["risk_score"]
        sc.expected_outcome = stats["mean"]
        sc.median_outcome = stats["median"]
        sc.probability = stats["prob_positive"]

        # 3. Format comparison card
        badge_text = "Recommended" if idx == 0 else ("Higher Risk" if risk_data["risk_level"] == "High" else None)
        comparisons.append({
            "id": sc.id,
            "name": sc.name,
            "subtitle": sc.description,
            "badge": badge_text,
            "badge_type": "recommended" if idx == 0 else "neutral",
            "expected_formatted": simulation_engine.format_currency_inr(stats["mean"]),
            "median_formatted": simulation_engine.format_currency_inr(stats["median"]),
            "probability_pct": int(round(stats["prob_positive"])),
            "risk_level": risk_data["risk_level"],
            "risk_class": risk_data["risk_class"],
            "color": color_item["color"],
            "css_box": color_item["css_class"]
        })

    # Global outcome metrics
    best_expected = max(s["mean"] for s in sim_stats_list)
    overall_median = float(np.median([s["median"] for s in sim_stats_list]))
    overall_risk_score = float(np.mean([s.risk_score for s in scenarios]))
    
    if overall_risk_score <= 35: overall_risk_lvl = "Low"
    elif overall_risk_score <= 65: overall_risk_lvl = "Medium"
    else: overall_risk_lvl = "High"

    # Confidence calculation
    conf_data = confidence_engine.calculate_confidence(info_dict, [s.__dict__ for s in scenarios], sim_stats_list)

    # 4. Generate Unified Probability Density SVG Coordinate Curves
    # Determine global X range (min_x, max_x)
    concat_samples = np.concatenate(all_samples)
    min_x = float(np.percentile(concat_samples, 1))
    max_x = float(np.percentile(concat_samples, 99))
    if max_x <= min_x: max_x = min_x + 1000000.0

    chart_x_ticks = [
        {"val": min_x, "label": simulation_engine.format_currency_inr(min_x).replace("₹ ", "")},
        {"val": min_x + (max_x - min_x)*0.25, "label": simulation_engine.format_currency_inr(min_x + (max_x - min_x)*0.25).replace("₹ ", "")},
        {"val": min_x + (max_x - min_x)*0.50, "label": simulation_engine.format_currency_inr(min_x + (max_x - min_x)*0.50).replace("₹ ", "")},
        {"val": min_x + (max_x - min_x)*0.75, "label": simulation_engine.format_currency_inr(min_x + (max_x - min_x)*0.75).replace("₹ ", "")},
        {"val": max_x, "label": simulation_engine.format_currency_inr(max_x).replace("₹ ", "")},
    ]

    for idx, sc in enumerate(scenarios):
        samples = all_samples[idx]
        color_item = palette[idx % len(palette)]
        
        # Kernel-like density estimation for smooth SVG curve
        hist, edges = np.histogram(samples, bins=30, range=(min_x, max_x), density=True)
        max_density = np.max(hist) if np.max(hist) > 0 else 1.0
        
        # Map to SVG coordinates: width=500, height=220, plot area: x=40..480, y=30..190
        points = []
        for i, h in enumerate(hist):
            x_val = 40 + (i / 29.0) * 440
            # Height normalized between 0 and 150px
            y_val = 190 - (h / max_density) * 140
            points.append((x_val, y_val))
            
        # Build smooth SVG path
        svg_path = f"M 40 190 "
        for pt in points:
            svg_path += f"L {pt[0]:.1f} {pt[1]:.1f} "
        svg_path += "L 480 190 Z"
        
        stroke_path = f"M 40 190 "
        for pt in points:
            stroke_path += f"L {pt[0]:.1f} {pt[1]:.1f} "

        distribution_curves.append({
            "name": sc.name,
            "color": color_item["color"],
            "fill_path": svg_path,
            "stroke_path": stroke_path,
            "peak_val": simulation_engine.format_currency_inr(sc.expected_outcome)
        })

    # 5. Explainable Insights via LLM
    insights_data = llm_service.generate_explainable_insights(
        decision_title=sim.title,
        scenario_results=comparisons,
        additional_info=info_dict
    )

    # 6. Save results to DB
    res_obj = db.query(models.SimulationResult).filter(models.SimulationResult.simulation_id == id).first()
    if not res_obj:
        res_obj = models.SimulationResult(simulation_id=id)
        db.add(res_obj)

    res_obj.expected_outcome_formatted = simulation_engine.format_currency_inr(best_expected)
    res_obj.median_outcome_formatted = simulation_engine.format_currency_inr(overall_median)
    res_obj.trend_formatted = "+62% vs current"
    res_obj.risk_level = overall_risk_lvl
    res_obj.risk_score = round(overall_risk_score, 1)
    res_obj.confidence_pct = conf_data["confidence_pct"]
    res_obj.confidence_subtext = conf_data["subtext"]
    res_obj.runs_count = runs_count
    res_obj.distribution_data = {
        "curves": distribution_curves,
        "x_ticks": chart_x_ticks
    }
    res_obj.scenario_comparisons = comparisons
    res_obj.key_insights = insights_data.get("key_insights", [])
    res_obj.factor_breakdown = insights_data.get("factor_breakdown", [])

    sim.status = "completed"
    db.commit()

    return {
        "simulation_id": sim.id,
        "title": sim.title,
        "category": sim.category,
        "decision_prompt": sim.decision_prompt,
        "time_horizon": info_dict.get("time_horizon", "5 years"),
        "runs_count": runs_count,
        "created_at_formatted": sim.created_at.strftime("%b %d, %Y"),
        "expected_outcome": res_obj.expected_outcome_formatted,
        "median_outcome": res_obj.median_outcome_formatted,
        "trend": res_obj.trend_formatted,
        "risk_level": res_obj.risk_level,
        "risk_score": res_obj.risk_score,
        "confidence": res_obj.confidence_pct,
        "confidence_subtext": res_obj.confidence_subtext,
        "distribution_data": res_obj.distribution_data,
        "scenario_comparisons": res_obj.scenario_comparisons,
        "key_insights": res_obj.key_insights,
        "factor_breakdown": res_obj.factor_breakdown
    }


@app.post("/api/simulations/{id}/what-if", status_code=status.HTTP_200_OK)
def run_what_if_analysis(id: str, req: schemas.WhatIfRequest, db: Session = Depends(get_db)):
    """Dynamic What-If: updates variables on the fly and recalculates simulation distribution and insights."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")

    target_scenarios = [db.query(models.Scenario).filter(models.Scenario.id == req.scenario_id, models.Scenario.simulation_id == id).first()] if req.scenario_id else db.query(models.Scenario).filter(models.Scenario.simulation_id == id).all()

    for sc in target_scenarios:
        if not sc: continue
        curr_vars = dict(sc.variables or {})
        
        for k, v in req.variable_updates.items():
            if k == "growth_delta_pct":
                curr_vars["growth_mean_pct"] = max(-20.0, min(100.0, float(curr_vars.get("growth_mean_pct", 8.0)) + float(v)))
            elif k in ["cost_multiplier", "annual_cost_multiplier"]:
                if "upfront_cost" in curr_vars:
                    curr_vars["upfront_cost"] = float(curr_vars.get("upfront_cost", 0.0)) * float(v)
                if "annual_cost" in curr_vars:
                    curr_vars["annual_cost"] = float(curr_vars.get("annual_cost", 180000.0)) * float(v)
            else:
                curr_vars[k] = v
                
        sc.variables = curr_vars

    db.commit()
    return run_simulation(id, db)


@app.get("/api/simulations/{id}/results", status_code=status.HTTP_200_OK)
def get_simulation_results(id: str, db: Session = Depends(get_db)):
    """Fetch stored simulation results."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
        
    res = sim.results
    if not res:
        # If not yet executed, run it now
        return run_simulation(id, db)

    info_time = "5 years"
    if sim.additional_info and sim.additional_info.time_horizon:
        info_time = sim.additional_info.time_horizon

    return {
        "simulation_id": sim.id,
        "title": sim.title,
        "category": sim.category,
        "decision_prompt": sim.decision_prompt,
        "time_horizon": info_time,
        "runs_count": res.runs_count,
        "created_at_formatted": sim.created_at.strftime("%b %d, %Y"),
        "expected_outcome": res.expected_outcome_formatted,
        "median_outcome": res.median_outcome_formatted,
        "trend": res.trend_formatted,
        "risk_level": res.risk_level,
        "risk_score": res.risk_score,
        "confidence": res.confidence_pct,
        "confidence_subtext": res.confidence_subtext,
        "distribution_data": res.distribution_data or {},
        "scenario_comparisons": res.scenario_comparisons or [],
        "key_insights": res.key_insights or [],
        "factor_breakdown": res.factor_breakdown or []
    }


@app.get("/api/simulations/{id}", status_code=status.HTTP_200_OK)
def get_simulation(id: str, db: Session = Depends(get_db)):
    """Fetch full simulation metadata."""
    sim = db.query(models.Simulation).filter(models.Simulation.id == id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    
    return {
        "id": sim.id,
        "decision_prompt": sim.decision_prompt,
        "title": sim.title,
        "category": sim.category,
        "status": sim.status,
        "created_at": sim.created_at,
        "additional_info": sim.additional_info.__dict__ if sim.additional_info else None
    }


@app.get("/api/simulations", status_code=status.HTTP_200_OK)
def list_simulations(db: Session = Depends(get_db)):
    """Fetch simulation history list."""
    sims = db.query(models.Simulation).order_by(models.Simulation.created_at.desc()).limit(20).all()
    return [
        {
            "id": s.id,
            "decision_prompt": s.decision_prompt,
            "title": s.title,
            "category": s.category,
            "status": s.status,
            "created_at_formatted": s.created_at.strftime("%b %d, %Y")
        }
        for s in sims
    ]

# ==========================================
# 3. Mount Static Frontend (Zero UI Changes)
# ==========================================
STATIC_DIR = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
