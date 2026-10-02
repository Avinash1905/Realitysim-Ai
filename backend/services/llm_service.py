import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from backend.config import settings

logger = logging.getLogger(__name__)

def clean_json_response(text: str) -> Dict[str, Any]:
    """Robust parser that extracts JSON object from raw LLM output."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try code fence extraction
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                try:
                    return json.loads(p)
                except Exception:
                    pass

    # Try regex matching between outermost { and }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        extracted = text[start:end+1].strip()
        try:
            return json.loads(extracted)
        except Exception:
            pass
            
    # Try fixing common JSON issues like trailing commas
    if start != -1 and end != -1:
        extracted = text[start:end+1].strip()
        cleaned = re.sub(r",\s*([\]}])", r"\1", extracted)
        return json.loads(cleaned)

    raise ValueError("Could not parse valid JSON from LLM output.")


def normalize_scenario_output(data: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensures raw LLM output is structured with required Monte Carlo mathematical variables."""
    info = additional_info or {}
    base_income = float(info.get("expected_annual_income", 600000.0))
    savings = float(info.get("current_savings", 50000.0))

    if not isinstance(data, dict):
        return data

    title = data.get("title", "Decision Simulation")
    category = data.get("category", "Decision")
    raw_scenarios = data.get("scenarios", [])

    palette_colors = ["blue", "red", "green", "purple"]
    palette_badges = ["Most Likely", "Higher Risk", "Balanced", "Conservative"]

    normalized_scenarios = []
    for idx, sc in enumerate(raw_scenarios):
        if not isinstance(sc, dict):
            continue
        sc_name = sc.get("name", f"Scenario {idx+1}")
        sc_desc = sc.get("description", "")
        badge = sc.get("badge", palette_badges[idx % len(palette_badges)])
        badge_color = sc.get("badge_color", palette_colors[idx % len(palette_colors)])

        # 1. Normalize assumptions
        raw_asms = sc.get("assumptions", [])
        norm_asms = []
        if isinstance(raw_asms, list):
            for a in raw_asms:
                if isinstance(a, dict):
                    norm_asms.append({
                        "key": a.get("key") or a.get("label") or "Assumption",
                        "value": str(a.get("value") or ""),
                        "icon": a.get("icon") or "📊",
                        "numeric_val": a.get("numeric_val"),
                        "param_key": a.get("param_key")
                    })
                elif isinstance(a, str):
                    norm_asms.append({
                        "key": "Key Assumption",
                        "value": a,
                        "icon": "📌"
                    })

        # 2. Normalize simulation variables
        raw_vars = sc.get("variables", {})
        norm_vars = {}
        if isinstance(raw_vars, dict):
            for k, v in raw_vars.items():
                try:
                    norm_vars[k] = float(v)
                except Exception:
                    pass

        # Guarantee all Monte Carlo parameters exist
        if "base_cashflow_annual" not in norm_vars:
            norm_vars["base_cashflow_annual"] = base_income * (1.1 if idx == 0 else (1.4 if idx == 1 else 1.0))
        if "growth_mean_pct" not in norm_vars:
            norm_vars["growth_mean_pct"] = 14.0 if idx == 1 else (8.5 if idx == 0 else 6.5)
        if "growth_std_pct" not in norm_vars:
            norm_vars["growth_std_pct"] = 6.0 if idx == 1 else 3.0
        if "upfront_cost" not in norm_vars:
            norm_vars["upfront_cost"] = savings * 0.6 if idx == 1 else 0.0
        if "annual_cost" not in norm_vars:
            norm_vars["annual_cost"] = base_income * 0.25
        if "success_rate_pct" not in norm_vars:
            norm_vars["success_rate_pct"] = 60.0 if idx == 1 else 82.0
        if "volatility_index" not in norm_vars:
            norm_vars["volatility_index"] = 0.45 if idx == 1 else 0.22
        if "downside_risk_pct" not in norm_vars:
            norm_vars["downside_risk_pct"] = 28.0 if idx == 1 else 12.0

        normalized_scenarios.append({
            "name": sc_name,
            "description": sc_desc,
            "badge": badge,
            "badge_color": badge_color,
            "assumptions": norm_asms,
            "variables": norm_vars
        })

    return {
        "title": title,
        "category": category,
        "scenarios": normalized_scenarios
    }


def normalize_insights_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures insights and factors are formatted for the UI."""
    if not isinstance(data, dict):
        data = {}
    key_insights = data.get("key_insights", [])
    factor_breakdown = data.get("factor_breakdown", [])

    norm_insights = []
    icon_types = ["green", "purple", "blue", "orange"]
    if isinstance(key_insights, list):
        for idx, item in enumerate(key_insights):
            if isinstance(item, dict):
                norm_insights.append({
                    "text": item.get("text") or str(item),
                    "icon_type": item.get("icon_type") or icon_types[idx % len(icon_types)]
                })
            elif isinstance(item, str):
                norm_insights.append({
                    "text": item,
                    "icon_type": icon_types[idx % len(icon_types)]
                })

    norm_factors = []
    bar_classes = ["bar-blue", "bar-indigo", "bar-purple", "bar-coral", "bar-orange"]
    if isinstance(factor_breakdown, list):
        for idx, item in enumerate(factor_breakdown):
            if isinstance(item, dict):
                norm_factors.append({
                    "label": item.get("label") or item.get("name") or f"Factor {idx+1}",
                    "percentage": int(item.get("percentage") or item.get("impact_pct") or 20),
                    "bar_class": item.get("bar_class") or bar_classes[idx % len(bar_classes)]
                })
            elif isinstance(item, str):
                norm_factors.append({
                    "label": item,
                    "percentage": 20,
                    "bar_class": bar_classes[idx % len(bar_classes)]
                })

    return {
        "key_insights": norm_insights,
        "factor_breakdown": norm_factors
    }


class LLMService:
    def __init__(self):
        self.gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.openrouter_key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model = settings.OPENROUTER_MODEL or os.getenv("OPENROUTER_MODEL", "google/gemini-3.8-flash")
        self.openrouter_base_url = settings.OPENROUTER_BASE_URL or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def extract_decision_context(self, decision_prompt: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts structured context, domain, variables, and missing information from user's decision prompt using OpenRouter.
        """
        prompt_text = f"""
You are RealitySim AI, an advanced decision simulation reasoning system.
User Decision Question: "{decision_prompt}"
Additional Context: {json.dumps(user_info or {}, indent=2)}

Analyze the decision and return a JSON object with:
1. "decision": Concise restatement of the core decision.
2. "domain": Primary category (e.g. "Career", "Real Estate", "Startup", "Finance", "Education", "Relocation").
3. "objective": Primary goal or objective to maximize/minimize.
4. "time_horizon": Suggested simulation horizon in years (integer between 1 and 30, default 5).
5. "variables": Array of key extracted variables or parameters explicitly mentioned or implied.
6. "assumptions": Array of baseline assumptions identified.
7. "missing_information": Array of critical missing parameters needed for precise simulation (e.g. initial budget, monthly expense, risk appetite).
8. "key_priorities": Array of recommended user priorities.

Return ONLY raw valid JSON, no markdown fences or other text.
"""
        if self.openrouter_key:
            for model_cand in [self.openrouter_model, "google/gemini-2.5-flash", "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]:
                try:
                    import openai
                    client = openai.OpenAI(
                        base_url=self.openrouter_base_url,
                        api_key=self.openrouter_key
                    )
                    response = client.chat.completions.create(
                        model=model_cand,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    return clean_json_response(response.choices[0].message.content)
                except Exception as e:
                    logger.warning(f"OpenRouter context extraction ({model_cand}) failed: {e}")

        # Gemini API fallback
        if self.gemini_key:
            for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_text
                    )
                    return clean_json_response(response.text)
                except Exception as e:
                    logger.warning(f"Gemini context extraction ({model_name}) failed: {e}")

        # OpenAI API fallback
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.3,
                    max_tokens=1500
                )
                return clean_json_response(response.choices[0].message.content)
            except Exception as e:
                logger.warning(f"OpenAI context extraction failed: {e}")

        # Intelligent fallback
        return {
            "decision": decision_prompt,
            "domain": "Decision",
            "objective": "Balanced Risk-Adjusted Outcome",
            "time_horizon": 5,
            "variables": ["initial_capital", "expected_growth", "annual_costs"],
            "assumptions": ["Stable market baseline", "Predictable time horizon"],
            "missing_information": ["Current savings", "Risk tolerance", "Annual expenses"],
            "key_priorities": ["Capital preservation", "Growth potential"]
        }

    def generate_scenarios_from_decision(self, decision_prompt: str, additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Uses LLM to understand decision context, determine category, and generate 3 dynamic scenarios with tailored assumptions and mathematical variables.
        """
        prompt_text = f"""
You are RealitySim AI, an advanced decision simulation reasoning system.
User Decision Question: "{decision_prompt}"
Additional User Context: {json.dumps(additional_info or {}, indent=2)}

Analyze the decision and return a JSON object with:
1. "title": Short clean decision title (e.g., "Job vs Master's Degree", "Buy vs Rent House", "SaaS Startup Launch").
2. "category": Decision domain (e.g., "Career", "Real Estate", "Startup", "Finance", "Education", "Relocation").
3. "scenarios": An array of 3 distinct, realistic scenarios. Each scenario must contain:
   - "name": Concise name (e.g., "Job (Immediate)", "Master's Degree", "Job + Part-time Study", "Buy 3BHK Apartment", "Rent & Invest in Equities")
   - "description": 1-2 sentence explanation of this pathway.
   - "badge": Status badge ("Most Likely", "Higher Risk", "Balanced", "Conservative", "Aggressive", etc.)
   - "badge_color": "blue", "red", "green", or "purple"
   - "assumptions": List of 5 key measurable assumptions. Each assumption has:
       * "key": Label (e.g., "Starting Salary", "Total Cost", "Annual Growth Rate", "Market Stability", "Location", "Switch Probability")
       * "value": Human-readable formatted value (e.g., "₹ 6.0 LPA", "₹ 20.0 Lakhs", "8%", "Moderate", "India", "30%")
       * "icon": Single emoji icon (e.g., "💼", "🎓", "📈", "🛡️", "📍", "⏱️", "⚖️", "🏠", "💰")
       * "numeric_val": Raw numeric value for computation
       * "param_key": normalized variable key (e.g. "initial_income", "initial_investment", "growth_rate_pct", "success_probability_pct")
   - "variables": Baseline simulation parameters:
       * "base_cashflow_annual": Annual baseline cashflow/income
       * "growth_mean_pct": Mean growth percentage (e.g. 8.0)
       * "growth_std_pct": Standard deviation of growth (e.g. 3.0)
       * "upfront_cost": Initial capital outlay / education cost (e.g. 2000000)
       * "annual_cost": Recurring cost / living differential
       * "success_rate_pct": Probability of baseline trajectory success (e.g. 75.0)
       * "volatility_index": 0.1 to 0.8
       * "downside_risk_pct": 5.0 to 40.0

Return ONLY raw valid JSON, no markdown fences or other text.
"""
        # 1. Try OpenRouter API if key provided
        if self.openrouter_key:
            for model_cand in [self.openrouter_model, "google/gemini-2.5-flash", "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]:
                try:
                    import openai
                    client = openai.OpenAI(
                        base_url=self.openrouter_base_url,
                        api_key=self.openrouter_key
                    )
                    response = client.chat.completions.create(
                        model=model_cand,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.3,
                        max_tokens=2500
                    )
                    parsed = clean_json_response(response.choices[0].message.content)
                    return normalize_scenario_output(parsed, additional_info)
                except Exception as e:
                    logger.warning(f"OpenRouter API call ({model_cand}) failed, attempting fallback: {e}")

        # 2. Try Gemini API if key provided
        if self.gemini_key:
            for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_text
                    )
                    parsed = clean_json_response(response.text)
                    return normalize_scenario_output(parsed, additional_info)
                except Exception as e:
                    logger.warning(f"Gemini API call ({model_name}) failed, attempting next: {e}")

        # 3. Try OpenAI API if key provided
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.3,
                    max_tokens=2500
                )
                parsed = clean_json_response(response.choices[0].message.content)
                return normalize_scenario_output(parsed, additional_info)
            except Exception as e:
                logger.warning(f"OpenAI API call failed, attempting fallback: {e}")

        # High-Fidelity Intelligent Heuristic Reasoning Engine
        return self._heuristic_scenario_generator(decision_prompt, additional_info)

    def generate_explainable_insights(self, decision_title: str, scenario_results: List[Dict[str, Any]], additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates explainable natural-language insights and influencing factor breakdown based on actual numerical simulation results.
        """
        # Sort scenarios by expected outcome
        sorted_scenarios = sorted(scenario_results, key=lambda s: s.get("expected_outcome", 0), reverse=True)
        best_scenario = sorted_scenarios[0] if sorted_scenarios else {}
        safest_scenario = min(scenario_results, key=lambda s: s.get("risk_score", 50)) if scenario_results else {}
        riskiest_scenario = max(scenario_results, key=lambda s: s.get("risk_score", 50)) if scenario_results else {}

        prompt_text = f"""
Decision: "{decision_title}"
Simulated Scenarios and Actual Monte Carlo Results:
{json.dumps(scenario_results, indent=2)}

Generate:
1. "key_insights": 4 crisp, actionable, explainable insights (array of objects with "text" and "icon_type": "green"|"purple"|"blue"|"orange").
2. "factor_breakdown": 5 key influencing factors and their relative weight percentage totaling 80-100% (e.g. Salary Growth Rate, Market Conditions, Education Cost, Living Expenses, Execution Risk) with "label", "percentage", and "bar_class": "bar-blue"|"bar-indigo"|"bar-purple"|"bar-coral"|"bar-orange".

Return ONLY raw valid JSON.
"""
        # 1. Try OpenRouter API if key provided
        if self.openrouter_key:
            for model_cand in [self.openrouter_model, "google/gemini-2.5-flash", "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]:
                try:
                    import openai
                    client = openai.OpenAI(
                        base_url=self.openrouter_base_url,
                        api_key=self.openrouter_key
                    )
                    response = client.chat.completions.create(
                        model=model_cand,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    parsed = clean_json_response(response.choices[0].message.content)
                    return normalize_insights_output(parsed)
                except Exception as e:
                    logger.warning(f"OpenRouter insight generation ({model_cand}) failed: {e}")

        # 2. Try Gemini API if key provided
        if self.gemini_key:
            for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_text
                    )
                    parsed = clean_json_response(response.text)
                    return normalize_insights_output(parsed)
                except Exception as e:
                    logger.warning(f"Gemini insight generation ({model_name}) failed: {e}")

        # 3. Try OpenAI API if key provided
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.3,
                    max_tokens=2000
                )
                parsed = clean_json_response(response.choices[0].message.content)
                return normalize_insights_output(parsed)
            except Exception as e:
                logger.warning(f"OpenAI insight generation failed: {e}")

        # 2. Try Gemini API if key provided
        if self.gemini_key:
            for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_text
                    )
                    text = response.text.strip()
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                    return json.loads(text)
                except Exception as e:
                    logger.warning(f"Gemini insight generation ({model_name}) failed: {e}")

        # 3. Try OpenAI API if key provided
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.3
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)
            except Exception as e:
                logger.warning(f"OpenAI insight generation failed: {e}")

        # Dynamic heuristic insights based on real calculated metrics
        best_name = best_scenario.get("name", "Leading option")
        riskiest_name = riskiest_scenario.get("name", "Higher risk option")
        safest_name = safest_scenario.get("name", "Conservative option")

        return {
            "key_insights": [
                {
                    "text": f"{best_name} delivers the highest simulated financial outcome over the time horizon.",
                    "icon_type": "green"
                },
                {
                    "text": f"{riskiest_name} demonstrates higher risk variance due to upfront commitment and market volatility.",
                    "icon_type": "purple"
                },
                {
                    "text": f"{safest_name} has more stable outcome clustering with lower downside exposure.",
                    "icon_type": "blue"
                },
                {
                    "text": "Results incorporate compound growth, inflation adjustments, and Monte Carlo probability distributions.",
                    "icon_type": "orange"
                }
            ],
            "factor_breakdown": [
                {"label": "Growth Rate Variance", "percentage": 28, "bar_class": "bar-blue"},
                {"label": "Opportunity Success Rate", "percentage": 22, "bar_class": "bar-indigo"},
                {"label": "Upfront Capital Outlay", "percentage": 18, "bar_class": "bar-purple"},
                {"label": "Market Conditions", "percentage": 15, "bar_class": "bar-coral"},
                {"label": "Ongoing Living/Operating Costs", "percentage": 10, "bar_class": "bar-orange"}
            ]
        }

    def _heuristic_scenario_generator(self, prompt: str, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Intelligent contextual scenario synthesizer for various decision archetypes."""
        p_lower = prompt.lower()
        info = info or {}
        exp_income = info.get("expected_annual_income", 600000.0)
        savings = info.get("current_savings", 50000.0)
        horizon_str = info.get("time_horizon", "5 years")

        # 1. Real Estate: Buy vs Rent
        if any(w in p_lower for w in ["buy", "rent", "house", "apartment", "property", "flat", "real estate"]):
            return {
                "title": "Buy House vs Continue Renting",
                "category": "Real Estate",
                "scenarios": [
                    {
                        "name": "Buy Property (Mortgage)",
                        "description": "Purchase residential property with down payment and 20-year home loan amortized over simulation horizon.",
                        "badge": "Higher Equity",
                        "badge_color": "blue",
                        "assumptions": [
                            {"key": "Property Value", "value": "₹ 75.0 Lakhs", "icon": "🏠", "numeric_val": 7500000, "param_key": "upfront_cost"},
                            {"key": "Down Payment", "value": "20% (₹15L)", "icon": "💰", "numeric_val": 1500000, "param_key": "down_payment"},
                            {"key": "Property Appreciation", "value": "6.5% p.a.", "icon": "📈", "numeric_val": 6.5, "param_key": "growth_mean_pct"},
                            {"key": "Home Loan Rate", "value": "8.5%", "icon": "🏦", "numeric_val": 8.5, "param_key": "interest_rate"},
                            {"key": "Maintenance & Tax", "value": "₹ 50,000 / yr", "icon": "🛠️", "numeric_val": 50000, "param_key": "annual_cost"}
                        ],
                        "variables": {
                            "base_cashflow_annual": exp_income * 0.8,
                            "growth_mean_pct": 6.5,
                            "growth_std_pct": 2.5,
                            "upfront_cost": 1500000,
                            "annual_cost": 650000,
                            "success_rate_pct": 85.0,
                            "volatility_index": 0.25,
                            "downside_risk_pct": 18.0
                        }
                    },
                    {
                        "name": "Continue Renting & Invest Surplus",
                        "description": "Rent a suitable home and actively invest the down payment and EMI differential into a balanced equity/debt portfolio.",
                        "badge": "High Liquidity",
                        "badge_color": "green",
                        "assumptions": [
                            {"key": "Monthly Rent", "value": "₹ 28,000 / mo", "icon": "🏢", "numeric_val": 336000, "param_key": "annual_rent"},
                            {"key": "Rent Inflation", "value": "5.0% p.a.", "icon": "📊", "numeric_val": 5.0, "param_key": "rent_inflation"},
                            {"key": "Portfolio Return", "value": "11.5% p.a.", "icon": "📈", "numeric_val": 11.5, "param_key": "growth_mean_pct"},
                            {"key": "Initial Invested Capital", "value": "₹ 15.0 Lakhs", "icon": "💵", "numeric_val": 1500000, "param_key": "initial_invested"},
                            {"key": "Liquidity Flexibility", "value": "High", "icon": "⚡", "numeric_val": 90.0, "param_key": "liquidity"}
                        ],
                        "variables": {
                            "base_cashflow_annual": exp_income,
                            "growth_mean_pct": 11.5,
                            "growth_std_pct": 4.5,
                            "upfront_cost": 0,
                            "annual_cost": 336000,
                            "success_rate_pct": 80.0,
                            "volatility_index": 0.35,
                            "downside_risk_pct": 12.0
                        }
                    },
                    {
                        "name": "Buy Lower-Cost Tier-2 Property",
                        "description": "Acquire an emerging suburban or tier-2 asset with low leverage and high rental yield potential.",
                        "badge": "Balanced",
                        "badge_color": "purple",
                        "assumptions": [
                            {"key": "Property Value", "value": "₹ 40.0 Lakhs", "icon": "🏡", "numeric_val": 4000000, "param_key": "upfront_cost"},
                            {"key": "Down Payment", "value": "₹ 8.0 Lakhs", "icon": "💰", "numeric_val": 800000, "param_key": "down_payment"},
                            {"key": "Rental Yield", "value": "4.2% p.a.", "icon": "📈", "numeric_val": 4.2, "param_key": "rental_yield"},
                            {"key": "Appreciation Potential", "value": "8.0% p.a.", "icon": "🚀", "numeric_val": 8.0, "param_key": "growth_mean_pct"},
                            {"key": "Tenant Occupancy", "value": "90%", "icon": "🔑", "numeric_val": 90.0, "param_key": "occupancy"}
                        ],
                        "variables": {
                            "base_cashflow_annual": exp_income * 0.9,
                            "growth_mean_pct": 8.0,
                            "growth_std_pct": 3.5,
                            "upfront_cost": 800000,
                            "annual_cost": 280000,
                            "success_rate_pct": 78.0,
                            "volatility_index": 0.30,
                            "downside_risk_pct": 15.0
                        }
                    }
                ]
            }

        # 2. Business / Startup / Entrepreneurship
        elif any(w in p_lower for w in ["startup", "business", "launch", "restaurant", "store", "company", "venture", "freelance"]):
            return {
                "title": "Startup Launch vs Traditional Employment",
                "category": "Entrepreneurship",
                "scenarios": [
                    {
                        "name": "Bootstrapped / Lean Launch",
                        "description": "Start with low initial capital, validate product-market fit, and scale sustainably from early cash flows.",
                        "badge": "Controlled Risk",
                        "badge_color": "green",
                        "assumptions": [
                            {"key": "Initial Capital", "value": "₹ 5.0 Lakhs", "icon": "💰", "numeric_val": 500000, "param_key": "upfront_cost"},
                            {"key": "Break-even Timeline", "value": "9 Months", "icon": "⏱️", "numeric_val": 9, "param_key": "breakeven_months"},
                            {"key": "Year 2 Revenue Potential", "value": "₹ 18.0 LPA", "icon": "📈", "numeric_val": 1800000, "param_key": "growth_mean_pct"},
                            {"key": "Survival Rate", "value": "70%", "icon": "🛡️", "numeric_val": 70.0, "param_key": "survival_rate"},
                            {"key": "Founder Equity Retained", "value": "100%", "icon": "👑", "numeric_val": 100.0, "param_key": "equity"}
                        ],
                        "variables": {
                            "base_cashflow_annual": 800000,
                            "growth_mean_pct": 25.0,
                            "growth_std_pct": 15.0,
                            "upfront_cost": 500000,
                            "annual_cost": 200000,
                            "success_rate_pct": 68.0,
                            "volatility_index": 0.55,
                            "downside_risk_pct": 25.0
                        }
                    },
                    {
                        "name": "Full-Scale Funded Launch",
                        "description": "Raise external seed capital to aggressively capture market share and scale infrastructure rapidly.",
                        "badge": "High Upside",
                        "badge_color": "red",
                        "assumptions": [
                            {"key": "Seed Funding", "value": "₹ 50.0 Lakhs", "icon": "🚀", "numeric_val": 5000000, "param_key": "funding"},
                            {"key": "Monthly Burn Rate", "value": "₹ 2.5 L / mo", "icon": "🔥", "numeric_val": 3000000, "param_key": "annual_cost"},
                            {"key": "Hyper-growth Potential", "value": "60% CAGR", "icon": "📈", "numeric_val": 60.0, "param_key": "growth_mean_pct"},
                            {"key": "Market Execution Risk", "value": "High", "icon": "⚡", "numeric_val": 60.0, "param_key": "risk"},
                            {"key": "Series A Probability", "value": "35%", "icon": "🎯", "numeric_val": 35.0, "param_key": "series_a"}
                        ],
                        "variables": {
                            "base_cashflow_annual": 1200000,
                            "growth_mean_pct": 45.0,
                            "growth_std_pct": 30.0,
                            "upfront_cost": 1000000,
                            "annual_cost": 400000,
                            "success_rate_pct": 42.0,
                            "volatility_index": 0.75,
                            "downside_risk_pct": 40.0
                        }
                    },
                    {
                        "name": "Part-Time Agency / Freelancing",
                        "description": "Retain stable core income while building high-margin consulting or client projects on the side.",
                        "badge": "Balanced",
                        "badge_color": "blue",
                        "assumptions": [
                            {"key": "Primary Income Retention", "value": "100% (₹6.0 LPA)", "icon": "💼", "numeric_val": 600000, "param_key": "base_income"},
                            {"key": "Side Revenue Added", "value": "₹ 3.0 L / yr", "icon": "💵", "numeric_val": 300000, "param_key": "side_income"},
                            {"key": "Initial Overhead", "value": "₹ 50,000", "icon": "💻", "numeric_val": 50000, "param_key": "upfront_cost"},
                            {"key": "Burnout Probability", "value": "Low/Moderate", "icon": "⚖️", "numeric_val": 25.0, "param_key": "burnout"},
                            {"key": "Compounding Client Base", "value": "15% p.a.", "icon": "📈", "numeric_val": 15.0, "param_key": "growth_mean_pct"}
                        ],
                        "variables": {
                            "base_cashflow_annual": exp_income + 300000,
                            "growth_mean_pct": 12.0,
                            "growth_std_pct": 5.0,
                            "upfront_cost": 50000,
                            "annual_cost": 60000,
                            "success_rate_pct": 88.0,
                            "volatility_index": 0.22,
                            "downside_risk_pct": 8.0
                        }
                    }
                ]
            }

        # 3. Default: Career & Higher Education
        return {
            "title": "Job vs Master's Degree",
            "category": "Career",
            "scenarios": [
                {
                    "name": "Job (Immediate)",
                    "description": "Start a software job immediately after graduation and gain continuous industry experience and compounding salary.",
                    "badge": "Most Likely",
                    "badge_color": "blue",
                    "assumptions": [
                        {"key": "Starting Salary", "value": f"₹ {exp_income/100000:.1f} LPA", "icon": "💼", "numeric_val": exp_income, "param_key": "starting_salary"},
                        {"key": "Annual Growth Rate", "value": "8%", "icon": "📈", "numeric_val": 8.0, "param_key": "growth_mean_pct"},
                        {"key": "Job Market Stability", "value": "Moderate", "icon": "🛡️", "numeric_val": 75.0, "param_key": "stability"},
                        {"key": "Location", "value": info.get("location", "India"), "icon": "📍", "numeric_val": 0, "param_key": "location"},
                        {"key": "Switch Probability (per year)", "value": "30%", "icon": "🔄", "numeric_val": 30.0, "param_key": "switch_probability"}
                    ],
                    "variables": {
                        "base_cashflow_annual": exp_income,
                        "growth_mean_pct": 8.0,
                        "growth_std_pct": 2.8,
                        "upfront_cost": 0,
                        "annual_cost": info.get("monthly_expenses", 15000) * 12,
                        "success_rate_pct": 82.0,
                        "volatility_index": 0.20,
                        "downside_risk_pct": 10.0
                    }
                },
                {
                    "name": "Master's Degree",
                    "description": "Pursue a 2-year specialized Master's program with higher upfront education costs for accelerated post-graduation compensation.",
                    "badge": "Higher Risk",
                    "badge_color": "red",
                    "assumptions": [
                        {"key": "Total Education Cost", "value": "₹ 20.0 Lakhs", "icon": "🎓", "numeric_val": 2000000, "param_key": "upfront_cost"},
                        {"key": "Program Duration", "value": "2 years", "icon": "⏱️", "numeric_val": 2, "param_key": "duration_years"},
                        {"key": "Post-Study Salary (Intl.)", "value": "₹ 45.0 LPA", "icon": "💼", "numeric_val": 4500000, "param_key": "post_salary"},
                        {"key": "Visa/Work Probability", "value": "70%", "icon": "🛂", "numeric_val": 70.0, "param_key": "visa_rate"},
                        {"key": "Location", "value": "USA/Canada", "icon": "📍", "numeric_val": 0, "param_key": "location"}
                    ],
                    "variables": {
                        "base_cashflow_annual": 2500000,
                        "growth_mean_pct": 16.0,
                        "growth_std_pct": 7.5,
                        "upfront_cost": 2000000,
                        "annual_cost": 300000,
                        "success_rate_pct": 64.0,
                        "volatility_index": 0.48,
                        "downside_risk_pct": 28.0
                    }
                },
                {
                    "name": "Job + Part-time Study",
                    "description": "Take a core engineering position while pursuing modular online/executive master's degree for balanced career growth.",
                    "badge": "Balanced",
                    "badge_color": "green",
                    "assumptions": [
                        {"key": "Starting Salary", "value": f"₹ {max(4.5, exp_income/100000 - 0.5):.1f} LPA", "icon": "💼", "numeric_val": exp_income * 0.9, "param_key": "starting_salary"},
                        {"key": "Part-time Study Cost", "value": "₹ 3.0 Lakhs", "icon": "🎓", "numeric_val": 300000, "param_key": "upfront_cost"},
                        {"key": "Annual Growth Rate", "value": "7%", "icon": "📈", "numeric_val": 7.0, "param_key": "growth_mean_pct"},
                        {"key": "Study Duration", "value": "3 years", "icon": "⏱️", "numeric_val": 3, "param_key": "duration_years"},
                        {"key": "Work-Study Balance", "value": "Manageable", "icon": "⚖️", "numeric_val": 80.0, "param_key": "balance"}
                    ],
                    "variables": {
                        "base_cashflow_annual": exp_income * 0.95,
                        "growth_mean_pct": 10.5,
                        "growth_std_pct": 3.2,
                        "upfront_cost": 300000,
                        "annual_cost": info.get("monthly_expenses", 15000) * 12,
                        "success_rate_pct": 78.0,
                        "volatility_index": 0.24,
                        "downside_risk_pct": 14.0
                    }
                }
            ]
        }

llm_service = LLMService()
