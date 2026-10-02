from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 1. Simulation Initial Decision Input
class CreateSimulationRequest(BaseModel):
    decision_prompt: str

class SimulationResponse(BaseModel):
    id: str
    decision_prompt: str
    title: str
    category: str
    status: str
    created_at: Optional[Any] = None

# 2. Additional Information
class AdditionalInfoRequest(BaseModel):
    age: Optional[int] = 21
    education_level: Optional[str] = "B.Tech (CSE - AI)"
    location: Optional[str] = "India"
    work_experience: Optional[str] = "Fresher (0 years)"
    current_savings: Optional[float] = 50000.0
    expected_annual_income: Optional[float] = 600000.0
    monthly_expenses: Optional[float] = 15000.0
    education_cost: Optional[float] = 2000000.0
    investment_capacity: Optional[float] = 5000.0
    risk_tolerance: Optional[str] = "Balanced"
    time_horizon: Optional[str] = "5 years"
    decision_priority: Optional[str] = "Balanced Growth"
    goals: Optional[List[str]] = []
    additional_context: Optional[str] = None
    external_factors: Optional[str] = None

# 3. Scenario & Assumptions
class AssumptionItem(BaseModel):
    key: str
    value: str
    icon: Optional[str] = "📊"
    color: Optional[str] = "icon-blue"
    numeric_val: Optional[float] = None
    param_key: Optional[str] = None

class ScenarioResponse(BaseModel):
    id: str
    simulation_id: str
    name: str
    description: str
    badge: str
    badge_color: str
    order_idx: int
    is_custom: bool
    probability: float
    risk_level: str
    risk_score: float
    expected_outcome: float
    median_outcome: float
    assumptions: List[Dict[str, Any]]
    variables: Dict[str, Any]

class UpdateScenarioRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    assumptions: Optional[List[Dict[str, Any]]] = None
    variables: Optional[Dict[str, Any]] = None

class CreateCustomScenarioRequest(BaseModel):
    name: str
    description: str
    assumptions: Optional[List[Dict[str, Any]]] = []
    variables: Optional[Dict[str, Any]] = {}

# 4. Simulation Results & What-If
class DistributionPoint(BaseModel):
    value: float
    frequency: float

class ScenarioComparisonItem(BaseModel):
    id: str
    name: str
    subtitle: str
    badge: Optional[str] = None
    badge_type: Optional[str] = "green"
    expected_formatted: str
    probability_pct: int
    risk_level: str
    risk_class: str
    color: str

class KeyInsightItem(BaseModel):
    text: str
    icon_type: str # green, purple, blue, orange

class FactorBreakdownItem(BaseModel):
    label: str
    percentage: int
    bar_class: str

class SimulationResultResponse(BaseModel):
    simulation_id: str
    title: str
    category: str
    decision_prompt: str
    time_horizon: str
    runs_count: int
    created_at_formatted: str
    
    # KPI metrics
    expected_outcome: str
    median_outcome: str
    trend: str
    risk_level: str
    risk_score: float
    confidence: int
    confidence_subtext: str
    
    # Graph data (Gaussian bell curve & frequency points)
    distribution_data: Dict[str, Any]
    
    # Scenario comparison cards
    scenario_comparisons: List[Dict[str, Any]]
    
    # Insights & factors
    key_insights: List[Dict[str, Any]]
    factor_breakdown: List[Dict[str, Any]]

class WhatIfRequest(BaseModel):
    scenario_id: Optional[str] = None
    variable_updates: Dict[str, Any] # e.g. {"inflation": 8.0, "education_cost": 2500000}
