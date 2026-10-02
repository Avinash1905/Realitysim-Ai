import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    decision_prompt = Column(Text, nullable=False)
    title = Column(String(255), nullable=True, default="Simulation")
    category = Column(String(100), nullable=True, default="Decision")
    status = Column(String(50), default="created")  # created, info_provided, scenarios_generated, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    additional_info = relationship("AdditionalInfo", back_populates="simulation", uselist=False, cascade="all, delete-orphan")
    scenarios = relationship("Scenario", back_populates="simulation", cascade="all, delete-orphan", order_by="Scenario.order_idx")
    results = relationship("SimulationResult", back_populates="simulation", uselist=False, cascade="all, delete-orphan")


class AdditionalInfo(Base):
    __tablename__ = "additional_information"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    simulation_id = Column(String(36), ForeignKey("simulations.id"), unique=True, nullable=False)
    
    age = Column(Integer, default=21)
    education_level = Column(String(100), default="B.Tech (CSE - AI)")
    location = Column(String(100), default="India")
    work_experience = Column(String(100), default="Fresher (0 years)")
    
    current_savings = Column(Float, default=50000.0)
    expected_annual_income = Column(Float, default=600000.0)
    monthly_expenses = Column(Float, default=15000.0)
    education_cost = Column(Float, default=2000000.0)
    investment_capacity = Column(Float, default=5000.0)
    
    risk_tolerance = Column(String(50), default="Balanced")
    time_horizon = Column(String(50), default="5 years")
    decision_priority = Column(String(100), default="Balanced Growth")
    
    goals = Column(JSON, default=list)  # ["High Income", "Career Growth", ...]
    additional_context = Column(Text, nullable=True)
    external_factors = Column(Text, nullable=True)

    simulation = relationship("Simulation", back_populates="additional_info")


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    simulation_id = Column(String(36), ForeignKey("simulations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    badge = Column(String(50), default="Standard")  # Most Likely, Higher Risk, Balanced, Recommended, etc.
    badge_color = Column(String(50), default="blue") # blue, red, green, purple
    order_idx = Column(Integer, default=0)
    is_custom = Column(Boolean, default=False)
    
    # Statistical Outputs
    probability = Column(Float, default=0.0)  # e.g., 72.0 (%)
    risk_level = Column(String(50), default="Medium")  # Low, Medium, High
    risk_score = Column(Float, default=50.0)  # 0 to 100
    expected_outcome = Column(Float, default=0.0)  # in base units / INR
    median_outcome = Column(Float, default=0.0)
    
    # Assumptions list: [{"key": "Starting Salary", "value": "₹ 6.0 LPA", "icon": "💼", "color": "icon-blue", "numeric_val": 600000, "param_key": "starting_salary"}]
    assumptions = Column(JSON, default=list)
    
    # Mathematical simulation parameters
    variables = Column(JSON, default=dict)

    simulation = relationship("Simulation", back_populates="scenarios")


class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    simulation_id = Column(String(36), ForeignKey("simulations.id"), unique=True, nullable=False)
    
    expected_outcome_formatted = Column(String(100), default="₹ 0.0 L")
    median_outcome_formatted = Column(String(100), default="₹ 0.0 L")
    trend_formatted = Column(String(100), default="+0% vs current")
    
    risk_level = Column(String(50), default="Medium")
    risk_score = Column(Float, default=50.0)
    confidence_pct = Column(Integer, default=75)
    confidence_subtext = Column(String(255), default="Based on input context and Monte Carlo variance")
    
    runs_count = Column(Integer, default=10000)
    
    # Distribution data for dynamic SVG graph
    distribution_data = Column(JSON, default=dict)
    
    # Comparison table data
    scenario_comparisons = Column(JSON, default=list)
    
    # Explainable key insights
    key_insights = Column(JSON, default=list)
    
    # Factors breakdown (Why this result?)
    factor_breakdown = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation = relationship("Simulation", back_populates="results")
