from typing import Dict, Any

class RiskEngine:
    @staticmethod
    def calculate_scenario_risk(simulation_stats: Dict[str, Any], variables: Dict[str, Any], additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculates mathematical risk score (0-100) and risk category based on Monte Carlo distribution variance, downside exposure, and capital requirements.
        """
        additional_info = additional_info or {}
        mean_val = simulation_stats.get("mean", 1.0)
        std_val = simulation_stats.get("std", 0.0)
        p10 = simulation_stats.get("p10", 0.0)
        p50 = simulation_stats.get("p50", 1.0)
        
        # 1. Coefficient of variation (relative volatility)
        cv = abs(std_val / (mean_val if abs(mean_val) > 1e-3 else 1.0))
        cv_component = min(cv * 35.0, 40.0)
        
        # 2. Downside spread (how far P10 drops below P50)
        downside_spread = max(0.0, (p50 - p10) / (abs(p50) + 100000.0))
        downside_component = min(downside_spread * 40.0, 35.0)
        
        # 3. Capital commitment risk
        upfront_cost = float(variables.get("upfront_cost", 0.0))
        savings = float(additional_info.get("current_savings", 50000.0))
        capital_ratio = upfront_cost / (savings + 200000.0)
        capital_component = min(capital_ratio * 15.0, 25.0)
        
        raw_score = cv_component + downside_component + capital_component
        risk_score = round(max(10.0, min(95.0, raw_score)), 1)
        
        if risk_score <= 35.0:
            risk_level = "Low"
            risk_class = "risk-low"
        elif risk_score <= 65.0:
            risk_level = "Medium"
            risk_class = "risk-med"
        else:
            risk_level = "High"
            risk_class = "risk-high"
            
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_class": risk_class
        }

risk_engine = RiskEngine()
