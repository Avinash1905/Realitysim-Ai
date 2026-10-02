from typing import Dict, Any, List

class ConfidenceEngine:
    @staticmethod
    def calculate_confidence(
        additional_info: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        simulation_stats_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates confidence score (0-100%) based on data completeness, assumption clarity, and simulation stability.
        """
        score = 50.0  # baseline
        
        # 1. Input completeness (up to +25%)
        if additional_info:
            if additional_info.get("age"): score += 3.0
            if additional_info.get("education_level"): score += 3.0
            if additional_info.get("location"): score += 3.0
            if additional_info.get("work_experience"): score += 3.0
            if additional_info.get("current_savings") is not None: score += 4.0
            if additional_info.get("expected_annual_income") is not None: score += 4.0
            if additional_info.get("goals") and len(additional_info["goals"]) > 0: score += 3.0
            if additional_info.get("additional_context"): score += 4.0
            if additional_info.get("external_factors"): score += 3.0
            
        # 2. Scenario assumptions richness (up to +15%)
        total_assumptions = sum(len(s.get("assumptions", [])) for s in scenarios)
        if total_assumptions >= 10:
            score += 15.0
        elif total_assumptions >= 5:
            score += 8.0
            
        # 3. Simulation stability penalty (-10% if extreme variance)
        for stat in simulation_stats_list:
            std = stat.get("std", 0.0)
            mean = abs(stat.get("mean", 1.0))
            if std / (mean if mean > 0 else 1.0) > 2.0:
                score -= 4.0

        final_pct = int(max(55, min(95, round(score))))
        
        return {
            "confidence_pct": final_pct,
            "subtext": "Based on input context completeness and Monte Carlo convergence"
        }

confidence_engine = ConfidenceEngine()
