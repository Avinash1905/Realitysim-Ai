import numpy as np
from typing import Dict, Any, List, Tuple

class MonteCarloEngine:
    def __init__(self, default_runs: int = 10000):
        self.default_runs = default_runs

    def run_scenario_simulation(
        self,
        scenario_data: Dict[str, Any],
        time_horizon_years: int = 5,
        runs: int = 10000,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Runs a real Monte Carlo simulation (10,000 iterations) for a given scenario.
        Returns full statistical distribution and summary metrics.
        """
        additional_info = additional_info or {}
        variables = scenario_data.get("variables", {}) or {}
        
        # Base variables
        base_cashflow = float(variables.get("base_cashflow_annual", 600000.0))
        growth_mean = float(variables.get("growth_mean_pct", 8.0)) / 100.0
        growth_std = float(variables.get("growth_std_pct", 3.0)) / 100.0
        upfront_cost = float(variables.get("upfront_cost", 0.0))
        annual_cost = float(variables.get("annual_cost", 180000.0))
        success_rate = float(variables.get("success_rate_pct", 75.0)) / 100.0
        volatility = float(variables.get("volatility_index", 0.25))
        downside_risk = float(variables.get("downside_risk_pct", 15.0)) / 100.0
        
        # Initial savings
        initial_savings = float(additional_info.get("current_savings", 50000.0))
        
        np.random.seed(None) # fresh entropy
        
        # 1. Generate stochastic growth matrices: shape (runs, time_horizon_years)
        annual_growths = np.random.normal(loc=growth_mean, scale=growth_std, size=(runs, time_horizon_years))
        
        # 2. Market shock / downside simulation
        shock_occurs = np.random.binomial(1, downside_risk, size=(runs, time_horizon_years))
        shock_penalties = np.random.uniform(0.1, 0.35, size=(runs, time_horizon_years))
        effective_growths = annual_growths - (shock_occurs * shock_penalties)

        # 3. Simulate year-by-year cashflow compounding
        cumulative_wealth = np.zeros(runs)
        initial_net_wealth = initial_savings - upfront_cost
        
        current_incomes = np.full(runs, base_cashflow)
        
        for yr in range(time_horizon_years):
            # Apply success probability filter for major milestones
            success_factor = np.random.binomial(1, success_rate, size=runs) * 0.4 + 0.8
            current_incomes = current_incomes * (1.0 + effective_growths[:, yr]) * success_factor
            
            # Net annual savings
            net_annual = current_incomes - annual_cost
            cumulative_wealth += net_annual

        total_financial_outcomes = initial_net_wealth + cumulative_wealth
        
        # Floor outcomes to avoid unrealistically negative numbers if debt is capped
        total_financial_outcomes = np.maximum(total_financial_outcomes, -upfront_cost * 1.2)

        # 4. Compute exact descriptive statistics
        mean_val = float(np.mean(total_financial_outcomes))
        median_val = float(np.median(total_financial_outcomes))
        std_val = float(np.std(total_financial_outcomes))
        min_val = float(np.min(total_financial_outcomes))
        max_val = float(np.max(total_financial_outcomes))
        
        p10 = float(np.percentile(total_financial_outcomes, 10))
        p25 = float(np.percentile(total_financial_outcomes, 25))
        p50 = float(np.percentile(total_financial_outcomes, 50))
        p75 = float(np.percentile(total_financial_outcomes, 75))
        p90 = float(np.percentile(total_financial_outcomes, 90))

        prob_positive = float(np.mean(total_financial_outcomes > 0) * 100.0)

        # 5. Generate density curve points for SVG Bell Curve graph
        hist_counts, bin_edges = np.histogram(total_financial_outcomes, bins=50, density=False)
        frequencies = (hist_counts / runs * 100.0).tolist()
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()

        return {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "prob_positive": prob_positive,
            "raw_samples": total_financial_outcomes,
            "histogram": {
                "bin_centers": bin_centers,
                "frequencies": frequencies
            }
        }

    @staticmethod
    def format_currency_inr(val: float) -> str:
        """Formats INR into Lakhs / Crores (e.g. ₹ 28.4 L or ₹ 1.2 Cr)."""
        if val >= 10000000:
            return f"₹ {val / 10000000:.2f} Cr"
        elif val >= 100000:
            return f"₹ {val / 100000:.1f} L"
        elif val <= -10000000:
            return f"-₹ {abs(val) / 10000000:.2f} Cr"
        elif val <= -100000:
            return f"-₹ {abs(val) / 100000:.1f} L"
        else:
            return f"₹ {val:,.0f}"

simulation_engine = MonteCarloEngine()
