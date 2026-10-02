/**
 * RealitySim AI - Universal Simulation & LLM Engine
 * Runs client-side Monte Carlo simulations and connects to OpenRouter / Gemini LLMs.
 * Works seamlessly on static hosting (Netlify) and local servers.
 */

const RealitySimConfig = {
  getApiKey: () => {
    return localStorage.getItem('realitysim_api_key') || 
           localStorage.getItem('openrouter_api_key') || 
           localStorage.getItem('gemini_api_key') || 
           '';
  },
  openRouterUrl: 'https://openrouter.ai/api/v1/chat/completions',
  fallbackModels: [
    'liquid/lfm-2.5-2.6b:free',
    'google/gemma-4-26b-a4b-it:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'qwen/qwen3.8-27b:free',
    'inclusionai/ling-3.0-flash-sante:free'
  ],
  defaultHorizonYears: 5,
  defaultRuns: 10000
};

// ==========================================
// 1. Monte Carlo Statistical Math Engine
// ==========================================
class MonteCarloEngine {
  // Standard Normal random variable using Box-Muller transform
  static randomNormal(mean = 0, std = 1) {
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }

  static runScenario(scenario, horizonYears = 5, runs = 10000, userInfo = {}) {
    const vars = scenario.variables || {};
    const baseIncome = parseFloat(vars.base_cashflow_annual || userInfo.expected_annual_income || 600000);
    const growthMean = (parseFloat(vars.growth_mean_pct || 8.5)) / 100.0;
    const growthStd = (parseFloat(vars.growth_std_pct || 3.5)) / 100.0;
    const upfrontCost = parseFloat(vars.upfront_cost || 0.0);
    const annualCost = parseFloat(vars.annual_cost || (userInfo.monthly_expenses ? userInfo.monthly_expenses * 12 : 180000));
    const successRate = (parseFloat(vars.success_rate_pct || 80.0)) / 100.0;
    const downsideRisk = (parseFloat(vars.downside_risk_pct || 15.0)) / 100.0;
    const initialSavings = parseFloat(userInfo.current_savings || 50000);

    const outcomes = new Float64Array(runs);
    const initialNetWealth = initialSavings - upfrontCost;

    for (let i = 0; i < runs; i++) {
      let income = baseIncome;
      let cumWealth = 0;

      for (let yr = 0; yr < horizonYears; yr++) {
        const shock = Math.random() < downsideRisk ? (0.10 + Math.random() * 0.25) : 0;
        const g = this.randomNormal(growthMean, growthStd) - shock;
        const success = (Math.random() < successRate ? 1 : 0) * 0.4 + 0.8;
        income = income * (1.0 + g) * success;
        const netAnnual = income - annualCost;
        cumWealth += netAnnual;
      }

      let total = initialNetWealth + cumWealth;
      if (upfrontCost > 0 && total < -upfrontCost * 1.2) {
        total = -upfrontCost * 1.2;
      }
      outcomes[i] = total;
    }

    outcomes.sort();

    // Summary Statistics
    let sum = 0;
    let positiveCount = 0;
    for (let i = 0; i < runs; i++) {
      sum += outcomes[i];
      if (outcomes[i] > 0) positiveCount++;
    }

    const mean = sum / runs;
    const median = outcomes[Math.floor(runs * 0.50)];
    const p10 = outcomes[Math.floor(runs * 0.10)];
    const p25 = outcomes[Math.floor(runs * 0.25)];
    const p50 = median;
    const p75 = outcomes[Math.floor(runs * 0.75)];
    const p90 = outcomes[Math.floor(runs * 0.90)];
    const minVal = outcomes[0];
    const maxVal = outcomes[runs - 1];
    const probPositive = (positiveCount / runs) * 100.0;

    // Variance & Std
    let varianceSum = 0;
    for (let i = 0; i < runs; i++) {
      const diff = outcomes[i] - mean;
      varianceSum += diff * diff;
    }
    const std = Math.sqrt(varianceSum / runs);

    // 50-bin Histogram for Bell Curve SVG
    const bins = 50;
    const binWidth = (p90 - p10) > 0 ? (p90 - p10) / (bins - 4) : 10000;
    const startBin = p10 - 2 * binWidth;
    const binCenters = [];
    const binCounts = new Array(bins).fill(0);

    for (let b = 0; b < bins; b++) {
      binCenters.push(startBin + (b + 0.5) * binWidth);
    }

    for (let i = 0; i < runs; i++) {
      const val = outcomes[i];
      const binIdx = Math.floor((val - startBin) / binWidth);
      if (binIdx >= 0 && binIdx < bins) {
        binCounts[binIdx]++;
      }
    }

    const frequencies = binCounts.map(c => (c / runs) * 100.0);

    return {
      mean,
      median,
      std,
      min: minVal,
      max: maxVal,
      p10,
      p25,
      p50,
      p75,
      p90,
      prob_positive: probPositive,
      histogram: {
        bin_centers: binCenters,
        frequencies: frequencies
      }
    };
  }

  static formatINR(val) {
    if (val === null || val === undefined || isNaN(val)) return '₹ 0';
    const isNeg = val < 0;
    const absVal = Math.abs(val);
    let str = '';

    if (absVal >= 10000000) {
      str = `₹ ${(absVal / 10000000).toFixed(2)} Cr`;
    } else if (absVal >= 100000) {
      str = `₹ ${(absVal / 100000).toFixed(1)} L`;
    } else if (absVal >= 1000) {
      str = `₹ ${(absVal / 1000).toFixed(1)} K`;
    } else {
      str = `₹ ${Math.round(absVal)}`;
    }

    return isNeg ? `-${str}` : str;
  }
}

// ==========================================
// 2. Realistic LLM & Domain Scenario Service
// ==========================================
class RealitySimAIService {
  static getApiKey() {
    return RealitySimConfig.getApiKey();
  }

  static setApiKey(key) {
    localStorage.setItem('realitysim_api_key', key);
  }

  static async callLLM(systemPrompt, userPrompt) {
    const apiKey = this.getApiKey();
    if (!apiKey) {
      throw new Error("No API key configured");
    }

    for (const model of RealitySimConfig.fallbackModels) {
      try {
        const res = await fetch(RealitySimConfig.openRouterUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
            'HTTP-Referer': window.location.origin || 'https://realitysim-ai.netlify.app',
            'X-Title': 'RealitySim AI'
          },
          body: JSON.stringify({
            model: model,
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userPrompt }
            ],
            temperature: 0.7
          })
        });

        if (res.ok) {
          const data = await res.json();
          const content = data.choices?.[0]?.message?.content;
          if (content) return content;
        }
      } catch (err) {
        console.warn(`Model ${model} failed, trying fallback...`, err);
      }
    }

    throw new Error("Could not reach LLM service");
  }

  static parseJSON(text) {
    text = text.trim();
    try {
      return JSON.parse(text);
    } catch (e) {}

    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (e) {}
    }
    throw new Error("Failed to parse JSON response");
  }

  static normalizeScenarios(data) {
    if (!data || !data.scenarios) return data;
    data.scenarios = data.scenarios.map((sc, idx) => {
      const prob = sc.probability || sc.probability_pct || sc.variables?.success_rate_pct || (idx === 0 ? 82 : (idx === 1 ? 65 : 75));
      return {
        id: sc.id || `sc-${idx + 1}`,
        name: sc.name || `Scenario ${idx + 1}`,
        badge: sc.badge || (idx === 0 ? 'Most Likely' : (idx === 1 ? 'Higher Risk' : 'Balanced')),
        badge_color: sc.badge_color || (idx === 0 ? 'blue' : (idx === 1 ? 'red' : 'green')),
        badge_type: sc.badge_type || (idx === 0 ? 'recommended' : (idx === 1 ? 'risky' : 'balanced')),
        description: sc.description || '',
        probability: prob,
        probability_pct: prob,
        assumptions: (sc.assumptions || []).map(a => ({
          icon: a.icon || '📌',
          label: a.label || a.key || 'Assumption',
          value: a.value || ''
        })),
        variables: {
          base_cashflow_annual: parseFloat(sc.variables?.base_cashflow_annual || 600000),
          growth_mean_pct: parseFloat(sc.variables?.growth_mean_pct || 10),
          growth_std_pct: parseFloat(sc.variables?.growth_std_pct || 3.5),
          upfront_cost: parseFloat(sc.variables?.upfront_cost || 0),
          annual_cost: parseFloat(sc.variables?.annual_cost || 180000),
          success_rate_pct: parseFloat(sc.variables?.success_rate_pct || prob),
          downside_risk_pct: parseFloat(sc.variables?.downside_risk_pct || 12),
          volatility_index: parseFloat(sc.variables?.volatility_index || 0.22)
        }
      };
    });
    return data;
  }

  static async generateScenarios(decisionPrompt, additionalInfo = {}) {
    const systemPrompt = `You are RealitySim AI, an advanced probabilistic decision modeling engine.
Analyze the user's decision and generate 3 distinct, hyper-realistic future scenarios tailored precisely to their topic.
Output ONLY valid JSON matching:
{
  "title": "Title of Decision",
  "category": "Domain Category",
  "scenarios": [
    {
      "name": "Specific Scenario Name",
      "badge": "Most Likely / Higher Risk / Balanced",
      "badge_color": "blue / red / green",
      "badge_type": "recommended / risky / balanced",
      "description": "Realistic outcome and roadmap description.",
      "probability": 80,
      "assumptions": [
        {"icon": "💼", "label": "Key Revenue / Salary", "value": "₹ 8.5 LPA"},
        {"icon": "📈", "label": "Yearly Growth", "value": "15%"},
        {"icon": "⏱️", "label": "Operating Effort", "value": "45 hrs/week"}
      ],
      "variables": {
        "base_cashflow_annual": 850000,
        "growth_mean_pct": 15,
        "growth_std_pct": 4.0,
        "upfront_cost": 150000,
        "annual_cost": 240000,
        "success_rate_pct": 80,
        "downside_risk_pct": 12,
        "volatility_index": 0.22
      }
    }
  ]
}`;

    const info = additionalInfo || {};
    const userPrompt = `Decision: "${decisionPrompt}"
Context & Profile:
- Age: ${info.age || 22}
- Education: ${info.education_level || 'B.Tech CSE'}
- Location: ${info.location || 'India'}
- Experience: ${info.work_experience || 'Fresher'}
- Current Savings: ₹ ${info.current_savings || 50000}
- Expected Income: ₹ ${info.expected_annual_income || 600000}
- Monthly Expenses: ₹ ${info.monthly_expenses || 15000}
- Time Horizon: ${info.time_horizon || '5 years'}
- Priority: ${info.decision_priority || 'Balanced Growth'}
- Goals: ${(info.goals || []).join(', ')}
- Additional Context: ${info.additional_context || 'None'}`;

    // 1. Try Netlify Serverless Function first (Uses secure Netlify Environment Variable)
    try {
      const netlifyRes = await fetch('/.netlify/functions/generate-scenarios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision_prompt: decisionPrompt, additional_info: additionalInfo })
      });
      if (netlifyRes.ok) {
        const netlifyData = await netlifyRes.json();
        if (netlifyData && netlifyData.scenarios && netlifyData.scenarios.length > 0) {
          return this.normalizeScenarios(netlifyData);
        }
      }
    } catch (e) {
      console.log("Netlify function not reachable:", e);
    }

    // 2. Try Client-Side LLM Call
    try {
      const rawText = await this.callLLM(systemPrompt, userPrompt);
      const parsed = this.parseJSON(rawText);
      if (parsed && parsed.scenarios && parsed.scenarios.length > 0) {
        return this.normalizeScenarios(parsed);
      }
    } catch (err) {
      console.warn("Using smart domain simulation generator:", err);
    }

    // 3. Dynamic Domain Generator (Provides domain-specific realism for any prompt)
    const fallback = this.generateSmartFallback(decisionPrompt, additionalInfo);
    return this.normalizeScenarios(fallback);
  }

  static generateSmartFallback(prompt, info = {}) {
    const p = (prompt || '').toLowerCase();
    const income = parseFloat(info.expected_annual_income || 600000);
    const savings = parseFloat(info.current_savings || 50000);
    const expenses = parseFloat(info.monthly_expenses ? info.monthly_expenses * 12 : 180000);

    // Domain 1: Tea Stall / Food / Kiosk / QSR
    if (p.includes('tea') || p.includes('chai') || p.includes('stall') || p.includes('cafe') || p.includes('restaurant') || p.includes('food')) {
      return {
        title: "Tea & Beverage Stall Business Simulation",
        category: "Retail Food & Beverage (QSR)",
        scenarios: [
          {
            id: 'sc-1',
            name: "High-Footfall Commercial Kiosk",
            badge: "Most Likely",
            badge_color: "blue",
            badge_type: "recommended",
            probability: 84,
            description: "Prime location near IT parks or colleges with 400+ cups/day sales and steady repeat footfall.",
            assumptions: [
              { icon: '☕', label: 'Daily Sales Volume', value: '420 cups @ ₹15-20/cup' },
              { icon: '💰', label: 'Monthly Net Profit', value: '₹ 62,000 / mo' },
              { icon: '📍', label: 'Setup & Rent Advance', value: '₹ 1.2 L upfront' }
            ],
            variables: {
              base_cashflow_annual: 744000,
              growth_mean_pct: 14.5,
              growth_std_pct: 3.5,
              upfront_cost: 120000,
              annual_cost: 160000,
              success_rate_pct: 84,
              downside_risk_pct: 12,
              volatility_index: 0.18
            }
          },
          {
            id: 'sc-2',
            name: "Multi-Outlet Franchise Expansion",
            badge: "Higher Risk",
            badge_color: "red",
            badge_type: "risky",
            probability: 62,
            description: "Aggressive multi-stall expansion with branded packaging, snacks menu, and staff hiring.",
            assumptions: [
              { icon: '🚀', label: 'Scale to 3 Outlets', value: 'Yr 2-3 Multi-Unit' },
              { icon: '📈', label: 'Annual Cash Flow Potential', value: '₹ 18.5 LPA' },
              { icon: '⚡', label: 'Capex & Equipment', value: '₹ 4.8 L initial' }
            ],
            variables: {
              base_cashflow_annual: 1450000,
              growth_mean_pct: 28.0,
              growth_std_pct: 8.0,
              upfront_cost: 480000,
              annual_cost: 320000,
              success_rate_pct: 62,
              downside_risk_pct: 26,
              volatility_index: 0.44
            }
          },
          {
            id: 'sc-3',
            name: "Low-Cost Mobile Cart / Express Stall",
            badge: "Balanced",
            badge_color: "green",
            badge_type: "balanced",
            probability: 79,
            description: "Minimal overhead cart model with high gross margins and immediate cash breakeven.",
            assumptions: [
              { icon: '🚲', label: 'Cart & Appliance Capex', value: '₹ 45,000' },
              { icon: '💵', label: 'Net Monthly Earnings', value: '₹ 42,000 / mo' },
              { icon: '🛡️', label: 'Downside Protection', value: 'Low fixed rent' }
            ],
            variables: {
              base_cashflow_annual: 504000,
              growth_mean_pct: 9.5,
              growth_std_pct: 3.0,
              upfront_cost: 45000,
              annual_cost: 120000,
              success_rate_pct: 79,
              downside_risk_pct: 10,
              volatility_index: 0.16
            }
          }
        ]
      };
    }

    // Domain 2: Masters / Higher Studies vs Job
    if (p.includes('master') || p.includes('ms') || p.includes('mtech') || p.includes('degree') || p.includes('higher study') || p.includes('gre') || p.includes('gate')) {
      return {
        title: "Industry Job vs Master's Degree Decision",
        category: "Higher Education vs Direct Career",
        scenarios: [
          {
            id: 'sc-1',
            name: "Direct Software Industry Role & Rapid Promotion",
            badge: "Most Likely",
            badge_color: "blue",
            badge_type: "recommended",
            probability: 86,
            description: "Start working immediately, earning continuous cash flow and gaining real-world production experience.",
            assumptions: [
              { icon: '💼', label: 'Starting Salary', value: MonteCarloEngine.formatINR(income) },
              { icon: '📈', label: 'Annual Appraisal Hike', value: '14%' },
              { icon: '💰', label: 'Zero Student Debt', value: '₹ 0 upfront loans' }
            ],
            variables: {
              base_cashflow_annual: income,
              growth_mean_pct: 14.0,
              growth_std_pct: 3.5,
              upfront_cost: 0,
              annual_cost: expenses,
              success_rate_pct: 86,
              downside_risk_pct: 10,
              volatility_index: 0.18
            }
          },
          {
            id: 'sc-2',
            name: "Full-Time Master's (MS / M.Tech) with Higher Payoff",
            badge: "Higher Risk",
            badge_color: "red",
            badge_type: "risky",
            probability: 68,
            description: "2 years of study and student loan commitment, followed by higher starting baseline and tier-1 roles.",
            assumptions: [
              { icon: '🎓', label: 'Tuition & Living Cost', value: MonteCarloEngine.formatINR(savings * 0.8 + 800000) },
              { icon: '🚀', label: 'Post-Degree Package', value: MonteCarloEngine.formatINR(income * 2.2) },
              { icon: '⏱️', label: 'Payback Period', value: '2.5 - 3 years' }
            ],
            variables: {
              base_cashflow_annual: income * 2.2,
              growth_mean_pct: 19.0,
              growth_std_pct: 6.5,
              upfront_cost: savings * 0.8 + 800000,
              annual_cost: expenses * 1.15,
              success_rate_pct: 68,
              downside_risk_pct: 22,
              volatility_index: 0.38
            }
          },
          {
            id: 'sc-3',
            name: "Work Full-Time + Executive / Online Master's",
            badge: "Balanced",
            badge_color: "green",
            badge_type: "balanced",
            probability: 81,
            description: "Maintain continuous salary while funding accredited certifications and part-time advanced degree.",
            assumptions: [
              { icon: '⚖️', label: 'Continuous Income Stream', value: MonteCarloEngine.formatINR(income * 1.1) },
              { icon: '📚', label: 'Program Expense', value: '₹ 1.8 L total' },
              { icon: '🎯', label: 'Career Switch Boost', value: '+35% hike' }
            ],
            variables: {
              base_cashflow_annual: income * 1.1,
              growth_mean_pct: 15.0,
              growth_std_pct: 4.0,
              upfront_cost: 180000,
              annual_cost: expenses,
              success_rate_pct: 81,
              downside_risk_pct: 12,
              volatility_index: 0.22
            }
          }
        ]
      };
    }

    // Domain 3: Startup / Business Venture
    if (p.includes('startup') || p.includes('business') || p.includes('company') || p.includes('saas') || p.includes('founder')) {
      return {
        title: "Startup Venture vs Corporate Stability",
        category: "Entrepreneurship & Venture Risk",
        scenarios: [
          {
            id: 'sc-1',
            name: "Full-Time Bootstrapped Startup Launch",
            badge: "Higher Risk",
            badge_color: "red",
            badge_type: "risky",
            probability: 60,
            description: "100% focus on building and distributing your product, tolerating initial lean months for exponential upside.",
            assumptions: [
              { icon: '⚡', label: 'Initial Runway Burn', value: MonteCarloEngine.formatINR(savings * 0.7) },
              { icon: '📈', label: 'Yr 3 ARR Potential', value: '₹ 25 LPA+' },
              { icon: '🎯', label: 'Product-Market Fit Risk', value: 'High volatility' }
            ],
            variables: {
              base_cashflow_annual: income * 1.8,
              growth_mean_pct: 32.0,
              growth_std_pct: 9.0,
              upfront_cost: savings * 0.7,
              annual_cost: expenses,
              success_rate_pct: 60,
              downside_risk_pct: 30,
              volatility_index: 0.48
            }
          },
          {
            id: 'sc-2',
            name: "Corporate Job + Weekend MVP Validation",
            badge: "Balanced",
            badge_color: "green",
            badge_type: "balanced",
            probability: 83,
            description: "Keep full salary safety net while validating customer willingness-to-pay before leaving job.",
            assumptions: [
              { icon: '🛡️', label: 'Salary Buffer', value: MonteCarloEngine.formatINR(income) },
              { icon: '⏳', label: 'Side Hustle Effort', value: '15 hrs/wk' },
              { icon: '🚀', label: 'Transition Point', value: 'When MRR > 50% salary' }
            ],
            variables: {
              base_cashflow_annual: income * 1.25,
              growth_mean_pct: 18.0,
              growth_std_pct: 4.5,
              upfront_cost: 60000,
              annual_cost: expenses,
              success_rate_pct: 83,
              downside_risk_pct: 12,
              volatility_index: 0.22
            }
          },
          {
            id: 'sc-3',
            name: "Join Funded Early-Stage Startup (High ESOP)",
            badge: "Most Likely",
            badge_color: "blue",
            badge_type: "recommended",
            probability: 78,
            description: "Gain rapid startup execution leadership and equity upside without personal runway burn.",
            assumptions: [
              { icon: '💼', label: 'Base + Equity Mix', value: MonteCarloEngine.formatINR(income * 1.2) },
              { icon: '📈', label: 'Equity Liquidity Value', value: 'High multiplier' },
              { icon: '⚡', label: 'Work Intensity', value: 'Fast paced' }
            ],
            variables: {
              base_cashflow_annual: income * 1.2,
              growth_mean_pct: 16.0,
              growth_std_pct: 5.0,
              upfront_cost: 0,
              annual_cost: expenses,
              success_rate_pct: 78,
              downside_risk_pct: 15,
              volatility_index: 0.26
            }
          }
        ]
      };
    }

    // Domain 4: General Smart Tailored Simulation
    return {
      title: prompt ? `Simulation: "${prompt.slice(0, 45)}..."` : "Strategic Decision Simulation",
      category: "Probabilistic Financial Strategy",
      scenarios: [
        {
          id: 'sc-1',
          name: "Direct Execution & Core Path",
          badge: "Most Likely",
          badge_color: "blue",
          badge_type: "recommended",
          probability: 82,
          description: `Direct execution focusing on immediate momentum and compounding value over ${info.time_horizon || '5 years'}.`,
          assumptions: [
            { icon: '💼', label: 'Baseline Cashflow', value: MonteCarloEngine.formatINR(income) },
            { icon: '📈', label: 'Annual Compounded Growth', value: '12.5%' },
            { icon: '🛡️', label: 'Downside Risk Buffer', value: 'Protected (10%)' }
          ],
          variables: {
            base_cashflow_annual: income,
            growth_mean_pct: 12.5,
            growth_std_pct: 3.5,
            upfront_cost: 0,
            annual_cost: expenses,
            success_rate_pct: 82,
            downside_risk_pct: 10,
            volatility_index: 0.18
          }
        },
        {
          id: 'sc-2',
          name: "High Leverage / Accelerated Investment",
          badge: "Higher Risk",
          badge_color: "red",
          badge_type: "risky",
          probability: 64,
          description: "Commits higher upfront capital and effort to achieve disproportionate long-term returns.",
          assumptions: [
            { icon: '⚡', label: 'Capital Commitment', value: MonteCarloEngine.formatINR(savings * 0.75 + 150000) },
            { icon: '🚀', label: 'Peak Compounded Return', value: MonteCarloEngine.formatINR(income * 1.9) },
            { icon: '📊', label: 'Market Volatility', value: 'Moderate-High' }
          ],
          variables: {
            base_cashflow_annual: income * 1.9,
            growth_mean_pct: 22.0,
            growth_std_pct: 7.0,
            upfront_cost: savings * 0.75 + 150000,
            annual_cost: expenses * 1.1,
            success_rate_pct: 64,
            downside_risk_pct: 24,
            volatility_index: 0.40
          }
        },
        {
          id: 'sc-3',
          name: "Diversified & Balanced Strategy",
          badge: "Balanced",
          badge_color: "green",
          badge_type: "balanced",
          probability: 80,
          description: "Maintains strong downside capital preservation while steadily compounding upside opportunities.",
          assumptions: [
            { icon: '⚖️', label: 'Dual Stream Yield', value: MonteCarloEngine.formatINR(income * 1.18) },
            { icon: '📈', label: 'Steady Growth Rate', value: '14.0%' },
            { icon: '🎯', label: 'Success Likelihood', value: '80%' }
          ],
          variables: {
            base_cashflow_annual: income * 1.18,
            growth_mean_pct: 14.0,
            growth_std_pct: 4.0,
            upfront_cost: 50000,
            annual_cost: expenses,
            success_rate_pct: 80,
            downside_risk_pct: 12,
            volatility_index: 0.22
          }
        }
      ]
    };
  }
}

window.MonteCarloEngine = MonteCarloEngine;
window.RealitySimAIService = RealitySimAIService;
