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
    if (val === null || val === undefined) return '₹ 0';
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
// 2. Realistic LLM & Scenario Service
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

  static async generateScenarios(decisionPrompt, additionalInfo = {}) {
    const systemPrompt = `You are RealitySim AI, an advanced probabilistic decision modeling engine.
Analyze the user's decision and generate 3 to 4 distinct, realistic future scenarios.
You MUST output ONLY valid JSON matching this structure:
{
  "title": "Short title of decision",
  "category": "Career / Financial / Education / Business",
  "scenarios": [
    {
      "name": "Scenario Name (e.g. Job + Immediate Up-skilling)",
      "badge": "Most Likely / Higher Risk / Balanced / Conservative",
      "badge_color": "blue / red / green / purple",
      "description": "Specific, realistic description of what happens in this scenario.",
      "assumptions": [
        {"icon": "💼", "label": "Starting Compensation", "value": "₹ 7.5 LPA"},
        {"icon": "📈", "label": "Annual Hike", "value": "12%"},
        {"icon": "⏱️", "label": "Effort / Study", "value": "10 hrs/week"}
      ],
      "variables": {
        "base_cashflow_annual": 750000,
        "growth_mean_pct": 12,
        "growth_std_pct": 3.5,
        "upfront_cost": 0,
        "annual_cost": 240000,
        "success_rate_pct": 85,
        "downside_risk_pct": 10,
        "volatility_index": 0.20
      }
    }
  ]
}`;

    const userPrompt = `Decision: "${decisionPrompt}"
Context & Profile:
- Age: ${additionalInfo.age || 22}
- Education: ${additionalInfo.education_level || 'B.Tech CSE'}
- Location: ${additionalInfo.location || 'India'}
- Experience: ${additionalInfo.work_experience || 'Fresher'}
- Current Savings: ₹ ${additionalInfo.current_savings || 50000}
- Expected Income: ₹ ${additionalInfo.expected_annual_income || 600000}
- Monthly Expenses: ₹ ${additionalInfo.monthly_expenses || 15000}
- Time Horizon: ${additionalInfo.time_horizon || '5 years'}
- Priority: ${additionalInfo.decision_priority || 'Balanced Growth'}
- Goals: ${(additionalInfo.goals || []).join(', ')}
- Additional Context: ${additionalInfo.additional_context || 'None'}`;

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
          return netlifyData;
        }
      }
    } catch (e) {
      console.log("Netlify function not reachable, trying client/fallback...", e);
    }

    // 2. Try Client-Side LLM Call
    try {
      const rawText = await this.callLLM(systemPrompt, userPrompt);
      const parsed = this.parseJSON(rawText);
      if (parsed && parsed.scenarios && parsed.scenarios.length > 0) {
        return parsed;
      }
    } catch (err) {
      console.warn("LLM API fallback to domain simulation generator:", err);
    }

    // Dynamic Domain Generator Fallback (Guarantees zero downtime & realism)
    return this.generateSmartFallback(decisionPrompt, additionalInfo);
  }

  static generateSmartFallback(prompt, info) {
    const p = (prompt || '').toLowerCase();
    const income = parseFloat(info.expected_annual_income || 650000);
    const savings = parseFloat(info.current_savings || 100000);
    const expenses = parseFloat(info.monthly_expenses ? info.monthly_expenses * 12 : 180000);

    let title = "Decision Analysis Simulation";
    let category = "Career & Financial Strategy";
    let sc1Name = "Primary Direct Route";
    let sc2Name = "High Growth / High Investment";
    let sc3Name = "Balanced / Hybrid Strategy";

    if (p.includes('master') || p.includes('ms') || p.includes('mtech') || p.includes('job') || p.includes('study')) {
      title = "B.Tech Job vs Master's Degree Decision";
      category = "Higher Education vs Industry";
      sc1Name = "Immediate Industry Job & Fast Promotions";
      sc2Name = "Full-Time Master's Degree & Global Pivot";
      sc3Name = "Part-Time Higher Studies while Working";
    } else if (p.includes('startup') || p.includes('business') || p.includes('company')) {
      title = "Startup Venture vs Stable Corporate Path";
      category = "Entrepreneurship & Career Risk";
      sc1Name = "Full-Time Venture Launch";
      sc2Name = "Corporate Role + Side Project MVP";
      sc3Name = "Join High-Growth Series-A Startup";
    } else if (p.includes('buy') || p.includes('rent') || p.includes('flat') || p.includes('house') || p.includes('invest')) {
      title = "Real Estate Purchase vs Market Investment";
      category = "Asset Allocation & Real Estate";
      sc1Name = "Property Purchase with Home Loan";
      sc2Name = "Rent & Disciplined Equity SIPs";
      sc3Name = "Hybrid: Fractional Real Estate & Debt";
    }

    return {
      title: title,
      category: category,
      scenarios: [
        {
          id: 'sc-1',
          name: sc1Name,
          badge: 'Most Likely',
          badge_color: 'blue',
          description: `Focuses on immediate entry and compounding early earnings with stable annual progressions.`,
          assumptions: [
            { icon: '💼', label: 'Starting Income', value: MonteCarloEngine.formatINR(income) },
            { icon: '📈', label: 'Annual Growth Rate', value: '11.5%' },
            { icon: '🛡️', label: 'Downside Risk', value: 'Low (10%)' }
          ],
          variables: {
            base_cashflow_annual: income,
            growth_mean_pct: 11.5,
            growth_std_pct: 3.0,
            upfront_cost: 0,
            annual_cost: expenses,
            success_rate_pct: 85,
            downside_risk_pct: 10,
            volatility_index: 0.18
          }
        },
        {
          id: 'sc-2',
          name: sc2Name,
          badge: 'Higher Risk',
          badge_color: 'red',
          description: `Requires higher upfront commitment and capital, unlocking accelerated long-term compensation spikes.`,
          assumptions: [
            { icon: '🎓', label: 'Upfront Capital', value: MonteCarloEngine.formatINR(savings * 0.8 + 400000) },
            { icon: '🚀', label: 'Post-Milestone Income', value: MonteCarloEngine.formatINR(income * 1.85) },
            { icon: '⚡', label: 'Growth Potential', value: '18.0%' }
          ],
          variables: {
            base_cashflow_annual: income * 1.85,
            growth_mean_pct: 18.0,
            growth_std_pct: 7.0,
            upfront_cost: savings * 0.8 + 400000,
            annual_cost: expenses * 1.1,
            success_rate_pct: 70,
            downside_risk_pct: 22,
            volatility_index: 0.42
          }
        },
        {
          id: 'sc-3',
          name: sc3Name,
          badge: 'Balanced',
          badge_color: 'green',
          description: `Diversifies risk by maintaining active cash flows while steadily building next-tier credentials.`,
          assumptions: [
            { icon: '⚖️', label: 'Dual Stream Income', value: MonteCarloEngine.formatINR(income * 1.15) },
            { icon: '📈', label: 'Compounded Hike', value: '14.0%' },
            { icon: '🎯', label: 'Success Probability', value: '82%' }
          ],
          variables: {
            base_cashflow_annual: income * 1.15,
            growth_mean_pct: 14.0,
            growth_std_pct: 4.0,
            upfront_cost: 80000,
            annual_cost: expenses,
            success_rate_pct: 82,
            downside_risk_pct: 12,
            volatility_index: 0.24
          }
        }
      ]
    };
  }
}

// Attach to global window object
window.MonteCarloEngine = MonteCarloEngine;
window.RealitySimAIService = RealitySimAIService;
