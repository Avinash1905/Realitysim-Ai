/**
 * RealitySim AI - Universal Simulation & LLM Engine
 * Runs client-side Monte Carlo simulations and connects to OpenRouter / Gemini LLMs.
 * Works seamlessly on static hosting (Netlify) and local servers.
 */

const RealitySimConfig = {
  getApiKey: () => {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem('realitysim_api_key') || 
             localStorage.getItem('openrouter_api_key') || 
             localStorage.getItem('gemini_api_key') || 
             '';
    }
    return '';
  },
  openRouterUrl: 'https://openrouter.ai/api/v1/chat/completions',
  fallbackModels: [
    'google/gemini-2.5-flash',
    'openai/gpt-4o-mini',
    'meta-llama/llama-3.3-70b-instruct',
    'liquid/lfm-2.5-2.6b:free',
    'google/gemma-4-26b-a4b-it:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'qwen/qwen3.8-27b:free'
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
            'HTTP-Referer': (typeof window !== 'undefined' && window.location && window.location.origin) ? window.location.origin : 'https://realitysim-ai.netlify.app',
            'X-Title': 'RealitySim AI'
          },
          body: JSON.stringify({
            model: model,
            max_tokens: 2000,
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
    const systemPrompt = `You are RealitySim AI, an advanced probabilistic decision simulation engine.
Analyze the user's exact decision question and profile, then generate 3 realistic, distinct, and customized future scenarios.
Do NOT output generic templates. Tailor every title, description, assumption, and financial variable specifically to the user's question and context.

You MUST output ONLY valid JSON matching this schema:
{
  "title": "Concise Decision Title (e.g., Food Truck vs Cafe in Bangalore)",
  "category": "Specific Domain (e.g., Retail F&B, Tech Career, Real Estate)",
  "scenarios": [
    {
      "name": "Specific Pathway Name",
      "badge": "Most Likely / Higher Risk / Balanced",
      "badge_color": "blue / red / green",
      "badge_type": "recommended / risky / balanced",
      "description": "Clear 1-2 sentence real-world explanation tailored to their decision.",
      "probability": 78,
      "assumptions": [
        {"icon": "💰", "label": "Upfront Investment / Cost", "value": "₹ X Lakhs"},
        {"icon": "📈", "label": "Expected Annual Revenue / Growth", "value": "₹ Y LPA"},
        {"icon": "🛡️", "label": "Key Operational Risk", "value": "Specific risk detail"}
      ],
      "variables": {
        "base_cashflow_annual": 600000,
        "growth_mean_pct": 12.0,
        "growth_std_pct": 3.5,
        "upfront_cost": 100000,
        "annual_cost": 180000,
        "success_rate_pct": 78,
        "downside_risk_pct": 14,
        "volatility_index": 0.22
      }
    }
  ]
}`;

    const info = additionalInfo || {};
    const userPrompt = `User Decision: "${decisionPrompt}"
Context & Profile:
- Age: ${info.age || 22}
- Education: ${info.education_level || 'Professional'}
- Location: ${info.location || 'India'}
- Current Status / Experience: ${info.current_status || info.work_experience || 'Beginner'}
- Current Savings: ₹ ${info.current_savings || 50000}
- Monthly Expenses: ₹ ${info.monthly_expenses || 15000}
- Stated Goals: ${(info.goals || []).join(', ') || 'Wealth creation & stability'}
- Additional Context: ${info.additional_context || 'None'}

Generate 3 deeply realistic, customized scenarios addressing this exact situation.`;

    // 1. Direct OpenRouter AI Call
    try {
      const rawText = await this.callLLM(systemPrompt, userPrompt);
      const parsed = this.parseJSON(rawText);
      if (parsed && parsed.scenarios && parsed.scenarios.length > 0) {
        return this.normalizeScenarios(parsed);
      }
    } catch (err) {
      console.error("OpenRouter scenario generation failed:", err);
      throw err;
    }
  }
}

if (typeof window !== 'undefined') {
  window.MonteCarloEngine = MonteCarloEngine;
  window.RealitySimAIService = RealitySimAIService;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MonteCarloEngine, RealitySimAIService };
}
