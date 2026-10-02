exports.handler = async function(event, context) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "API key not configured in Netlify environment" })
    };
  }

  try {
    const { decision_prompt, additional_info } = JSON.parse(event.body || "{}");

    const systemPrompt = `You are RealitySim AI, a practical decision simulator built for everyday people.
Analyze the user's decision and generate 3 realistic, grounded scenarios in SIMPLE, EVERYDAY ENGLISH (no complex financial jargon, no inflated "fancy" numbers).

Guidelines for realistic estimates:
1. Break down real monthly/daily economics (e.g. for shops/stalls: daily customer count, setup deposit, daily sales, and honest monthly take-home profit after all bills).
2. For jobs/careers: use realistic in-hand salary, living expenses, and normal yearly appraisal hikes.
3. Write clear 1-2 sentence descriptions explaining how each path works in plain English.
4. Keep initial upfront costs and profits realistic for India / local markets.

You MUST output ONLY valid JSON in this exact structure:
{
  "title": "Simple Decision Name",
  "category": "Practical Category",
  "scenarios": [
    {
      "name": "Simple Name of Option (e.g. Standard Kiosk / Direct Job)",
      "badge": "Most Likely / Higher Risk / Balanced",
      "badge_color": "blue / red / green",
      "badge_type": "recommended / risky / balanced",
      "description": "Simple 1-2 sentence explanation of what this option means in real life.",
      "probability": 82,
      "assumptions": [
        {"icon": "💰", "label": "Setup Cost", "value": "₹ 60,000 upfront"},
        {"icon": "💵", "label": "Monthly Take-Home", "value": "₹ 35,000 / mo"},
        {"icon": "🛡️", "label": "Main Risk", "value": "Location footfall"}
      ],
      "variables": {
        "base_cashflow_annual": 420000,
        "growth_mean_pct": 10.0,
        "growth_std_pct": 3.0,
        "upfront_cost": 60000,
        "annual_cost": 150000,
        "success_rate_pct": 82,
        "downside_risk_pct": 10,
        "volatility_index": 0.18
      }
    }
  ]
}`;

    const info = additional_info || {};
    const userPrompt = `Decision: "${decision_prompt}"
Context & Profile:
- Age: ${info.age || 22}
- Education: ${info.education_level || 'General / Professional'}
- Location: ${info.location || 'India'}
- Experience: ${info.work_experience || 'Fresher / Beginner'}
- Current Savings: ₹ ${info.current_savings || 50000}
- Expected Income: ₹ ${info.expected_annual_income || 400000}
- Monthly Expenses: ₹ ${info.monthly_expenses || 15000}
- Time Horizon: ${info.time_horizon || '5 years'}
- Priority: ${info.decision_priority || 'Balanced Growth'}
- Goals: ${(info.goals || []).join(', ')}
- Additional Context: ${info.additional_context || 'None'}`;

    const primaryModel = process.env.OPENROUTER_MODEL || 'google/gemini-2.5-flash';
    const models = [
      primaryModel,
      'google/gemini-2.5-flash',
      'openai/gpt-4o-mini',
      'meta-llama/llama-3.3-70b-instruct',
      'liquid/lfm-2.5-2.6b:free',
      'google/gemma-4-26b-a4b-it:free',
      'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
      'qwen/qwen3.8-27b:free'
    ];

    for (const model of models) {
      try {
        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`,
            "HTTP-Referer": "https://realitysim-ai.netlify.app",
            "X-Title": "RealitySim AI"
          },
          body: JSON.stringify({
            model: model,
            max_tokens: 2000,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userPrompt }
            ],
            temperature: 0.7
          })
        });

        if (response.ok) {
          const data = await response.json();
          const content = data.choices?.[0]?.message?.content;
          if (content) {
            let parsed = null;
            try {
              parsed = JSON.parse(content);
            } catch (e) {
              const match = content.match(/\{[\s\S]*\}/);
              if (match) parsed = JSON.parse(match[0]);
            }
            if (parsed && parsed.scenarios) {
              parsed.scenarios = parsed.scenarios.map((sc, idx) => {
                const prob = sc.probability || sc.probability_pct || sc.variables?.success_rate_pct || (idx === 0 ? 82 : (idx === 1 ? 65 : 78));
                return {
                  id: sc.id || `sc-${idx + 1}`,
                  name: sc.name || `Scenario ${idx + 1}`,
                  badge: sc.badge || (idx === 0 ? 'Most Likely' : (idx === 1 ? 'Higher Risk' : 'Balanced')),
                  badge_color: sc.badge_color || (idx === 0 ? 'blue' : (idx === 1 ? 'red' : 'green')),
                  badge_type: sc.badge_type || (idx === 0 ? 'recommended' : (idx === 1 ? 'risky' : 'balanced')),
                  description: sc.description || '',
                  probability: prob,
                  probability_pct: prob,
                  assumptions: sc.assumptions || [],
                  variables: sc.variables || {}
                };
              });
              return {
                statusCode: 200,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(parsed)
              };
            }
          }
        }
      } catch (err) {
        console.warn(`Model ${model} failed in function:`, err);
      }
    }

    return {
      statusCode: 500,
      body: JSON.stringify({ error: "All AI models were busy. Please try again." })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
};
