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

    const systemPrompt = `You are RealitySim AI, an advanced probabilistic decision modeling engine.
Analyze the user's decision and generate 3 to 4 distinct, realistic future scenarios.
You MUST output ONLY valid JSON matching this structure:
{
  "title": "Short title of decision",
  "category": "Career / Financial / Education / Business",
  "scenarios": [
    {
      "name": "Scenario Name",
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

    const info = additional_info || {};
    const userPrompt = `Decision: "${decision_prompt}"
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

    const models = [
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
