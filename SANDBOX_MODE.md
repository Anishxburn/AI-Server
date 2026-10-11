# EMS Sandbox Mode

The AI MCP Trace Dashboard has a separate **Sandbox** tab at `/debug/ai`. Questions entered there go to the configured Ollama chat model with recent sandbox conversation context and a synthetic EMS dataset. This route does not invoke DaxView authorization, the intent planner, MCP, or the live DaxView job flow.

The sample dataset is generated locally by `ems_sandbox.py` and includes seven daily site-energy readings, five device energy/demand summaries, hourly demand and electrical readings, and sample alarms. All values and identifiers are illustrative. The emissions factor is explicitly an illustrative assumption (`0.4 kgCO2e/kWh`), not a reporting recommendation.

## Response contract

`POST /debug/sandbox` accepts:

```json
{
  "message": "Rank the top 5 energy-consuming devices and show a chart",
  "history": [
    {"role": "user", "content": "What is peak demand?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

It immediately returns HTTP `202` with a `job_id`. Poll `GET /debug/sandbox/jobs/{job_id}` for `queued` or `running`; when complete, the result contains `answer`, `model`, `dataset` provenance, deterministic `evidence`, optional `charts`, optional `report`, `formula_reference`, and `mcp_called: false`. This avoids holding the browser request open while Ollama generates. Chart specs use `type`, `title`, `labels`, and `series[]` with `name`, `unit`, and numeric `data`. The V2 frontend can render these chart specs with Chart.js and render report `sections` as cards, tables, or exportable reports. The response object is model-independent: a hosted provider can replace Ollama later without changing the front-end contract.

Ranking, sums, extrema, and the demo carbon estimate are calculated in Python before answer generation. The LLM explains the facts and handles general questions and follow-ups; it does not define chart values. The formula reference documents required inputs and limitations for energy, maximum demand, load factor, power factor, emissions estimates, and percentage difference.

Sandbox chat history exists in the browser page session only. Debug traces include sandbox inputs, evidence, model answer, chart specs, and report structure in the same in-memory trace buffer used by the dashboard.

## Deploy

From `/opt/chatbot-stack/Chatbot-UI`:

```bash
docker compose build chatbot-api agent-poc-api
docker compose up -d --force-recreate chatbot-api agent-poc-api chatbot-ui
docker compose ps
```

Then open `http://<server-host>:3030/debug/ai` (or the configured `AGENT_POC_PORT`) and choose **Sandbox**. The configured `CHAT_MODEL` must already be available in Ollama. The sandbox dashboard requires `AI_DEBUG_DASHBOARD_ENABLED=true`.
