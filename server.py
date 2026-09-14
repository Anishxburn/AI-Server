"""EMS-only chatbot API with Ollama, PostgreSQL, and pgvector RAG."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import time
import uuid
from collections import deque
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.runnables import RunnableLambda
import psycopg
from psycopg.rows import dict_row


HOST = os.getenv("CHATBOT_HOST", "127.0.0.1")
PORT = int(os.getenv("CHATBOT_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:1.5b")
CHAT_MODEL = os.getenv("CHAT_MODEL", OLLAMA_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DATABASE_URL = os.getenv("DATABASE_URL", "")
RAG_MATCH_LIMIT = int(os.getenv("RAG_MATCH_LIMIT", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
TRACE_LIMIT = int(os.getenv("CHATBOT_TRACE_LIMIT", "25"))
DAXVIEW_MCP_ENABLED = os.getenv("DAXVIEW_MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_MCP_URL = os.getenv("DAXVIEW_MCP_URL", "").strip()
DAXVIEW_MCP_AUTH_TOKEN = os.getenv("DAXVIEW_MCP_AUTH_TOKEN", "").strip()
DAXVIEW_MCP_TIMEOUT = int(os.getenv("DAXVIEW_MCP_TIMEOUT", "20"))
DAXVIEW_MCP_PROTOCOL_VERSION = os.getenv("DAXVIEW_MCP_PROTOCOL_VERSION", "2025-06-18")
DAXVIEW_MCP_SESSION_ID = None
TRACES = deque(maxlen=TRACE_LIMIT)
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv(
        "CHATBOT_ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
}

EMS_KEYWORDS = {
    "ems", "energy", "electric", "electricity", "power", "kwh", "kw", "kvar",
    "demand", "tariff", "billing", "meter", "submeter", "load", "consumption",
    "iso 50001", "iec", "ieee", "protocol", "modbus", "bacnet", "scada",
    "bms", "power factor", "harmonic", "baseline", "enpi", "audit",
    "carbon", "emission", "peak", "transformer", "switchgear", "solar",
    "janitza", "janitzata", "umg", "umg 509", "umg509", "umg 96", "umg 604",
    "voltage", "voltage sag", "sag", "dip", "voltage dip", "voltage swell",
    "swell", "transient", "flicker", "unbalance", "imbalance", "phase loss",
    "overvoltage", "undervoltage", "current", "frequency", "thd", "power quality",
    "event", "alarm", "fault", "disturbance", "waveform", "rms", "l-n", "l-l",
    "daxview", "site summary", "device summary", "inventory", "open alarms",
}

DAXVIEW_TOOL_KEYWORDS = {
    "health_check": {
        "mcp health", "mcp status", "mcp server", "mcp ok",
    },
    "get_daxview_api_status": {
        "daxview health", "daxview status", "api status", "api health",
        "backend status", "is daxview healthy", "daxview ok",
    },
    "get_daxview_inventory": {
        "inventory", "equipment list", "assets",
    },
    "get_daxview_device_summary": {
        "device summary", "meter summary", "device status", "meter status",
        "janitza status", "umg status", "max demand", "maximum demand",
        "demand reading", "meter demand", "main switch board", "devices",
        "device list", "meters", "meter list", "online devices",
        "devices online", "how many daxview devices", "how many devices",
        "device count", "online count",
    },
    "get_daxview_site_summary": {
        "site summary", "site status", "site overview", "sites", "building summary",
        "facility summary",
    },
    "get_daxview_open_alarms": {
        "open alarm", "open alarms", "active alarm", "active alarms", "current alarm",
        "current alarms", "alarm status", "alarm count", "active alarm count",
        "open alert", "open alerts", "active alert", "active alerts", "current alert",
        "current alerts", "alert", "alerts", "alert count", "active alert count",
        "faults", "events",
    },
    "get_daxview_billing": {
        "billing", "bill", "current month", "this month cost", "monthly cost",
        "invoice summary", "energy cost", "tariff cost",
    },
}

DAXVIEW_SCOPE_REQUIRED_TOOLS = {
    "get_daxview_site_summary",
    "get_daxview_device_summary",
    "get_daxview_open_alarms",
    "get_daxview_billing",
}

DAXVIEW_GLOBAL_SCOPE_PHRASES = {
    "all", "overall", "global", "entire", "every", "current", "today",
    "now", "latest", "system wide", "system-wide", "whole site",
    "all sites", "all meters", "all devices", "all buildings",
}

DAXVIEW_TARGET_SCOPE_PATTERNS = (
    r"\bsite\s*[:#-]?\s*[\w.-]+",
    r"\bbuilding\s*[:#-]?\s*[\w.-]+",
    r"\bblock\s*[:#-]?\s*[\w.-]+",
    r"\bfloor\s*[:#-]?\s*[\w.-]+",
    r"\bmeter\s*[:#-]?\s*[\w.-]+",
    r"\bdevice\s*[:#-]?\s*[\w.-]+",
    r"\bfeeder\s*[:#-]?\s*[\w.-]+",
    r"\bpanel\s*[:#-]?\s*[\w.-]+",
    r"\bmsb\b",
    r"\bmain switch board\b",
    r"\bumg[\s-]?\d+\b",
)

REFUSAL = (
    "I can only assist with Energy Management System related questions, including "
    "EMS data, ISO 50001, IEC, IEEE, energy usage, power monitoring, meters, "
    "Janitza UMG devices, voltage sag, power quality, demand, tariff, alarms, "
    "and related technical topics."
)

SAFETY_DOCTRINE = """YOU ARE THE CHIEF ENERGY MANAGER AI FOR AN EMS SYSTEM.

LANGUAGE CONTROL RULE STRICT:
1. Detect the primary language used in the user's prompt, including English, Malay, Manglish, or Technical Malay.
2. Always respond in the exact same primary language as the user's prompt.
3. If the user uses Manglish or Technical Malay, respond in the same Manglish or Technical Malay style.
4. Keep technical and engineering terms intact, including Busbar, Feeder Cable, Load Shedding, Power Factor, Rated Capacity, EMS, Daxview, Janitza, UMG, ISO 50001, IEC, IEEE, kW, kWh, THD, voltage sag, and alarm identifiers.
5. Do not translate equipment names, model names, standards, units, API names, or alarm identifiers.
6. Do not mix languages unless the user's prompt mixes languages or explicitly asks for translation.

PRIMARY MISSION:
Ensure energy efficiency without compromising physical safety and hardware limits.

SAFETY GUARDRAIL RULES STRICT HARD LIMITS:
1. REJECT AT ALL COSTS:
If any operating parameter including current, voltage, temperature, pressure, power, or frequency exceeds or will exceed maximum hardware rated capacity or limits, strictly reject any request to increase load or override alarms.

2. PHYSICS IS NOT OPTIONAL:
Never provide tolerant or hesitant answers like "if safe proceed", "proceed with caution", or "try first" when hardware limits are breached. Operating above 100% capacity is an absolute physical hazard.

3. RESPONSE STRUCTURE FOR CRITICAL LIMIT BREACHES:
Your response must contain exactly these sections:
- DIRECT REJECTION: State clearly that the action is rejected.
- PHYSICAL REASONING: Explain the risk of hardware damage, insulation failure, or fire hazard.
- MITIGATION ACTION: Direct immediate load shedding or isolation to reduce load below rated limits.

Tone: Professional, direct, concise, strict, no conversational fluff."""

MATH_DOCTRINE = """MATHEMATICAL ACCURACY RULE:
When calculating energy cost, always use this exact formula:
Cost (RM) = Power (kW) x Operating Hours (h) x Tariff Rate (RM/kWh).
Show the step-by-step arithmetic when a cost calculation is requested.
Do not invent missing values. If power, hours, or tariff rate is missing, ask for the missing value or state the assumption clearly."""

POC_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>EMS Multi-Agent POC</title>
    <style>
      :root { font-family: Inter, Arial, sans-serif; color: #17202a; background: #f4f7fb; }
      * { box-sizing: border-box; }
      body { margin: 0; min-height: 100vh; }
      main { width: min(1120px, calc(100% - 32px)); margin: 20px auto; }
      header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 18px; }
      h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
      .muted { color: #64748b; margin: 6px 0 0; }
      .status { padding: 8px 10px; border: 1px solid #cbd5e1; background: #fff; font-weight: 700; }
      .panel { border: 1px solid #d6deea; background: #fff; padding: 16px; margin-bottom: 14px; }
      form { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
      textarea { min-height: 74px; resize: vertical; padding: 12px; border: 1px solid #cbd5e1; font: inherit; }
      button { border: 0; background: #0f766e; color: #fff; font-weight: 800; padding: 0 18px; cursor: pointer; }
      button:disabled { opacity: .55; cursor: not-allowed; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
      .wide { grid-column: 1 / -1; }
      h2 { margin: 0 0 10px; font-size: 14px; text-transform: uppercase; color: #475569; }
      .answer-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
      .pill { border: 1px solid #cbd5e1; background: #f8fafc; color: #334155; padding: 5px 8px; font-size: 12px; font-weight: 700; }
      pre { white-space: pre-wrap; margin: 0; line-height: 1.5; font: inherit; }
      ol, ul { margin: 0; padding-left: 20px; }
      li { margin: 0 0 8px; }
      strong { display: block; color: #0f766e; }
      .empty { color: #64748b; }
      @media (max-width: 760px) {
        header, form, .grid { grid-template-columns: 1fr; display: grid; }
        button { min-height: 44px; }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <div>
          <h1>EMS Safety Multi-Agent POC</h1>
          <p class="muted">Safety-first EMS agents -> DeepSeek final decision maker with strict hardware limits</p>
        </div>
        <div id="status" class="status">Checking...</div>
      </header>

      <section class="panel">
        <form id="form">
          <textarea id="message">My Janitza UMG 509 shows voltage sag and one feeder is near rated current. What are the likely causes and safe EMS action?</textarea>
          <button id="send" type="submit">Run POC</button>
        </form>
      </section>

      <section class="grid">
        <article class="panel wide">
          <h2>Answer</h2>
          <div id="answer-meta" class="answer-meta"></div>
          <pre id="reply" class="empty">Run a question to see the answer.</pre>
        </article>
        <article class="panel wide">
          <h2>Daxview MCP</h2>
          <pre id="mcp-data" class="empty">Waiting.</pre>
        </article>
        <article class="panel">
          <h2 id="agent-a-title">Agent 1</h2>
          <pre id="agent-a" class="empty">Waiting.</pre>
        </article>
        <article class="panel">
          <h2 id="agent-b-title">Agent 2</h2>
          <pre id="agent-b" class="empty">Waiting.</pre>
        </article>
        <article class="panel wide">
          <h2>Sources</h2>
          <ul id="sources"><li class="empty">Waiting.</li></ul>
        </article>
      </section>
    </main>

    <script>
      const form = document.querySelector("#form");
      const message = document.querySelector("#message");
      const send = document.querySelector("#send");
      const statusBox = document.querySelector("#status");
      const reply = document.querySelector("#reply");
      const answerMeta = document.querySelector("#answer-meta");
      const mcpData = document.querySelector("#mcp-data");
      const agentATitle = document.querySelector("#agent-a-title");
      const agentBTitle = document.querySelector("#agent-b-title");
      const agentA = document.querySelector("#agent-a");
      const agentB = document.querySelector("#agent-b");
      const sources = document.querySelector("#sources");

      function setText(el, text) {
        el.classList.remove("empty");
        el.textContent = text || "No data.";
      }

      function setList(el, items, render) {
        el.innerHTML = "";
        if (!items || items.length === 0) {
          const li = document.createElement("li");
          li.className = "empty";
          li.textContent = "No data.";
          el.append(li);
          return;
        }
        items.forEach((item) => {
          const li = document.createElement("li");
          li.innerHTML = render(item);
          el.append(li);
        });
      }

      function setAgent(titleEl, bodyEl, agent) {
        titleEl.textContent = agent ? `${agent.agent} | ${agent.model}` : "Agent";
        setText(bodyEl, agent ? `Role: ${agent.role}\n\n${agent.answer}` : "No data.");
      }

      function summarizeMcp(mcp) {
        if (!mcp || (!mcp.tools?.length && !mcp.results?.length && !mcp.errors?.length)) {
          return "No Daxview MCP tool was selected for this question.";
        }
        const lines = [
          `Enabled: ${Boolean(mcp.enabled)}`,
          `Selected tools: ${(mcp.tools || []).join(", ") || "none"}`,
          `Successful results: ${(mcp.results || []).length}`,
          `Errors: ${(mcp.errors || []).length}`,
        ];
        (mcp.results || []).forEach((item, index) => {
          const structured = item.result?.structuredContent || item.result;
          const backend = structured?.backend_response;
          lines.push("");
          lines.push(`Result ${index + 1}: ${item.tool}`);
          if (structured?.backend_endpoint) lines.push(`Backend endpoint: ${structured.backend_endpoint}`);
          if (structured?.backend_http_status) lines.push(`Backend HTTP status: ${structured.backend_http_status}`);
          if (backend?.count !== undefined) lines.push(`Backend count: ${backend.count}`);
          if (backend?.active_count !== undefined) lines.push(`Active count: ${backend.active_count}`);
          if (backend?.open_count !== undefined) lines.push(`Open count: ${backend.open_count}`);
          const alarmList = backend?.alarms || backend?.alerts || backend?.open_alarms || backend?.active_alarms;
          if (Array.isArray(alarmList)) {
            lines.push(`Alerts returned: ${alarmList.length}`);
            alarmList.slice(0, 10).forEach((alarm) => {
              lines.push(`- ${alarm.title || alarm.name || alarm.rule_name || alarm.id || alarm.alarm_id || "alert"} | ${alarm.status || alarm.state || "unknown"} | ${alarm.severity || "no severity"}`);
            });
          }
          if (Array.isArray(backend?.devices)) {
            lines.push("Devices:");
            backend.devices.forEach((device) => {
              lines.push(`- ${device.device_name || device.remote_device_id} | ${device.remote_device_id || "no id"} | ${device.site_name || "no site"} | ${device.status || "no status"}`);
            });
          }
        });
        (mcp.errors || []).forEach((item) => {
          lines.push("");
          lines.push(`Error from ${item.tool}: ${item.error}`);
        });
        return lines.join("\n");
      }

      function setMeta(data) {
        answerMeta.innerHTML = "";
        const source = data.sources?.[0];
        const items = [
          `Processing time: ${Number(data.processing_time_seconds || 0).toFixed(1)}s`,
          `Decision model: ${data.decision_model || data.model || "n/a"}`,
        ];
        if (source) {
          items.push(`Top source score: ${Number(source.score || 0).toFixed(2)}`);
        }
        items.forEach((text) => {
          const span = document.createElement("span");
          span.className = "pill";
          span.textContent = text;
          answerMeta.append(span);
        });
      }

      async function checkHealth() {
        try {
          const response = await fetch("/health");
          const data = await response.json();
          statusBox.textContent = data.database === "ok" ? "API + DB online" : "API online";
        } catch {
          statusBox.textContent = "Offline";
        }
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        send.disabled = true;
        send.textContent = "Running...";
        setText(reply, "Running multi-agent flow...");
        setText(mcpData, "Waiting for Daxview MCP...");
        setText(agentA, "Waiting for Agent 1...");
        setText(agentB, "Waiting for Agent 2...");
        try {
          const response = await fetch("/multi-agent-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message.value.trim() }),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || "Request failed");
          setMeta(data);
          setText(reply, data.reply);
          setText(mcpData, summarizeMcp(data.daxview_mcp));
          setAgent(agentATitle, agentA, data.agent_discussion?.[0]);
          setAgent(agentBTitle, agentB, data.agent_discussion?.[1]);
          setList(sources, data.sources, (item) => {
            const score = Number(item.score || 0).toFixed(2);
            return `<strong>${item.title || "EMS Library"}</strong>${item.standard_name || "general"} | score ${score}`;
          });
        } catch (error) {
          setText(reply, `Error: ${error.message}`);
        } finally {
          send.disabled = false;
          send.textContent = "Run POC";
        }
      });

      checkHealth();
    </script>
  </body>
</html>"""


def log_event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}..."


def db() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def is_ems_related(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in EMS_KEYWORDS)


def has_daxview_scope(message: str) -> bool:
    lowered = message.lower()
    if any(phrase in lowered for phrase in DAXVIEW_GLOBAL_SCOPE_PHRASES):
        return True
    return any(re.search(pattern, lowered) for pattern in DAXVIEW_TARGET_SCOPE_PATTERNS)


def daxview_clarification_needed(message: str, tools: list[str]) -> bool:
    if not tools:
        return False
    if not any(tool in DAXVIEW_SCOPE_REQUIRED_TOOLS for tool in tools):
        return False
    lowered = message.lower()
    if any(phrase in lowered for phrase in ("how many", "count", "list", "show all", "all daxview")):
        return False
    return not has_daxview_scope(message)


def build_daxview_clarification(message: str, tools: list[str]) -> str:
    if "get_daxview_billing" in tools:
        return "Which site, building, or meter should I check for the Daxview billing summary?"
    if "get_daxview_open_alarms" in tools:
        return "Which site, building, meter, or device should I check for Daxview open alarms? You can also say \"all sites\" for a system-wide check."
    if "get_daxview_device_summary" in tools:
        return "Which Daxview device, meter, or site should I summarize? You can also say \"all devices\" for a full device summary."
    if "get_daxview_site_summary" in tools:
        return "Which Daxview site or building should I summarize? You can also say \"all sites\" for an overall summary."
    return "Which Daxview site, building, meter, or device should I check?"


def ollama_json(path: str, payload: dict, timeout: int = 300) -> dict:
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_mcp_response(raw: str) -> dict:
    """Accept normal JSON or simple SSE-style MCP responses."""
    raw = raw.strip()
    if not raw:
        return {}
    chunks = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            if payload and payload != "[DONE]":
                chunks.append(payload)
    for payload in reversed(chunks):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return json.loads(raw)


def mcp_headers(include_session: bool = True) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": DAXVIEW_MCP_PROTOCOL_VERSION,
    }
    if DAXVIEW_MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {DAXVIEW_MCP_AUTH_TOKEN}"
    if include_session and DAXVIEW_MCP_SESSION_ID:
        headers["Mcp-Session-Id"] = DAXVIEW_MCP_SESSION_ID
    return headers


def mcp_post(payload: dict, include_session: bool = True) -> tuple[dict, object]:
    if not DAXVIEW_MCP_URL:
        raise RuntimeError("DAXVIEW_MCP_URL is not configured")
    request = Request(
        DAXVIEW_MCP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=mcp_headers(include_session),
        method="POST",
    )
    with urlopen(request, timeout=DAXVIEW_MCP_TIMEOUT) as response:
        return parse_mcp_response(response.read().decode("utf-8")), response.headers


def initialize_mcp_session() -> None:
    global DAXVIEW_MCP_SESSION_ID
    if DAXVIEW_MCP_SESSION_ID:
        return
    payload = {
        "jsonrpc": "2.0",
        "id": f"initialize:{uuid.uuid4()}",
        "method": "initialize",
        "params": {
            "protocolVersion": DAXVIEW_MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "ems-chatbot-ai-server", "version": "1.0.0"},
        },
    }
    response, headers = mcp_post(payload, include_session=False)
    if response.get("error"):
        raise RuntimeError(response["error"])
    DAXVIEW_MCP_SESSION_ID = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")


def mcp_json_rpc(method: str, params: dict | None = None, request_id: str | int | None = None) -> dict:
    if method != "initialize":
        initialize_mcp_session()
    payload = {
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    response, _ = mcp_post(payload)
    return response


def call_daxview_mcp_tool(tool_name: str, arguments: dict | None, request_id: str) -> dict:
    started_at = time.perf_counter()
    log_event("mcp_tool_request", request_id=request_id, tool=tool_name, url=DAXVIEW_MCP_URL)
    response = mcp_json_rpc(
        "tools/call",
        {"name": tool_name, "arguments": arguments or {}},
        request_id=f"{request_id}:{tool_name}",
    )
    if response.get("error"):
        raise RuntimeError(response["error"])
    result = response.get("result", response)
    log_event(
        "mcp_tool_response",
        request_id=request_id,
        tool=tool_name,
        duration_ms=round((time.perf_counter() - started_at) * 1000),
    )
    return result


def select_daxview_tools(message: str) -> list[str]:
    lowered = message.lower()
    selected = []
    for tool_name, phrases in DAXVIEW_TOOL_KEYWORDS.items():
        if any(phrase in lowered for phrase in phrases):
            selected.append(tool_name)
    if "daxview" in lowered and not selected:
        selected.append("get_daxview_api_status")
    return selected[:3]


def retrieve_daxview_context(message: str, request_id: str) -> dict:
    tools = select_daxview_tools(message)
    if not DAXVIEW_MCP_ENABLED or not DAXVIEW_MCP_URL or not tools:
        return {"enabled": DAXVIEW_MCP_ENABLED, "tools": [], "results": [], "errors": []}

    results = []
    errors = []
    for tool_name in tools:
        try:
            results.append({"tool": tool_name, "result": call_daxview_mcp_tool(tool_name, {}, request_id)})
        except Exception as error:
            errors.append({"tool": tool_name, "error": str(error)})
            log_event("mcp_tool_error", request_id=request_id, tool=tool_name, error=str(error))
    return {"enabled": True, "tools": tools, "results": results, "errors": errors}


def mcp_structured_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return {}
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            text = item.get("text") if isinstance(item, dict) else None
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return result


def summarize_mcp_result(tool_name: str, result: dict) -> list[str]:
    structured = mcp_structured_result(result)
    backend = structured.get("backend_response") if isinstance(structured.get("backend_response"), dict) else {}
    lines = [f"Tool: {tool_name}"]
    if structured.get("status"):
        lines.append(f"MCP status: {structured.get('status')}")
    if structured.get("backend_endpoint"):
        lines.append(f"Backend endpoint: {structured.get('backend_endpoint')}")
    if structured.get("backend_http_status"):
        lines.append(f"Backend HTTP status: {structured.get('backend_http_status')}")
    if backend.get("count") is not None:
        lines.append(f"Returned count: {backend.get('count')}")
    alarm_count = extract_alarm_count(structured) if tool_name == "get_daxview_open_alarms" else None
    if alarm_count is not None:
        lines.append(f"Active alerts: {alarm_count}")
    devices = backend.get("devices")
    if isinstance(devices, list):
        online_count = sum(1 for device in devices if str(device.get("status", "")).lower() == "online")
        lines.append(f"Devices returned: {len(devices)}")
        lines.append(f"Online devices: {online_count}")
        for device in devices[:20]:
            lines.append(
                "- "
                f"{device.get('device_name') or device.get('remote_device_id')} | "
                f"{device.get('remote_device_id') or 'no id'} | "
                f"site {device.get('site_name') or device.get('site_id') or 'unknown'} | "
                f"status {device.get('status') or 'unknown'} | "
                f"data {device.get('data_status') or 'unknown'}"
            )
    return lines


def first_numeric_value(data: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def extract_alarm_count(structured: dict) -> int | None:
    if not isinstance(structured, dict):
        return None
    candidates = [structured]
    backend = structured.get("backend_response")
    if isinstance(backend, dict):
        candidates.insert(0, backend)
        totals = backend.get("totals")
        if isinstance(totals, dict):
            candidates.insert(0, totals)
    count_keys = (
        "active_alerts", "active_alert_count", "open_alerts", "open_alert_count",
        "active_alarms", "active_alarm_count", "open_alarms", "open_alarm_count",
        "alarm_count", "alert_count", "active_count", "open_count", "count", "total",
    )
    for candidate in candidates:
        value = first_numeric_value(candidate, count_keys)
        if value is not None:
            return value
    list_keys = ("alarms", "alerts", "open_alarms", "active_alarms", "open_alerts", "active_alerts")
    for candidate in candidates:
        for key in list_keys:
            value = candidate.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def answer_from_mcp_if_direct_count_question(message: str, mcp_context: dict) -> str | None:
    lowered = message.lower()
    if not any(phrase in lowered for phrase in ("how many", "count", "online devices", "devices online")):
        return None
    wants_alarm_count = any(
        phrase in lowered
        for phrase in ("alarm", "alarms", "alert", "alerts", "fault", "faults", "event", "events")
    )
    if wants_alarm_count:
        for item in mcp_context.get("results") or []:
            if item.get("tool") != "get_daxview_open_alarms":
                continue
            structured = mcp_structured_result(item.get("result") or {})
            alarm_count = extract_alarm_count(structured)
            if alarm_count is not None:
                return f"{alarm_count} Daxview alerts are active."
        return "I could not determine the active Daxview alert count from the MCP response."
    for item in mcp_context.get("results") or []:
        if item.get("tool") != "get_daxview_device_summary":
            continue
        structured = mcp_structured_result(item.get("result") or {})
        backend = structured.get("backend_response") if isinstance(structured.get("backend_response"), dict) else {}
        devices = backend.get("devices")
        if not isinstance(devices, list):
            continue
        online = [
            device for device in devices
            if str(device.get("status", "")).lower() == "online"
        ]
        return f"{len(online)} Daxview devices are online out of {len(devices)} returned devices."
    return None


def format_daxview_context(mcp_context: dict) -> str:
    results = mcp_context.get("results") or []
    errors = mcp_context.get("errors") or []
    if not results and not errors:
        return "No live Daxview MCP data was requested or available for this question."
    lines = []
    for item in results:
        lines.extend(summarize_mcp_result(item.get("tool"), item.get("result") or {}))
    for item in errors:
        lines.append(f"Tool {item.get('tool')} error: {item.get('error')}")
    return "\n\n".join(lines)


def daxview_trace_status(mcp_context: dict) -> tuple[str, str]:
    tools = mcp_context.get("tools") or []
    results = mcp_context.get("results") or []
    errors = mcp_context.get("errors") or []
    if not tools:
        return "skipped", "No live Daxview tool was needed for this question."
    if results and errors:
        return "partial", f"Called {len(results)} Daxview tool(s); {len(errors)} tool(s) failed."
    if results:
        return "completed", f"Called {len(results)} Daxview tool(s)."
    return "failed", f"Tried {len(tools)} Daxview tool(s); {len(errors)} failed."


def embed_text(text: str) -> list[float]:
    data = ollama_json("/api/embeddings", {"model": EMBEDDING_MODEL, "prompt": text})
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama returned an invalid embedding")
    return embedding


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            split_at = cleaned.rfind(". ", start, end)
            if split_at > start + 400:
                end = split_at + 1
        chunks.append(cleaned[start:end].strip())
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def retrieve_context(question: str) -> list[dict]:
    if not DATABASE_URL:
        return []
    embedding = vector_literal(embed_text(question))
    query = """
        SELECT
            dc.chunk_text,
            dc.page_number,
            dc.section_reference,
            d.title,
            d.standard_name,
            1 - (dc.embedding <=> %s::vector) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE 1 - (dc.embedding <=> %s::vector) >= %s
        ORDER BY dc.embedding <=> %s::vector
        LIMIT %s
    """
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (embedding, embedding, RAG_MIN_SCORE, embedding, RAG_MATCH_LIMIT))
            return list(cur.fetchall())


def build_prompt(message: str, contexts: list[dict], mcp_context: dict | None = None) -> str:
    context_block = "\n\n".join(
        f"Source {idx}: {row.get('title')} / {row.get('standard_name') or 'EMS library'}\n"
        f"{row.get('chunk_text')}"
        for idx, row in enumerate(contexts, start=1)
    )
    if not context_block:
        context_block = "No matching EMS library context was found."
    daxview_block = format_daxview_context(mcp_context or {})

    return f"""You are an Energy Management System specialist.
Answer only EMS, energy management, ISO 50001, IEC, IEEE, power monitoring, metering, tariff, demand, and electrical energy questions.
Keep the final answer simple and compact: maximum 5 short bullets or 1 short paragraph.
Use the EMS library context when relevant. If the context is insufficient, say what is missing and give a cautious EMS-focused answer.
Use live Daxview data when it is provided. If a Daxview tool failed, say live Daxview data is currently unavailable for that part.
For Janitza UMG device, voltage sag, power quality, alarm, THD, or meter troubleshooting questions, prioritize likely root causes, what readings to check, and practical EMS investigation steps.
If the user says "main cost" in a voltage sag or fault context, treat it as possibly meaning "main cause" and clarify both cause and cost impact briefly.
Do not answer unrelated general questions.

Live Daxview MCP data:
{daxview_block}

EMS library context:
{context_block}

User question:
{message}

Answer:"""


def ask_ollama(message: str, contexts: list[dict], request_id: str, mcp_context: dict | None = None) -> str:
    prompt = build_prompt(message, contexts, mcp_context)
    started_at = time.perf_counter()
    log_event("api_to_ollama_request", request_id=request_id, model=CHAT_MODEL, prompt_preview=preview(prompt))
    try:
        data = ollama_json(
            "/api/generate",
            {
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        log_event("api_to_ollama_error", request_id=request_id, error=str(error))
        raise RuntimeError(f"Ollama is not ready: {error}") from error
    reply = str(data.get("response", "")).strip()
    if not reply:
        raise RuntimeError("Ollama returned an empty response")
    log_event("api_from_ollama_response", request_id=request_id, duration_ms=round((time.perf_counter() - started_at) * 1000))
    return reply


def ask_role_agent(
    agent_name: str,
    role: str,
    model: str,
    message: str,
    contexts: list[dict],
    request_id: str,
    mcp_context: dict | None = None,
) -> str:
    context_block = "\n\n".join(
        f"{row.get('title')} / {row.get('standard_name') or 'EMS library'}\n{row.get('chunk_text')}"
        for row in contexts[:3]
    ) or "No matching EMS library context was found."
    daxview_block = format_daxview_context(mcp_context or {})
    prompt = f"""You are {agent_name}.
Role: {role}
{SAFETY_DOCTRINE}
{MATH_DOCTRINE}

Answer only from this role. Keep it to 3 compact bullets.
Use live Daxview MCP data when provided. If live data conflicts with assumptions, live data wins.
For read-only questions such as list, count, summarize, status, inventory, billing summary, or open alarm counts, report the MCP facts directly. Do not reject read-only data requests as unsafe.
If this is about voltage sag, list likely causes first, then readings/checks.
Use DIRECT REJECTION only when the user asks for a physical action that would exceed rated limits, bypass alarms, increase unsafe load, or override protection.

Live Daxview MCP data:
{daxview_block}

EMS library context:
{context_block}

Question:
{message}

{agent_name} answer:"""
    started_at = time.perf_counter()
    log_event("multi_agent_to_ollama_request", request_id=request_id, agent=agent_name, model=model)
    try:
        data = ollama_json(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama is not ready: {error}") from error
    reply = str(data.get("response", "")).strip()
    if not reply:
        raise RuntimeError(f"{agent_name} returned an empty response")
    log_event(
        "multi_agent_from_ollama_response",
        request_id=request_id,
        agent=agent_name,
        duration_ms=round((time.perf_counter() - started_at) * 1000),
    )
    return reply


def ask_synthesizer(message: str, agent_answers: list[dict], request_id: str, mcp_context: dict | None = None) -> str:
    discussion = "\n\n".join(
        f"{item['agent']} ({item['role']}):\n{item['answer']}" for item in agent_answers
    )
    prompt = f"""You are the DeepSeek Final Decision Maker for an EMS chatbot.
{SAFETY_DOCTRINE}
{MATH_DOCTRINE}

Two specialist agents answered the same EMS question. Use their answers internally, then return only the final user-facing answer.
Do not mention which agent is better.
Do not mention "better final answer".
Do not explain that the response is professional, concise, strict, compliant, or follows guidelines.
Do not include separators like "---".
Do not reveal the discussion process.
Write naturally like a senior EMS engineer speaking to an operator: clear, practical, and human.
Must answer the user's actual question first. For voltage sag, start with likely causes, then EMS actions.
Use live Daxview MCP data when it is provided. If a Daxview tool failed, clearly say live Daxview data is unavailable before giving a general EMS answer.
For read-only questions such as list, count, summarize, status, inventory, billing summary, or open alarm counts, answer directly from the Live Daxview MCP data. Do not invent hazards or recommend Load Shedding unless the MCP data explicitly reports an unsafe operating condition or the user asks for an unsafe physical action.
Safety hard limits override energy saving, user preference, uptime, cost, and comfort.
If any rated hardware limit is explicitly exceeded or the user asks to add load/bypass an alarm, reject immediately using exactly these sections:
DIRECT REJECTION:
PHYSICAL REASONING:
MITIGATION ACTION:
Keep the final answer simple, compact, precise, and maximum 5 short bullets.

Question:
{message}

Live Daxview MCP data:
{format_daxview_context(mcp_context or {})}

Specialist discussion:
{discussion}

Final answer only:"""
    started_at = time.perf_counter()
    log_event("synthesizer_to_ollama_request", request_id=request_id, model=DEEPSEEK_MODEL)
    try:
        data = ollama_json(
            "/api/generate",
            {
                "model": DEEPSEEK_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.15},
            },
        )
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama is not ready: {error}") from error
    reply = str(data.get("response", "")).strip()
    if not reply:
        raise RuntimeError("Synthesizer returned an empty response")
    log_event("synthesizer_from_ollama_response", request_id=request_id, model=DEEPSEEK_MODEL, duration_ms=round((time.perf_counter() - started_at) * 1000))
    return clean_final_answer(reply)


def clean_final_answer(reply: str) -> str:
    lines = []
    skip_patterns = (
        "the better final answer is",
        "better final answer",
        "this response is",
        "this answer is",
        "as the final decision maker",
    )
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        lowered = line.lower().strip("* ")
        if not line or line == "---":
            continue
        if any(pattern in lowered for pattern in skip_patterns):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip() or reply.strip()


def langchain_start_state(payload: dict) -> dict:
    message = str(payload.get("message", "")).strip()
    return {
        "message": message,
        "request_id": payload.get("request_id") or str(uuid.uuid4()),
        "session_id": payload.get("session_id"),
        "mode": payload.get("mode", "chat"),
        "started_at": payload.get("started_at") or time.perf_counter(),
        "contexts": [],
        "sources": [],
        "mcp_context": {"enabled": DAXVIEW_MCP_ENABLED, "tools": [], "results": [], "errors": []},
        "agent_answers": [],
        "agent_trace": [],
        "reply": "",
        "provider": "ems-guard",
        "model": None,
    }


def langchain_classify_and_validate(state: dict) -> dict:
    message = state["message"]
    related = is_ems_related(message)
    selected_tools = select_daxview_tools(message) if related else []
    state.update(
        {
            "is_ems_related": related,
            "selected_tools": selected_tools,
            "needs_clarification": related and daxview_clarification_needed(message, selected_tools),
        }
    )
    if not related:
        state["reply"] = REFUSAL
        state["provider"] = "ems-guard"
    elif state["needs_clarification"]:
        state["reply"] = build_daxview_clarification(message, selected_tools)
        state["provider"] = "daxview-question-filter"
        state["mcp_context"] = {"enabled": DAXVIEW_MCP_ENABLED, "tools": selected_tools, "results": [], "errors": []}
    return state


def langchain_retrieve_context(state: dict) -> dict:
    if not state.get("is_ems_related") or state.get("needs_clarification"):
        return state
    message = state["message"]
    contexts = retrieve_context(message)
    state["contexts"] = contexts
    state["sources"] = [
        {
            "title": row.get("title"),
            "standard_name": row.get("standard_name"),
            "score": float(row.get("score") or 0),
            "section_reference": row.get("section_reference"),
        }
        for row in contexts
    ]
    state["mcp_context"] = retrieve_daxview_context(message, state["request_id"])
    return state


def langchain_direct_mcp_answer(state: dict) -> dict:
    if state.get("reply") or not state.get("is_ems_related"):
        return state
    direct_answer = answer_from_mcp_if_direct_count_question(state["message"], state["mcp_context"])
    if direct_answer:
        state["reply"] = direct_answer
        state["provider"] = "daxview-mcp-direct"
        state["model"] = None
        state["decision_model"] = "Daxview MCP direct extractor"
        state["agent_answers"] = [
            {
                "agent": "Daxview MCP Direct Extractor",
                "role": "Read structured MCP data and answer deterministic count/status questions.",
                "model": "deterministic",
                "answer": direct_answer,
            }
        ]
    return state


def langchain_generate_chat_answer(state: dict) -> dict:
    if state.get("reply") or not state.get("is_ems_related"):
        return state
    state["reply"] = ask_ollama(state["message"], state["contexts"], state["request_id"], state["mcp_context"])
    state["provider"] = "ollama"
    state["model"] = CHAT_MODEL
    return state


def langchain_generate_multi_agent_answer(state: dict) -> dict:
    if state.get("reply") or not state.get("is_ems_related"):
        return state
    message = state["message"]
    contexts = state["contexts"]
    request_id = state["request_id"]
    mcp_context = state["mcp_context"]
    agent_answers = [
        {
            "agent": "Agent 1 - Ollama EMS Triage",
            "role": "Quickly classify the issue and identify immediate EMS checks.",
            "model": OLLAMA_MODEL,
            "answer": ask_role_agent(
                "Agent 1 - Ollama EMS Triage",
                "Quickly classify the issue and identify immediate EMS checks.",
                OLLAMA_MODEL,
                message,
                contexts,
                request_id,
                mcp_context,
            ),
        },
        {
            "agent": "Agent 2 - Qwen Power Quality",
            "role": "Find likely electrical root causes and UMG meter readings to inspect.",
            "model": OLLAMA_MODEL,
            "answer": ask_role_agent(
                "Agent 2 - Qwen Power Quality",
                "Find likely electrical root causes and UMG meter readings to inspect.",
                OLLAMA_MODEL,
                message,
                contexts,
                request_id,
                mcp_context,
            ),
        },
    ]
    state["agent_answers"] = agent_answers
    state["reply"] = ask_synthesizer(message, agent_answers, request_id, mcp_context)
    state["provider"] = "multi-agent-poc"
    state["model"] = DEEPSEEK_MODEL
    state["decision_model"] = DEEPSEEK_MODEL
    return state


def langchain_persist_and_trace(state: dict) -> dict:
    related = bool(state.get("is_ems_related"))
    qa_id = store_chat(
        state["message"],
        state["reply"],
        related,
        state.get("sources") or [],
        state.get("session_id"),
    )
    state["qa_log_id"] = qa_id
    mcp_context = state.get("mcp_context") or {}
    if not related:
        state["agent_trace"] = build_agent_trace(False, [], state["reply"], qa_id, mcp_context)
    elif state.get("needs_clarification"):
        state["agent_trace"] = [
            {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/Daxview related."},
            {"agent": "Question Completeness Filter", "status": "needs_clarification", "detail": "Daxview data request is missing site, building, meter, device, or explicit all-scope target."},
            {"agent": "Daxview MCP", "status": "skipped", "detail": "MCP was not called because the question needs clarification first."},
            {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
            {"agent": "Final Response", "status": "completed", "detail": preview(state["reply"], 120)},
        ]
    elif state.get("provider") == "daxview-mcp-direct":
        mcp_status, mcp_detail = daxview_trace_status(mcp_context)
        state["agent_trace"] = [
            {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/Daxview related."},
            {"agent": "Knowledge Retriever", "status": "completed", "detail": f"Found {len(state.get('contexts') or [])} matching EMS library source(s)."},
            {"agent": "Daxview MCP", "status": mcp_status, "detail": mcp_detail},
            {"agent": "MCP Direct Extractor", "status": "completed", "detail": "Answered directly from structured MCP device data."},
            {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
            {"agent": "Final Response", "status": "completed", "detail": preview(state["reply"], 120)},
        ]
    else:
        state["agent_trace"] = build_agent_trace(
            True,
            state.get("contexts") or [],
            state["reply"],
            qa_id,
            mcp_context,
        )
        if state.get("mode") == "multi-agent":
            mcp_status, mcp_detail = daxview_trace_status(mcp_context)
            state["agent_trace"] = [
                {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/power related."},
                {"agent": "Knowledge Retriever", "status": "completed", "detail": f"Found {len(state.get('contexts') or [])} EMS source(s)."},
                {"agent": "Daxview MCP", "status": mcp_status, "detail": mcp_detail},
                {"agent": "Agent 1 - Ollama EMS Triage", "status": "completed", "detail": f"Answered with {OLLAMA_MODEL}."},
                {"agent": "Agent 2 - Qwen Power Quality", "status": "completed", "detail": f"Answered with {OLLAMA_MODEL}."},
                {"agent": "DeepSeek Final Decision Maker", "status": "completed", "detail": f"Combined both answers with {DEEPSEEK_MODEL}."},
                {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
            ]
    return state


CHAT_LANGCHAIN = (
    RunnableLambda(langchain_start_state)
    | RunnableLambda(langchain_classify_and_validate)
    | RunnableLambda(langchain_retrieve_context)
    | RunnableLambda(langchain_direct_mcp_answer)
    | RunnableLambda(langchain_generate_chat_answer)
    | RunnableLambda(langchain_persist_and_trace)
)

MULTI_AGENT_LANGCHAIN = (
    RunnableLambda(langchain_start_state)
    | RunnableLambda(langchain_classify_and_validate)
    | RunnableLambda(langchain_retrieve_context)
    | RunnableLambda(langchain_direct_mcp_answer)
    | RunnableLambda(langchain_generate_multi_agent_answer)
    | RunnableLambda(langchain_persist_and_trace)
)


def langchain_chat_response(message: str, request_id: str, session_id: str | None) -> dict:
    state = CHAT_LANGCHAIN.invoke(
        {
            "message": message,
            "request_id": request_id,
            "session_id": session_id,
            "mode": "chat",
            "started_at": time.perf_counter(),
        }
    )
    return {
        "reply": state["reply"],
        "provider": state.get("provider"),
        "model": state.get("model"),
        "is_ems_related": state.get("is_ems_related"),
        "sources": state.get("sources") or [],
        "agent_trace": state.get("agent_trace") or [],
        "daxview_mcp": state.get("mcp_context"),
        "mcp_summary": format_daxview_context(state.get("mcp_context") or {}),
        "qa_log_id": state.get("qa_log_id"),
    }


def langchain_multi_agent_response(message: str, request_id: str) -> dict:
    started_at = time.perf_counter()
    state = MULTI_AGENT_LANGCHAIN.invoke(
        {
            "message": message,
            "request_id": request_id,
            "session_id": None,
            "mode": "multi-agent",
            "started_at": started_at,
        }
    )
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    return {
        "reply": state["reply"],
        "provider": state.get("provider"),
        "model": state.get("model"),
        "decision_model": state.get("decision_model") or state.get("model"),
        "is_ems_related": state.get("is_ems_related"),
        "sources": state.get("sources") or [],
        "agent_discussion": state.get("agent_answers") or [],
        "agent_trace": state.get("agent_trace") or [],
        "daxview_mcp": state.get("mcp_context"),
        "mcp_summary": format_daxview_context(state.get("mcp_context") or {}),
        "qa_log_id": state.get("qa_log_id"),
        "duration_ms": duration_ms,
        "processing_time_seconds": duration_ms / 1000,
    }


def run_multi_agent_poc(message: str, request_id: str) -> dict:
    return langchain_multi_agent_response(message, request_id)


def run_legacy_multi_agent_poc(message: str, request_id: str) -> dict:
    started_at = time.perf_counter()
    related = is_ems_related(message)
    if not related:
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "reply": REFUSAL,
            "provider": "ems-guard",
            "model": None,
            "is_ems_related": False,
            "agent_discussion": [],
            "agent_trace": build_agent_trace(False, [], REFUSAL),
            "sources": [],
            "duration_ms": duration_ms,
            "processing_time_seconds": duration_ms / 1000,
        }

    selected_tools = select_daxview_tools(message)
    if daxview_clarification_needed(message, selected_tools):
        reply = build_daxview_clarification(message, selected_tools)
        qa_id = store_chat(message, reply, True, [], None)
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "reply": reply,
            "provider": "daxview-question-filter",
            "model": None,
            "decision_model": None,
            "is_ems_related": True,
            "sources": [],
            "agent_discussion": [],
            "agent_trace": [
                {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/Daxview related."},
                {"agent": "Question Completeness Filter", "status": "needs_clarification", "detail": "Daxview data request is missing site, building, meter, device, or explicit all-scope target."},
                {"agent": "Daxview MCP", "status": "skipped", "detail": "MCP was not called because the question needs clarification first."},
                {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
            ],
            "daxview_mcp": {"enabled": DAXVIEW_MCP_ENABLED, "tools": selected_tools, "results": [], "errors": []},
            "mcp_summary": "No MCP tool was called because the question needs clarification first.",
            "qa_log_id": qa_id,
            "duration_ms": duration_ms,
            "processing_time_seconds": duration_ms / 1000,
        }

    contexts = retrieve_context(message)
    mcp_context = retrieve_daxview_context(message, request_id)
    sources = [
        {
            "title": row.get("title"),
            "standard_name": row.get("standard_name"),
            "score": float(row.get("score") or 0),
            "section_reference": row.get("section_reference"),
        }
        for row in contexts
    ]
    direct_answer = answer_from_mcp_if_direct_count_question(message, mcp_context)
    if direct_answer:
        qa_id = store_chat(message, direct_answer, True, sources, None)
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        mcp_status, mcp_detail = daxview_trace_status(mcp_context)
        return {
            "reply": direct_answer,
            "provider": "daxview-mcp-direct",
            "model": None,
            "decision_model": "Daxview MCP direct extractor",
            "is_ems_related": True,
            "sources": sources,
            "agent_discussion": [
                {
                    "agent": "Daxview MCP Direct Extractor",
                    "role": "Read structured MCP data and answer deterministic count/status questions.",
                    "model": "deterministic",
                    "answer": direct_answer,
                }
            ],
            "agent_trace": [
                {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/Daxview related."},
                {"agent": "Knowledge Retriever", "status": "completed", "detail": f"Found {len(contexts)} EMS source(s)."},
                {"agent": "Daxview MCP", "status": mcp_status, "detail": mcp_detail},
                {"agent": "MCP Direct Extractor", "status": "completed", "detail": "Answered directly from structured MCP device data."},
                {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
            ],
            "daxview_mcp": mcp_context,
            "mcp_summary": format_daxview_context(mcp_context),
            "qa_log_id": qa_id,
            "duration_ms": duration_ms,
            "processing_time_seconds": duration_ms / 1000,
        }
    agent_answers = [
        {
            "agent": "Agent 1 - Ollama EMS Triage",
            "role": "Quickly classify the issue and identify immediate EMS checks.",
            "model": OLLAMA_MODEL,
            "answer": ask_role_agent(
                "Agent 1 - Ollama EMS Triage",
                "Quickly classify the issue and identify immediate EMS checks.",
                OLLAMA_MODEL,
                message,
                contexts,
                request_id,
                mcp_context,
            ),
        },
        {
            "agent": "Agent 2 - Qwen Power Quality",
            "role": "Find likely electrical root causes and UMG meter readings to inspect.",
            "model": OLLAMA_MODEL,
            "answer": ask_role_agent(
                "Agent 2 - Qwen Power Quality",
                "Find likely electrical root causes and UMG meter readings to inspect.",
                OLLAMA_MODEL,
                message,
                contexts,
                request_id,
                mcp_context,
            ),
        },
    ]
    final_answer = ask_synthesizer(message, agent_answers, request_id, mcp_context)
    qa_id = store_chat(message, final_answer, True, sources, None)
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    mcp_status, mcp_detail = daxview_trace_status(mcp_context)
    agent_trace = [
        {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/power related."},
        {"agent": "Knowledge Retriever", "status": "completed", "detail": f"Found {len(contexts)} EMS source(s)."},
        {
            "agent": "Daxview MCP",
            "status": mcp_status,
            "detail": mcp_detail,
        },
        {"agent": "Agent 1 - Ollama EMS Triage", "status": "completed", "detail": f"Answered with {OLLAMA_MODEL}."},
        {"agent": "Agent 2 - Qwen Power Quality", "status": "completed", "detail": f"Answered with {OLLAMA_MODEL}."},
        {"agent": "DeepSeek Final Decision Maker", "status": "completed", "detail": f"Combined both answers with {DEEPSEEK_MODEL}."},
        {"agent": "DB Logger", "status": "completed", "detail": f"Saved QA log {qa_id}."},
    ]
    return {
        "reply": final_answer,
        "provider": "multi-agent-poc",
        "model": DEEPSEEK_MODEL,
        "decision_model": DEEPSEEK_MODEL,
        "is_ems_related": True,
        "sources": sources,
        "agent_discussion": agent_answers,
        "agent_trace": agent_trace,
        "daxview_mcp": mcp_context,
        "mcp_summary": format_daxview_context(mcp_context),
        "qa_log_id": qa_id,
        "duration_ms": duration_ms,
        "processing_time_seconds": duration_ms / 1000,
    }


def build_agent_trace(
    related: bool,
    contexts: list[dict],
    reply: str,
    qa_id: str | None = None,
    mcp_context: dict | None = None,
) -> list[dict]:
    trace = [
        {
            "agent": "EMS Guard",
            "status": "passed" if related else "blocked",
            "detail": "Question is EMS/power related." if related else "Question is outside EMS scope.",
        }
    ]
    if related:
        trace.append(
            {
                "agent": "Knowledge Retriever",
                "status": "completed",
                "detail": f"Found {len(contexts)} matching EMS library source(s).",
            }
        )
        mcp_context = mcp_context or {}
        mcp_status, mcp_detail = daxview_trace_status(mcp_context)
        trace.append(
            {
                "agent": "Daxview MCP",
                "status": mcp_status,
                "detail": mcp_detail,
            }
        )
        trace.append(
            {
                "agent": "Answer Generator",
                "status": "completed",
                "detail": f"Generated compact answer with {CHAT_MODEL}.",
            }
        )
    else:
        trace.append(
            {
                "agent": "Answer Generator",
                "status": "skipped",
                "detail": "LLM was not called because the EMS guard blocked the question.",
            }
        )
    trace.append(
        {
            "agent": "DB Logger",
            "status": "completed" if qa_id else "skipped",
            "detail": f"Saved QA log {qa_id}." if qa_id else "No QA log id was created.",
        }
    )
    trace.append(
        {
            "agent": "Final Response",
            "status": "completed",
            "detail": preview(reply, 120),
        }
    )
    return trace


def store_chat(question: str, answer: str, is_related: bool, sources: list[dict], session_id: str | None) -> str:
    qa_id = str(uuid.uuid4())
    if not DATABASE_URL:
        return qa_id
    session = session_id or str(uuid.uuid4())
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO chat_sessions (id) VALUES (%s) ON CONFLICT (id) DO NOTHING", (session,))
            cur.execute(
                "INSERT INTO chat_messages (id, session_id, role, message) VALUES (%s, %s, 'user', %s)",
                (str(uuid.uuid4()), session, question),
            )
            cur.execute(
                "INSERT INTO chat_messages (id, session_id, role, message) VALUES (%s, %s, 'assistant', %s)",
                (str(uuid.uuid4()), session, answer),
            )
            cur.execute(
                "INSERT INTO qa_logs (id, session_id, question, answer, is_ems_related, sources_used) VALUES (%s, %s, %s, %s, %s, %s)",
                (qa_id, session, question, answer, is_related, json.dumps(sources)),
            )
            if is_related:
                cur.execute(
                    "INSERT INTO qa_embeddings (id, qa_log_id, question, answer, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                    (str(uuid.uuid4()), qa_id, question, answer, vector_literal(embed_text(question))),
                )
        conn.commit()
    return qa_id


def ingest_document(title: str, text: str, source_type: str = "manual", standard_name: str | None = None, file_name: str | None = None) -> dict:
    if not title:
        raise ValueError("title is required")
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("text did not contain ingestible content")
    document_id = str(uuid.uuid4())
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, title, source_type, standard_name, file_name) VALUES (%s, %s, %s, %s, %s)",
                (document_id, title, source_type, standard_name, file_name),
            )
            for chunk in chunks:
                cur.execute(
                    "INSERT INTO document_chunks (id, document_id, chunk_text, embedding) VALUES (%s, %s, %s, %s::vector)",
                    (str(uuid.uuid4()), document_id, chunk, vector_literal(embed_text(chunk))),
                )
        conn.commit()
    return {"document_id": document_id, "chunks": len(chunks)}


class ChatHandler(BaseHTTPRequestHandler):
    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, POC_PAGE)
            return
        if path == "/health":
            db_status = "disabled"
            if DATABASE_URL:
                try:
                    with db() as conn:
                        conn.execute("SELECT 1")
                    db_status = "ok"
                except Exception as error:
                    db_status = f"error: {error}"
            self._send_json(200, {"status": "ok", "service": "AI-Server", "database": db_status})
            return
        if path == "/debug/traces":
            self._send_json(200, {"traces": list(TRACES)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if path == "/knowledge":
            try:
                result = ingest_document(
                    title=str(request.get("title", "")).strip(),
                    text=str(request.get("text", "")).strip(),
                    source_type=str(request.get("source_type", "manual")).strip() or "manual",
                    standard_name=str(request.get("standard_name", "")).strip() or None,
                    file_name=str(request.get("file_name", "")).strip() or None,
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
                return
            self._send_json(201, result)
            return

        if path == "/multi-agent-chat":
            message = str(request.get("message", "")).strip()
            if not message:
                self._send_json(400, {"error": "message is required"})
                return
            request_id = str(uuid.uuid4())
            try:
                result = run_multi_agent_poc(message, request_id)
            except RuntimeError as error:
                self._send_json(503, {"error": str(error), "provider": "ollama"})
                return
            self._send_json(200, result)
            return

        if path != "/chat":
            self._send_json(404, {"error": "Not found"})
            return

        message = str(request.get("message", "")).strip()
        session_id = str(request.get("session_id", "")).strip() or None
        if not message:
            self._send_json(400, {"error": "message is required"})
            return

        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        try:
            response_payload = langchain_chat_response(message, request_id, session_id)
        except RuntimeError as error:
            log_event("api_to_ui_error", request_id=request_id, error=str(error))
            self._send_json(503, {"error": str(error), "provider": "ollama"})
            return

        trace = {
            "request_id": request_id,
            "status": "ok",
            "is_ems_related": response_payload.get("is_ems_related"),
            "sources": response_payload.get("sources") or [],
            "daxview_mcp": response_payload.get("daxview_mcp"),
            "mcp_summary": response_payload.get("mcp_summary"),
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        }
        TRACES.appendleft(trace)
        self._send_json(200, response_payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"AI-Server listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAI-Server stopped")
    finally:
        server.server_close()
