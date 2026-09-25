"""EMS-only chatbot API with Ollama, PostgreSQL, and pgvector RAG."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from collections import deque
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
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
OLLAMA_GENERATE_TIMEOUT = int(os.getenv("OLLAMA_GENERATE_TIMEOUT", "120"))
OLLAMA_EMBEDDING_TIMEOUT = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
RAG_MATCH_LIMIT = int(os.getenv("RAG_MATCH_LIMIT", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
TRACE_LIMIT = int(os.getenv("CHATBOT_TRACE_LIMIT", "25"))
DAXVIEW_MCP_ENABLED = os.getenv("DAXVIEW_MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_MCP_URL = os.getenv("DAXVIEW_MCP_URL", "").strip()
DAXVIEW_MCP_AUTH_TOKEN = os.getenv("DAXVIEW_MCP_AUTH_TOKEN", "").strip()
DAXVIEW_MCP_TIMEOUT = int(os.getenv("DAXVIEW_MCP_TIMEOUT", "20"))
DAXVIEW_MCP_PROTOCOL_VERSION = os.getenv("DAXVIEW_MCP_PROTOCOL_VERSION", "2025-06-18")
DAXVIEW_MCP_DEBUG_RESPONSE = os.getenv("DAXVIEW_MCP_DEBUG_RESPONSE", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_MCP_DEBUG_RESPONSE_LIMIT = int(os.getenv("DAXVIEW_MCP_DEBUG_RESPONSE_LIMIT", "4000"))
DAXVIEW_MCP_SESSION_ID = None
DAXVIEW_DEPLOYMENT_ID = os.getenv("DAXVIEW_DEPLOYMENT_ID", "v2-dev")
AI_SERVER_API_KEY = os.getenv("AI_SERVER_API_KEY", "").strip()
AI_SERVER_API_KEY_PREVIOUS = os.getenv("AI_SERVER_API_KEY_PREVIOUS", "").strip()
AI_SERVER_AUDIENCE = os.getenv("AI_SERVER_AUDIENCE", "daxview-ai").strip()
AI_CHAT_ASSERTION_SECRET = os.getenv("AI_CHAT_ASSERTION_SECRET", "").strip()
AI_JOB_WORKERS = int(os.getenv("AI_JOB_WORKERS", "4"))
DAXVIEW_ALLOW_MISSING_COMPANY = os.getenv("DAXVIEW_ALLOW_MISSING_COMPANY", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_CALLBACK_BASE_URL = os.getenv("DAXVIEW_CALLBACK_BASE_URL", "").strip().rstrip("/")
DAXVIEW_CALLBACK_KEY = os.getenv("DAXVIEW_CALLBACK_KEY", "").strip()
DAXVIEW_CALLBACK_KEY_PREVIOUS = os.getenv("DAXVIEW_CALLBACK_KEY_PREVIOUS", "").strip()
DAXVIEW_ALLOWED_HISTORICAL_TOOLS = {
    "telemetry_top_consumers",
    "site_energy_summary",
    "alarm_frequency_summary",
}
DAXVIEW_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=AI_JOB_WORKERS)
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
    "telemetry_top_consumers": {
        "top consumer", "top consumers", "most energy", "highest usage",
        "highest consumption", "largest load", "biggest consumer",
        "top consuming", "energy-consuming", "energy consuming",
        "top 5", "top five", "top devices",
    },
    "site_energy_summary": {
        "energy summary", "site energy", "usage trend", "consumption trend",
        "kwh summary", "last 7 days", "weekly energy", "daily energy",
        "energy consumption", "consumption", "usage", "difference in energy",
        "energy difference",
    },
    "alarm_frequency_summary": {
        "alarm frequency", "frequent alarm", "most alarms", "alarm summary",
        "alarm history", "repeated alarms", "historical alarms",
    },
}

DAXVIEW_SCOPE_REQUIRED_TOOLS = {
    "telemetry_top_consumers",
    "site_energy_summary",
    "alarm_frequency_summary",
}

ENERGY_COMPLIANCE_CONTEXT = [
    {
        "label": "Data protocol",
        "standard": "Janitza UMG / Modbus, BACnet, SNMP",
        "note": "Janitza UMG devices commonly expose measured values through Modbus RTU/TCP, optional BACnet/IP, SNMP, and related Ethernet services depending on model and license.",
    },
    {
        "label": "Energy management",
        "standard": "ISO 50001 / ISO 50006",
        "note": "kWh consumption can be used for an EnMS, energy baseline, and EnPI tracking, but ISO conformity belongs to the management process, not to one isolated reading.",
    },
    {
        "label": "Metering accuracy",
        "standard": "IEC 62053 series / device active-energy class",
        "note": "Active-energy accuracy depends on the meter model, CT/PT setup, calibration, and rated class; treat reported kWh as meter data unless calibration evidence is available.",
    },
    {
        "label": "Power quality context",
        "standard": "IEC 61000-4-30 / IEEE 1159",
        "note": "These are relevant when the same device data is used for voltage sag, swell, interruption, harmonics, or disturbance analysis rather than simple consumption totals.",
    },
]

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
          <textarea id="message" placeholder="Ask an EMS or Daxview question..."></textarea>
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
      const POC_BROWSER_TIMEOUT_MS = 180000;

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
        return lines.join("\\n");
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
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        try {
          const response = await fetch("/health", { signal: controller.signal });
          clearTimeout(timeout);
          const data = await response.json();
          statusBox.textContent = data.database === "ok" ? "API + DB online" : "API online";
        } catch {
          clearTimeout(timeout);
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
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), POC_BROWSER_TIMEOUT_MS);
          const response = await fetch("/multi-agent-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message.value.trim() }),
            signal: controller.signal,
          });
          clearTimeout(timeout);
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
          const message = error.name === "AbortError"
            ? "Error: The POC took too long to respond. Try a shorter question, check Ollama load, or increase POC_BROWSER_TIMEOUT_MS."
            : `Error: ${error.message}`;
          setText(reply, message);
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
    sensitive_markers = ("key", "token", "secret", "assertion", "authorization", "prompt", "answer", "message")
    safe_fields = {
        name: "[redacted]" if any(marker in name.lower() for marker in sensitive_markers) else value
        for name, value in fields.items()
    }
    print(json.dumps({"event": event, **safe_fields}, ensure_ascii=False), flush=True)


def preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}..."


def redact_debug_value(value):
    sensitive_markers = ("key", "token", "secret", "assertion", "authorization", "authorization_id")
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in sensitive_markers):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact_debug_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_debug_value(item) for item in value[:20]]
    return value


def debug_json_sample(value, limit: int) -> str:
    text = json.dumps(redact_debug_value(value), ensure_ascii=False, default=str)
    if len(text) > limit:
        return f"{text[:limit]}...[truncated]"
    return text


def mcp_result_shape(value: object) -> dict:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    shape = {"top_keys": sorted(str(key) for key in value.keys())}
    structured = value.get("structuredContent")
    if isinstance(structured, dict):
        shape["structured_keys"] = sorted(str(key) for key in structured.keys())
        backend = structured.get("backend_response")
        if isinstance(backend, dict):
            shape["backend_keys"] = sorted(str(key) for key in backend.keys())
            for list_key in ("items", "devices", "rows", "results", "data", "buckets", "alarms"):
                items = backend.get(list_key)
                if isinstance(items, list):
                    shape["list_key"] = list_key
                    shape["item_count"] = len(items)
                    if items and isinstance(items[0], dict):
                        shape["sample_item_keys"] = sorted(str(key) for key in items[0].keys())
                    break
    return shape


def db() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def json_error(code: str, message: str, retryable: bool = False, request_id: str | None = None) -> dict:
    error = {"code": code, "message": message, "retryable": retryable}
    if request_id:
        error["request_id"] = request_id
    return {"error": error}


def b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def verify_hs256_jwt(token: str, secret: str) -> dict:
    if not secret:
        raise ValueError("AI_CHAT_ASSERTION_SECRET is not configured")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("assertion must be a JWT")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    signature = b64url_decode(parts[2])
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid assertion signature")
    header = json.loads(b64url_decode(parts[0]).decode("utf-8"))
    if header.get("alg") != "HS256":
        raise ValueError("unsupported assertion algorithm")
    claims = json.loads(b64url_decode(parts[1]).decode("utf-8"))
    now = int(time.time())
    exp = int(claims.get("exp", 0))
    iat = int(claims.get("iat", 0))
    if exp < now:
        raise ValueError("assertion expired")
    if iat > now + 60:
        raise ValueError("assertion issued in the future")
    if claims.get("aud") != AI_SERVER_AUDIENCE:
        raise ValueError("wrong assertion audience")
    if claims.get("iss") != DAXVIEW_DEPLOYMENT_ID:
        raise ValueError("wrong deployment issuer")
    if claims.get("sub") in (None, ""):
        raise ValueError("missing assertion subject")
    if claims.get("company_id") in (None, "") and not DAXVIEW_ALLOW_MISSING_COMPANY:
        raise ValueError("missing assertion company")
    if claims.get("company_id") in (None, ""):
        claims["company_id"] = "missing-company"
    return claims


def daxview_identity(headers) -> dict:
    auth_header = headers.get("Authorization", "")
    expected = {key for key in (AI_SERVER_API_KEY, AI_SERVER_API_KEY_PREVIOUS) if key}
    if not expected:
        raise PermissionError("AI_SERVER_API_KEY is not configured")
    if not auth_header.startswith("Bearer "):
        raise PermissionError("missing bearer service key")
    supplied_key = auth_header.removeprefix("Bearer ").strip()
    if supplied_key not in expected:
        raise PermissionError("invalid service key")
    assertion = headers.get("X-DaxView-Assertion", "").strip()
    if not assertion:
        raise PermissionError("missing DaxView assertion")
    claims = verify_hs256_jwt(assertion, AI_CHAT_ASSERTION_SECRET)
    return {
        "deployment_id": str(claims["iss"]),
        "company_id": str(claims["company_id"]),
        "user_id": str(claims["sub"]),
        "claims": claims,
    }


def require_database() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")


def safe_job_id(turn_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_-]", "", turn_id)
    return f"job_{compact[:48] or uuid.uuid4().hex}"


def add_job_event(job_id: str, event_type: str, data: dict) -> int:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(event_id), 0) + 1 AS next_id FROM daxview_job_events WHERE job_id = %s",
                (job_id,),
            )
            event_id = int(cur.fetchone()["next_id"])
            cur.execute(
                "INSERT INTO daxview_job_events (job_id, event_id, event_type, event_data) VALUES (%s, %s, %s, %s)",
                (job_id, event_id, event_type, json.dumps(data)),
            )
        conn.commit()
    return event_id


def is_job_cancelled(job_id: str) -> bool:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM daxview_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            return bool(row and row.get("status") == "cancelled")


def update_job_status(job_id: str, status: str) -> None:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE daxview_jobs SET status = %s, updated_at = now() WHERE id = %s",
                (status, job_id),
            )
        conn.commit()


def select_historical_operation(message: str) -> str | None:
    operations = select_historical_operations(message)
    return operations[0] if operations else None


def select_historical_operations(message: str) -> list[str]:
    lowered = message.lower()
    operations = []
    if any(
        phrase in lowered
        for phrase in (
            "top consumer",
            "top consumers",
            "top consuming",
            "energy-consuming",
            "energy consuming",
            "top 5",
            "top five",
            "top devices",
            "most energy",
            "highest usage",
            "highest consumption",
            "largest load",
            "biggest consumer",
        )
    ):
        operations.append("telemetry_top_consumers")
    if any(
        phrase in lowered
        for phrase in (
            "energy summary",
            "site energy",
            "usage trend",
            "consumption trend",
            "kwh summary",
            "energy consumption",
            "consumption",
            "usage",
            "difference in energy",
            "energy difference",
        )
    ) or (
        any(word in lowered for word in ("energy", "kwh", "consumption", "usage"))
        and any(word in lowered for word in ("compare", "comparison", "difference", "between"))
        and any(month in lowered for month in MONTH_NAMES)
    ):
        operations.append("site_energy_summary")
    if any(phrase in lowered for phrase in ("alarm frequency", "frequent alarm", "most alarms", "alarm summary")):
        operations.append("alarm_frequency_summary")
    return operations


def default_historical_range() -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timezone": "Asia/Kuala_Lumpur",
    }


def relative_historical_range(message: str) -> tuple[datetime, datetime] | None:
    lowered = message.lower()
    end = datetime.now(timezone.utc)
    match = re.search(r"\b(?:last\s+|for\s+)?(\d{1,3})\s*[- ]?\s*(day|days|week|weeks|month|months)\b", lowered)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("week"):
        days = amount * 7
    elif unit.startswith("month"):
        days = amount * 30
    else:
        days = amount
    return end - timedelta(days=max(days, 1)), end


def wants_relative_summary(message: str) -> bool:
    return relative_historical_range(message) is not None


def requested_month_ranges(message: str, fallback_year: int) -> list[tuple[date, date]]:
    lowered = message.lower()
    ranges = []
    seen = set()
    for match in re.finditer(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"(?:\s+(\d{4}))?\b",
        lowered,
    ):
        month = MONTH_NAMES[match.group(1)]
        year = int(match.group(2) or fallback_year)
        key = (year, month)
        if key in seen:
            continue
        seen.add(key)
        start_date = date(year, month, 1)
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        ranges.append((start_date, next_month))
    return ranges


def requested_historical_range(message: str) -> dict:
    lowered = message.lower()
    end = datetime.now(timezone.utc)
    requested_dates = requested_comparison_dates(message, end.year)
    requested_months = [] if requested_dates else requested_month_ranges(message, end.year)
    relative_range = relative_historical_range(message)
    if requested_dates and relative_range:
        start, end = relative_range
    elif requested_dates:
        parsed_dates = [datetime.fromisoformat(date).date() for date in requested_dates]
        start_date = min(parsed_dates)
        end_date = max(parsed_dates)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
    elif requested_months:
        start_date = min(month_start for month_start, _ in requested_months)
        end_date = max(month_end for _, month_end in requested_months) - timedelta(days=1)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
    elif "today" in lowered:
        start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    elif "yesterday" in lowered:
        today_start = end.replace(hour=0, minute=0, second=0, microsecond=0)
        start = today_start - timedelta(days=1)
        end = today_start
    else:
        relative_range = relative_historical_range(message)
        if not relative_range:
            return default_historical_range()
        start, end = relative_range
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timezone": "Asia/Kuala_Lumpur",
    }


def build_historical_arguments(operation_id: str, context: dict, message: str = "") -> dict:
    site_id = context.get("site_id")
    if not site_id:
        raise ValueError("site_id is required for Daxview historical data")
    args = {
        "site_id": int(site_id),
        **requested_historical_range(message),
    }
    if context.get("building_id"):
        args["building_id"] = int(context["building_id"])
    if operation_id == "telemetry_top_consumers":
        args["limit"] = int(context.get("limit") or 5)
    elif operation_id == "alarm_frequency_summary":
        args["limit"] = int(context.get("limit") or 10)
    elif operation_id == "site_energy_summary":
        args["bucket"] = str(context.get("bucket") or "day")
    return args


def request_daxview_data_plan(turn_id: str, operation_id: str, arguments: dict, request_id: str) -> dict:
    if not DAXVIEW_CALLBACK_BASE_URL or not DAXVIEW_CALLBACK_KEY:
        raise RuntimeError("DAXVIEW_CALLBACK_BASE_URL and DAXVIEW_CALLBACK_KEY are required for historical data")
    payload = {
        "turn_id": turn_id,
        "continuation_id": str(uuid.uuid4()),
        "operation_id": operation_id,
        "arguments": arguments,
    }
    request = Request(
        f"{DAXVIEW_CALLBACK_BASE_URL}/api/ai/integration/data-request-plans",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "User-Agent": "DaxView-AI-Server/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DAXVIEW_CALLBACK_KEY}",
        },
        method="POST",
    )
    safe_arguments = {
        key: value
        for key, value in arguments.items()
        if key in {"site_id", "building_id", "start", "end", "timezone", "bucket", "limit"}
    }
    log_event(
        "daxview_data_plan_request",
        request_id=request_id,
        turn_id=turn_id,
        operation_id=operation_id,
        arguments=safe_arguments,
    )
    try:
        with urlopen(request, timeout=DAXVIEW_MCP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        safe_body = body
        try:
            parsed = json.loads(body)
            safe_body = {
                key: parsed.get(key)
                for key in ("detail", "error", "error_code", "code", "message", "retryable", "request_id")
                if key in parsed
            }
        except json.JSONDecodeError:
            pass
        log_event(
            "daxview_data_plan_error",
            request_id=request_id,
            turn_id=turn_id,
            operation_id=operation_id,
            status=error.code,
            response=safe_body,
        )
        raise


def call_authorized_historical_tool(operation_id: str, authorization_id: str, arguments: dict, request_id: str) -> dict:
    if operation_id not in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        raise ValueError("historical operation is not allowlisted")
    return call_daxview_mcp_tool(operation_id, {"authorization_id": authorization_id, **arguments}, request_id)


def first_dict_with_list(value, list_keys: tuple[str, ...]) -> dict | None:
    if not isinstance(value, dict):
        return None
    if any(isinstance(value.get(key), list) for key in list_keys):
        return value
    for key in ("data", "backend_response", "result", "results", "payload"):
        nested = value.get(key)
        if isinstance(nested, dict):
            found = first_dict_with_list(nested, list_keys)
            if found:
                return found
    return None


def historical_result_data(mcp_result: dict, list_keys: tuple[str, ...] = ("rows", "items", "results", "data")) -> dict:
    structured = mcp_structured_result(mcp_result)
    data = structured.get("data")
    if isinstance(data, dict) and (data or any(isinstance(data.get(key), list) for key in list_keys)):
        return data
    found = first_dict_with_list(structured, list_keys)
    return found if found else {}


def format_number(value, precision: int = 2) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, str):
        return value
    if value is None:
        return "not available"
    return str(value)


def requested_energy_unit(message: str, source_unit: str) -> tuple[float, str, int]:
    lowered = message.lower()
    if "mwh" in lowered or "megawatt-hour" in lowered or "megawatt hour" in lowered or "mkh" in lowered:
        return 1000.0, "MWh", 2
    if "million" in lowered:
        return 1_000_000.0, f"million {source_unit}", 4
    return 1.0, source_unit, 2


def format_energy_value(value, factor: float, unit: str, precision: int) -> str:
    if not isinstance(value, (int, float)):
        return f"{format_number(value, precision)} {unit}"
    return f"{format_number(value / factor, precision)} {unit}"


def format_coverage_note(data: dict) -> str | None:
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        return None
    candidate_devices = coverage.get("candidate_devices")
    devices_with_data = coverage.get("devices_with_data")
    if candidate_devices is None and devices_with_data is None:
        return None
    return (
        "Coverage note: "
        f"{candidate_devices if candidate_devices is not None else 'unknown'} candidate device(s), "
        f"{devices_with_data if devices_with_data is not None else 'unknown'} device(s) with historical data."
    )


def build_compliance_context(message: str = "", operation_ids: list[str] | None = None) -> list[dict]:
    lowered = message.lower()
    operation_ids = operation_ids or []
    wants_energy = (
        "site_energy_summary" in operation_ids
        or any(term in lowered for term in ("kwh", "energy", "consumption", "usage", "enpi", "baseline"))
    )
    wants_power_quality = any(
        term in lowered
        for term in ("sag", "dip", "swell", "transient", "harmonic", "thd", "flicker", "power quality")
    )
    wants_protocol = any(term in lowered for term in ("janitza", "umg", "protocol", "modbus", "bacnet", "snmp"))
    context = []
    for item in ENERGY_COMPLIANCE_CONTEXT:
        standard = item["standard"].lower()
        if "power quality" in item["label"].lower() and not wants_power_quality:
            continue
        if "data protocol" in item["label"].lower() and not (wants_protocol or wants_energy):
            continue
        if wants_energy or ("iso" in standard and ("iso" in lowered or "enpi" in lowered or "baseline" in lowered)):
            context.append(item)
        elif wants_protocol and "janitza" in standard:
            context.append(item)
        elif wants_power_quality and ("iec 61000" in standard or "ieee" in standard):
            context.append(item)
    return context


def format_compliance_context(context: list[dict]) -> str:
    if not context:
        return ""
    lines = ["Applicable protocol / standards context:"]
    for item in context:
        lines.append(f"- {item['label']}: {item['standard']} - {item['note']}")
    return "\n".join(lines)


def parse_datetime(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def format_time_window(data: dict, arguments: dict | None = None) -> str:
    arguments = arguments or {}
    start = data.get("start") or data.get("from") or data.get("start_time") or arguments.get("start")
    end = data.get("end") or data.get("to") or data.get("end_time") or arguments.get("end")
    days = data.get("days") or data.get("day_count") or arguments.get("days")
    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    if days is None and start_dt and end_dt:
        duration_days = round((end_dt - start_dt).total_seconds() / 86400)
        if duration_days > 0:
            days = duration_days
    if days:
        day_count = format_number(days, 0)
        suffix = "day" if day_count == "1" else "days"
        return f"last {day_count} {suffix}"
    if start_dt and end_dt:
        return f"{start_dt.date().isoformat()} to {end_dt.date().isoformat()}"
    if start and end:
        return f"{start} to {end}"
    return "selected time range"


def first_list(data: dict, keys: tuple[str, ...]) -> list:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def first_value(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if row.get(key) is not None:
            return row.get(key)
    return None


def reading_detail(row: dict, value_keys: tuple[str, ...], time_keys: tuple[str, ...], unit: str, precision: int) -> str | None:
    value = first_value(row, value_keys)
    if value is None:
        return None
    detail = format_number(value, precision)
    row_unit = row.get("reading_unit") or row.get("unit") or unit
    timestamp = first_value(row, time_keys)
    if timestamp:
        return f"{detail} {row_unit} at {timestamp}"
    return f"{detail} {row_unit}"


def summarize_top_consumers(data: dict, arguments: dict | None = None) -> str:
    rows = first_list(data, ("rows", "items", "results", "top_consumers", "consumers", "devices"))
    unit = data.get("unit") or "kWh"
    time_window = format_time_window(data, arguments)
    lines = [
        f"These are the top energy-consuming devices for this site over the {time_window}, ranked by total consumption:",
        "",
    ]
    if not rows:
        lines.append("No consuming devices were returned for this site and time range.")
    for index, row in enumerate(rows[:10], 1):
        if not isinstance(row, dict):
            continue
        name = (
            row.get("device_name")
            or row.get("name")
            or row.get("label")
            or f"Device {row.get('device_id', 'unknown')}"
        )
        precision = int(row.get("precision") if isinstance(row.get("precision"), int) else 2)
        value = format_number(
            first_value(
                row,
                (
                    "value",
                    "kwh",
                    "total_kwh",
                    "consumption",
                    "energy",
                    "consumption_delta",
                    "stored_consumption_delta_sum",
                ),
            ),
            precision,
        )
        row_unit = row.get("unit") or unit
        highest = reading_detail(
            row,
            ("highest_reading", "highest", "max_reading", "max", "peak_reading", "peak", "maximum"),
            ("highest_time", "highest_at", "max_time", "max_at", "peak_time", "peak_at"),
            row_unit,
            precision,
        )
        lowest = reading_detail(
            row,
            ("lowest_reading", "lowest", "min_reading", "min", "minimum"),
            ("lowest_time", "lowest_at", "min_time", "min_at"),
            row_unit,
            precision,
        )
        lines.append(f"{index}. {name}")
        lines.append(f"   Total consumption: {value} {row_unit}")
        if highest:
            lines.append(f"   Highest reading: {highest}")
        if lowest:
            lines.append(f"   Lowest reading: {lowest}")
        lines.append("")
    if rows:
        shown = min(len(rows), 10)
        lines.append(f"Showing {shown} device(s). Ranking is based on total consumption over the {time_window}.")
    return "\n".join(lines)


MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def local_bucket_date(timestamp: str, timezone_name: str) -> str | None:
    parsed = parse_datetime(timestamp)
    if not parsed:
        return None
    if timezone_name == "Asia/Kuala_Lumpur":
        parsed = parsed.astimezone(timezone.utc) + timedelta(hours=8)
    return parsed.date().isoformat()


def requested_comparison_dates(message: str, fallback_year: int) -> list[str]:
    lowered = message.lower()
    dates = []
    for match in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"(?:\s+(\d{4}))?\b",
        lowered,
    ):
        day = int(match.group(1))
        month = MONTH_NAMES[match.group(2)]
        year = int(match.group(3) or fallback_year)
        try:
            dates.append(datetime(year, month, day, tzinfo=timezone.utc).date().isoformat())
        except ValueError:
            continue
    return dates


def month_label(value: date) -> str:
    return value.strftime("%B %Y")


def requested_comparison_months(message: str, fallback_year: int) -> list[tuple[str, date, date]]:
    return [(month_label(month_start), month_start, month_end) for month_start, month_end in requested_month_ranges(message, fallback_year)]


def summarize_site_energy(data: dict, message: str = "", arguments: dict | None = None) -> str:
    display = data.get("display") if isinstance(data.get("display"), dict) else {}
    unit = data.get("unit") or display.get("unit") or "kWh"
    precision = int(display.get("precision") if isinstance(display.get("precision"), int) else 2)
    conversion_factor, display_unit, display_precision = requested_energy_unit(message, unit)
    precision = display_precision if conversion_factor != 1.0 else precision
    range_info = data.get("range") if isinstance(data.get("range"), dict) else {}
    timezone_name = range_info.get("timezone") or (arguments or {}).get("timezone") or "Asia/Kuala_Lumpur"
    fallback_year = datetime.now(timezone.utc).year
    requested_dates = requested_comparison_dates(message, fallback_year)
    requested_months = [] if requested_dates else requested_comparison_months(message, fallback_year)
    requested_date_set = set(requested_dates)
    has_specific_dates = bool(requested_date_set)
    has_specific_months = bool(requested_months)
    include_range_summary = wants_relative_summary(message)
    lines = ["Site energy summary returned by Daxview MCP:"]
    summary_keys = ("total", "total_value", "value", "total_kwh", "consumption", "energy")
    total_value = next((data.get(key) for key in summary_keys if data.get(key) is not None), None)
    if total_value is not None and not has_specific_dates and not has_specific_months:
        lines.append(f"Total: {format_energy_value(total_value, conversion_factor, display_unit, precision)}.")
    buckets = data.get("buckets") or data.get("rows") or data.get("series") or data.get("data")
    daily_values = {}
    if isinstance(buckets, list) and buckets:
        if has_specific_months:
            lines.append("Requested monthly values:")
        else:
            lines.append("Daily values:" if include_range_summary else "Requested daily values:" if has_specific_dates else "Daily values:")
        for index, item in enumerate(buckets):
            if not isinstance(item, dict):
                continue
            timestamp = item.get("timestamp") or item.get("bucket") or item.get("start") or item.get("date")
            label = local_bucket_date(timestamp, timezone_name) if isinstance(timestamp, str) else None
            label = label or timestamp or "period"
            value = item.get("value") if item.get("value") is not None else item.get("kwh")
            if isinstance(value, (int, float)):
                daily_values[str(label)] = float(value)
            if has_specific_months:
                continue
            if has_specific_dates and not include_range_summary and str(label) not in requested_date_set:
                continue
            if index < 10:
                row_unit = item.get("unit") or unit
                row_factor, row_display_unit, row_precision = requested_energy_unit(message, row_unit)
                lines.append(f"- {label}: {format_energy_value(value, row_factor, row_display_unit, row_precision if row_factor != 1.0 else precision)}")
    elif total_value is None:
        lines.append("No energy values were returned for this site and time range.")
    wants_difference = any(phrase in message.lower() for phrase in ("difference", "compare", "comparison", "between"))
    if wants_difference and len(daily_values) >= 2:
        dates = requested_dates
        if len(dates) >= 2:
            earlier_date, later_date = sorted(dates[:2])
            earlier_value = daily_values.get(earlier_date)
            later_value = daily_values.get(later_date)
            if earlier_value is not None and later_value is not None:
                difference = later_value - earlier_value
                direction = "higher" if difference > 0 else "lower" if difference < 0 else "the same"
                lines.append(
                    f"Difference: {later_date} was {format_energy_value(abs(difference), conversion_factor, display_unit, precision)} "
                    f"{direction} than {earlier_date} "
                    f"({format_energy_value(later_value, conversion_factor, display_unit, precision)} - "
                    f"{format_energy_value(earlier_value, conversion_factor, display_unit, precision)})."
                )
            else:
                missing = [date for date, value in ((earlier_date, earlier_value), (later_date, later_value)) if value is None]
                lines.append(f"Could not calculate the requested difference because no daily value was returned for {', '.join(missing)}.")
    if has_specific_months and daily_values:
        monthly_totals = {}
        for label, value in daily_values.items():
            try:
                bucket_date = datetime.fromisoformat(label).date()
            except ValueError:
                continue
            for month_name, month_start, month_end in requested_months:
                if month_start <= bucket_date < month_end:
                    monthly_totals[month_name] = monthly_totals.get(month_name, 0.0) + value
        for month_name, _, _ in requested_months:
            if month_name in monthly_totals:
                lines.append(f"- {month_name}: {format_energy_value(monthly_totals[month_name], conversion_factor, display_unit, precision)}")
        if "difference" in message.lower() or "compare" in message.lower() or "between" in message.lower():
            available = [(month_name, monthly_totals.get(month_name)) for month_name, _, _ in requested_months]
            available = [(month_name, value) for month_name, value in available if value is not None]
            if len(available) >= 2:
                earlier_name, earlier_value = available[0]
                later_name, later_value = available[1]
                difference = later_value - earlier_value
                direction = "higher" if difference > 0 else "lower" if difference < 0 else "the same"
                percent = (difference / earlier_value * 100) if earlier_value else None
                detail = (
                    f"Difference: {later_name} was {format_energy_value(abs(difference), conversion_factor, display_unit, precision)} "
                    f"{direction} than {earlier_name}"
                )
                if percent is not None:
                    detail += f" ({format_number(abs(percent), 2)}%)."
                else:
                    detail += "."
                lines.append(detail)
            else:
                lines.append("Could not calculate the requested monthly difference because one of the month totals was not returned.")
    coverage_note = format_coverage_note(data)
    if coverage_note:
        lines.append(coverage_note)
    compliance_note = format_compliance_context(build_compliance_context(message, ["site_energy_summary"]))
    if compliance_note:
        lines.append("")
        lines.append(compliance_note)
    return "\n".join(lines)


def summarize_alarm_frequency(data: dict) -> str:
    rows = data.get("rows") or data.get("alarms") or data.get("items")
    rows = rows if isinstance(rows, list) else []
    lines = ["Alarm frequency summary returned by Daxview MCP:"]
    if not rows:
        lines.append("No alarm-frequency rows were returned for this site and time range.")
    for index, row in enumerate(rows[:10], 1):
        if not isinstance(row, dict):
            continue
        name = row.get("alarm_name") or row.get("name") or row.get("type") or row.get("severity") or "Alarm"
        count = row.get("count") or row.get("frequency") or row.get("total") or row.get("value")
        lines.append(f"{index}. {name} - {format_number(count, 0)} occurrence(s)")
    if data.get("row_count") is not None:
        lines.append(f"Rows returned: {data.get('row_count')}.")
    return "\n".join(lines)


def summarize_historical_answer(
    message: str,
    operation_id: str,
    mcp_result: dict,
    request_id: str,
    arguments: dict | None = None,
) -> str:
    data = historical_result_data(mcp_result)
    if operation_id == "telemetry_top_consumers":
        return summarize_top_consumers(data, arguments)
    if operation_id == "site_energy_summary":
        return summarize_site_energy(data, message, arguments)
    if operation_id == "alarm_frequency_summary":
        return summarize_alarm_frequency(data)
    context = {
        "enabled": True,
        "tools": [operation_id],
        "results": [{"tool": operation_id, "result": mcp_result}],
        "errors": [],
    }
    return ask_ollama(message, retrieve_context(message), request_id, context)


def summarize_historical_answers(
    message: str,
    results: list[dict],
    request_id: str,
) -> str:
    if not results:
        return ask_ollama(message, retrieve_context(message), request_id)
    if len(results) == 1:
        item = results[0]
        return summarize_historical_answer(
            message,
            item["operation_id"],
            item["result"],
            request_id,
            item.get("arguments"),
        )

    sections = []
    for item in results:
        operation_id = item["operation_id"]
        title = {
            "telemetry_top_consumers": "Top energy-consuming devices",
            "site_energy_summary": "Site energy summary",
            "alarm_frequency_summary": "Alarm frequency summary",
        }.get(operation_id, operation_id)
        summary = summarize_historical_answer(
            message,
            operation_id,
            item["result"],
            request_id,
            item.get("arguments"),
        )
        sections.append(f"{title}\n{summary}")
    return "\n\n".join(sections)


FOLLOW_UP_PHRASES = (
    "more detail",
    "more details",
    "explain more",
    "why",
    "what do you mean",
    "regarding your explanation",
    "regarding ur explanation",
    "continue",
    "convert",
    "change the unit",
    "bigger unit",
    "larger unit",
    "breakdown",
    "this calculation",
    "breakdown of this calculation",
    "calculation breakdown",
    "how did you calculate",
    "how you calculated",
    "show calculation",
    "show me calculation",
    "more explanation",
)


def is_follow_up_message(message: str) -> bool:
    lowered = message.lower().strip()
    return any(phrase in lowered for phrase in FOLLOW_UP_PHRASES)


def previous_daxview_turn(conversation_id: str, current_turn_id: str) -> dict | None:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_message, context
                FROM daxview_turns
                WHERE conversation_id = %s AND id <> %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id, current_turn_id),
            )
            row = cur.fetchone()
    return row if row else None


def resolve_follow_up_message(turn_id: str, message: str, context: dict, conversation_id: str, request_id: str) -> tuple[str, dict]:
    if not is_follow_up_message(message):
        return message, context
    previous = previous_daxview_turn(conversation_id, turn_id)
    if not previous or not previous.get("user_message"):
        return message, context
    previous_message = str(previous["user_message"])
    previous_context = previous.get("context") if isinstance(previous.get("context"), dict) else {}
    merged_context = {**previous_context, **context}
    resolved = (
        "Provide a more detailed EMS breakdown and explanation for this previous request: "
        f"{previous_message}. "
        f"Follow-up request: {message}"
    )
    log_event(
        "daxview_follow_up_resolved",
        request_id=request_id,
        turn_id=turn_id,
        previous_turn_id=str(previous.get("id")),
    )
    return resolved, merged_context


def run_daxview_integration_turn(turn_id: str, message: str, context: dict, request_id: str, session_id: str) -> dict:
    message, context = resolve_follow_up_message(turn_id, message, context, session_id, request_id)
    operation_ids = select_historical_operations(message)
    if not operation_ids:
        return langchain_chat_response(message, request_id, session_id)
    results = []
    for operation_id in operation_ids:
        try:
            arguments = build_historical_arguments(operation_id, context, message)
        except ValueError:
            return {
                "provider": "daxview-question-filter",
                "reply": "Choose a site and time range before I access Daxview historical data.",
            }
        plan = request_daxview_data_plan(turn_id, operation_id, arguments, request_id)
        authorization_id = plan.get("authorization_id")
        normalized_arguments = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else arguments
        if not authorization_id:
            raise RuntimeError(f"Daxview did not return a data authorization for {operation_id}")
        result = call_authorized_historical_tool(operation_id, str(authorization_id), normalized_arguments, request_id)
        results.append(
            {
                "operation_id": operation_id,
                "arguments": normalized_arguments,
                "result": result,
            }
        )
    return {
        "provider": "daxview-historical-mcp",
        "reply": summarize_historical_answers(message, results, request_id),
    }


def process_daxview_turn(job_id: str, turn_id: str, message: str, context: dict, request_id: str, session_id: str) -> None:
    try:
        update_job_status(job_id, "running")
        add_job_event(job_id, "status", {"text": "Preparing response"})
        if is_job_cancelled(job_id):
            return
        result = run_daxview_integration_turn(turn_id, message, context, request_id, session_id)
        if is_job_cancelled(job_id):
            return
        if result.get("provider") == "daxview-question-filter":
            add_job_event(
                job_id,
                "waiting_for_user",
                {"prompt": result["reply"], "fields": ["site_id", "building_id", "time_range"]},
            )
            update_job_status(job_id, "completed")
            add_job_event(job_id, "completed", {"status": "completed"})
            return
        add_job_event(job_id, "message", {"text": result["reply"]})
        update_job_status(job_id, "completed")
        add_job_event(job_id, "completed", {"status": "completed"})
    except Exception as error:
        log_event("daxview_job_failed", request_id=request_id, job_id=job_id, error=str(error))
        update_job_status(job_id, "failed")
        add_job_event(job_id, "failed", {"status": "failed", "text": "AI response generation failed."})


def is_ems_related(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in EMS_KEYWORDS)


def has_daxview_scope(message: str) -> bool:
    lowered = message.lower()
    if any(phrase in lowered for phrase in DAXVIEW_GLOBAL_SCOPE_PHRASES):
        return True
    return any(re.search(pattern, lowered) for pattern in DAXVIEW_TARGET_SCOPE_PATTERNS)


def needs_device_choice(message: str) -> bool:
    lowered = message.lower()
    if "device" not in lowered:
        return False
    if any(phrase in lowered for phrase in ("all devices", "every device", "top devices")):
        return False
    return not any(re.search(pattern, lowered) for pattern in (r"\bdevice\s*[:#-]\s*[\w.-]+", r"\bumg[\s-]?\d+\b"))


def daxview_clarification_needed(message: str, tools: list[str]) -> bool:
    if not tools:
        return False
    if not any(tool in DAXVIEW_SCOPE_REQUIRED_TOOLS for tool in tools):
        return False
    lowered = message.lower()
    if any(phrase in lowered for phrase in ("how many", "count", "list", "show all", "all daxview")):
        return False
    if needs_device_choice(message):
        return True
    return not has_daxview_scope(message)


def build_daxview_clarification(message: str, tools: list[str]) -> str:
    if needs_device_choice(message):
        return "Which device should I use for this energy consumption request? Choose a specific device, or say all devices if you want the whole site."
    if "telemetry_top_consumers" in tools:
        return "Choose a Daxview site and time range before I check top energy consumers."
    if "site_energy_summary" in tools:
        return "Choose a Daxview site and time range before I summarize site energy."
    if "alarm_frequency_summary" in tools:
        return "Choose a Daxview site and time range before I summarize alarm frequency."
    return "Choose a Daxview site and time range before I access historical data."


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
        "User-Agent": "DaxView-AI-Server/1.0",
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
    if DAXVIEW_MCP_DEBUG_RESPONSE:
        log_event(
            "mcp_tool_request_debug",
            request_id=request_id,
            tool=tool_name,
            arguments=redact_debug_value(arguments or {}),
        )
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
    if DAXVIEW_MCP_DEBUG_RESPONSE:
        log_event(
            "mcp_tool_response_debug",
            request_id=request_id,
            tool=tool_name,
            shape=mcp_result_shape(result),
            sample=debug_json_sample(result, DAXVIEW_MCP_DEBUG_RESPONSE_LIMIT),
        )
    return result


def select_daxview_tools(message: str) -> list[str]:
    lowered = message.lower()
    selected = []
    for tool_name, phrases in DAXVIEW_TOOL_KEYWORDS.items():
        if any(phrase in lowered for phrase in phrases):
            selected.append(tool_name)
    return selected[:3]


def retrieve_daxview_context(message: str, request_id: str) -> dict:
    tools = select_daxview_tools(message)
    if not DAXVIEW_MCP_ENABLED or not DAXVIEW_MCP_URL or not tools:
        return {"enabled": DAXVIEW_MCP_ENABLED, "tools": [], "results": [], "errors": []}
    if any(tool in DAXVIEW_ALLOWED_HISTORICAL_TOOLS for tool in tools):
        return {
            "enabled": True,
            "tools": tools,
            "results": [],
            "errors": [
                {
                    "tool": tool,
                    "error": "Historical Daxview MCP tools require data-plan authorization through the integration turn API.",
                }
                for tool in tools
            ],
        }

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
    alarm_count = extract_alarm_count(structured) if tool_name == "alarm_frequency_summary" else None
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
            if item.get("tool") != "alarm_frequency_summary":
                continue
            structured = mcp_structured_result(item.get("result") or {})
            alarm_count = extract_alarm_count(structured)
            if alarm_count is not None:
                return f"The Daxview alarm-frequency summary returned {alarm_count} alert records."
        return "I could not determine the Daxview alarm count from the MCP response."
    return None


def format_daxview_context(mcp_context: dict) -> str:
    results = mcp_context.get("results") or []
    errors = mcp_context.get("errors") or []
    if not results and not errors:
        return "No historical Daxview MCP data was requested or available for this question."
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
        return "skipped", "No historical Daxview MCP tool was needed for this question."
    if results and errors:
        return "partial", f"Called {len(results)} Daxview tool(s); {len(errors)} tool(s) failed."
    if results:
        return "completed", f"Called {len(results)} Daxview tool(s)."
    return "failed", f"Tried {len(tools)} Daxview tool(s); {len(errors)} failed."


def embed_text(text: str) -> list[float]:
    data = ollama_json("/api/embeddings", {"model": EMBEDDING_MODEL, "prompt": text}, timeout=OLLAMA_EMBEDDING_TIMEOUT)
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
Use the EMS library context when relevant. If no matching EMS library context is found, still answer the EMS question using general domain knowledge, and clearly state when site-specific proof, device configuration, calibration, or source evidence is missing.
Use historical Daxview data when it is provided. If a Daxview tool failed, say historical Daxview data is currently unavailable for that part.
For ranked historical answers, number the result from 1 to last, include the time window, and omit internal fields such as source_count, raw row_count, aggregation names, backend endpoints, and last_updated unless the user explicitly asks for diagnostics.
When explaining kWh or consumption compliance, distinguish data protocols from standards: protocols such as Modbus, BACnet, and SNMP describe data transport; ISO 50001/50006 describe energy-management baselines and EnPIs; IEC meter standards and device active-energy class describe measurement accuracy. Do not claim a reading is certified or in accordance with a standard unless source data proves that certification, calibration, and device configuration.
For Janitza UMG device, voltage sag, power quality, alarm, THD, or meter troubleshooting questions, prioritize likely root causes, what readings to check, and practical EMS investigation steps.
If the user says "main cost" in a voltage sag or fault context, treat it as possibly meaning "main cause" and clarify both cause and cost impact briefly.
Do not answer unrelated general questions.

Historical Daxview MCP data:
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
            timeout=OLLAMA_GENERATE_TIMEOUT,
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
If no matching EMS library context is found, still answer from general EMS domain knowledge and state what cannot be confirmed from site/device evidence.
Use historical Daxview MCP data when provided. If historical data conflicts with assumptions, historical data wins.
For historical read-only questions such as top consumers, site energy summary, or alarm frequency summary, report the MCP facts directly. Do not reject approved historical data requests as unsafe.
For ranked historical answers, number from 1 to last, include the time window, and skip internal metadata like source_count, raw row_count, aggregation, backend endpoint, and last_updated unless asked.
When discussing kWh compliance, say protocols such as Modbus, BACnet, and SNMP are data transport/integration protocols, while ISO 50001/50006 are EnMS baseline/EnPI frameworks and IEC meter standards/device class are measurement-accuracy context. Do not overclaim certification without explicit evidence.
If this is about voltage sag, list likely causes first, then readings/checks.
Use DIRECT REJECTION only when the user asks for a physical action that would exceed rated limits, bypass alarms, increase unsafe load, or override protection.

Historical Daxview MCP data:
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
            timeout=OLLAMA_GENERATE_TIMEOUT,
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
Use historical Daxview MCP data when it is provided. If a Daxview tool failed, clearly say historical Daxview data is unavailable before giving a general EMS answer.
For approved historical read-only questions such as top consumers, site energy summary, or alarm frequency summary, answer directly from the Historical Daxview MCP data. Do not invent hazards or recommend Load Shedding unless the MCP data explicitly reports an unsafe operating condition or the user asks for an unsafe physical action.
For ranked historical answers, number from 1 to last, include the time window, and omit internal metadata like source_count, raw row_count, aggregation, backend endpoint, and last_updated unless the user asks for diagnostics.
Safety hard limits override energy saving, user preference, uptime, cost, and comfort.
If any rated hardware limit is explicitly exceeded or the user asks to add load/bypass an alarm, reject immediately using exactly these sections:
DIRECT REJECTION:
PHYSICAL REASONING:
MITIGATION ACTION:
Keep the final answer simple, compact, precise, and maximum 5 short bullets.

Question:
{message}

Historical Daxview MCP data:
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
            timeout=OLLAMA_GENERATE_TIMEOUT,
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
    agent_specs = [
        (
            "Agent 1 - Ollama EMS Triage",
            "Quickly classify the issue and identify immediate EMS checks.",
            OLLAMA_MODEL,
        ),
        (
            "Agent 2 - Qwen Power Quality",
            "Find likely electrical root causes and UMG meter readings to inspect.",
            OLLAMA_MODEL,
        ),
    ]

    def run_agent(spec: tuple[str, str, str]) -> dict:
        agent_name, role, model = spec
        return {
            "agent": agent_name,
            "role": role,
            "model": model,
            "answer": ask_role_agent(agent_name, role, model, message, contexts, request_id, mcp_context),
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        agent_answers = list(executor.map(run_agent, agent_specs))
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
    selected_tools = state.get("selected_tools") or []
    return {
        "reply": state["reply"],
        "provider": state.get("provider"),
        "model": state.get("model"),
        "is_ems_related": state.get("is_ems_related"),
        "sources": state.get("sources") or [],
        "compliance_context": build_compliance_context(message, selected_tools),
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
    selected_tools = state.get("selected_tools") or []
    return {
        "reply": state["reply"],
        "provider": state.get("provider"),
        "model": state.get("model"),
        "decision_model": state.get("decision_model") or state.get("model"),
        "is_ems_related": state.get("is_ems_related"),
        "sources": state.get("sources") or [],
        "compliance_context": build_compliance_context(message, selected_tools),
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-DaxView-Assertion")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        return json.loads(self.rfile.read(content_length))

    def _daxview_auth(self) -> dict | None:
        try:
            return daxview_identity(self.headers)
        except PermissionError as error:
            self._send_json(401, json_error("unauthorized", str(error), False))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(401, json_error("invalid_assertion", str(error), False))
        return None

    def _send_db_error(self, error: Exception) -> None:
        status = 503 if isinstance(error, RuntimeError) else 500
        self._send_json(status, json_error("ai_store_unavailable", str(error), True))

    def handle_daxview_get(self, path: str, query: dict[str, list[str]]) -> bool:
        if not path.startswith("/v1/integrations/daxview/"):
            return False
        if path == "/v1/integrations/daxview/health":
            identity = self._daxview_auth()
            if not identity:
                return True
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "daxview-ai",
                    "schema_version": "1.0",
                    "conversation_store": bool(DATABASE_URL),
                    "job_queue": True,
                    "mcp_client": DAXVIEW_MCP_ENABLED,
                    "supported_manifest_versions": ["2026-09-14"],
                },
            )
            return True
        identity = self._daxview_auth()
        if not identity:
            return True
        try:
            require_database()
            if path == "/v1/integrations/daxview/conversations":
                limit = min(max(int((query.get("limit") or ["20"])[0]), 1), 50)
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, title, created_at, updated_at
                            FROM daxview_conversations
                            WHERE deployment_id = %s AND company_id = %s AND user_id = %s AND status = 'active'
                            ORDER BY updated_at DESC
                            LIMIT %s
                            """,
                            (identity["deployment_id"], identity["company_id"], identity["user_id"], limit),
                        )
                        rows = cur.fetchall()
                self._send_json(
                    200,
                    {
                        "items": [
                            {
                                "conversation_id": str(row["id"]),
                                "title": row["title"],
                                "created_at": row["created_at"].isoformat(),
                                "updated_at": row["updated_at"].isoformat(),
                            }
                            for row in rows
                        ],
                        "next_cursor": None,
                    },
                )
                return True
            match = re.fullmatch(r"/v1/integrations/daxview/conversations/([^/]+)/turns", path)
            if match:
                conversation_id = match.group(1)
                limit = min(max(int((query.get("limit") or ["5"])[0]), 1), 20)
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id FROM daxview_conversations
                            WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s AND status = 'active'
                            """,
                            (conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                        if not cur.fetchone():
                            self._send_json(404, json_error("conversation_not_found", "The conversation is unavailable."))
                            return True
                        cur.execute(
                            """
                            SELECT t.id, t.created_at, t.user_message, j.id AS job_id
                            FROM daxview_turns t
                            LEFT JOIN daxview_jobs j ON j.turn_id = t.id
                            WHERE t.conversation_id = %s
                            ORDER BY t.created_at DESC
                            LIMIT %s
                            """,
                            (conversation_id, limit),
                        )
                        turns = cur.fetchall()
                        items = []
                        for turn in reversed(turns):
                            cur.execute(
                                """
                                SELECT event_type, event_data
                                FROM daxview_job_events
                                WHERE job_id = %s AND event_type IN ('message', 'waiting_for_user', 'failed', 'cancelled')
                                ORDER BY event_id ASC
                                """,
                                (turn["job_id"],),
                            )
                            events = cur.fetchall() if turn["job_id"] else []
                            items.append(
                                {
                                    "turn_id": str(turn["id"]),
                                    "created_at": turn["created_at"].isoformat(),
                                    "user_message": {"text": turn["user_message"]},
                                    "assistant_events": [
                                        {
                                            "type": event["event_type"],
                                            "text": (event["event_data"] or {}).get("text") or (event["event_data"] or {}).get("prompt") or "",
                                        }
                                        for event in events
                                    ],
                                }
                            )
                self._send_json(200, {"conversation_id": conversation_id, "items": items, "previous_cursor": None})
                return True
            match = re.fullmatch(r"/v1/integrations/daxview/jobs/([^/]+)/events", path)
            if match:
                job_id = match.group(1)
                after = max(int((query.get("after") or ["0"])[0]), 0)
                limit = min(max(int((query.get("limit") or ["100"])[0]), 1), 100)
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id FROM daxview_jobs
                            WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s
                            """,
                            (job_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                        if not cur.fetchone():
                            self._send_json(404, json_error("job_not_found", "The job is unavailable."))
                            return True
                        cur.execute(
                            """
                            SELECT event_id, event_type, event_data
                            FROM daxview_job_events
                            WHERE job_id = %s AND event_id > %s
                            ORDER BY event_id ASC
                            LIMIT %s
                            """,
                            (job_id, after, limit),
                        )
                        rows = cur.fetchall()
                self._send_json(
                    200,
                    {
                        "events": [
                            {"id": row["event_id"], "type": row["event_type"], "data": row["event_data"]}
                            for row in rows
                        ]
                    },
                )
                return True
        except Exception as error:
            self._send_db_error(error)
            return True
        self._send_json(404, json_error("not_found", "Not found."))
        return True

    def handle_daxview_post(self, path: str, request: dict) -> bool:
        if not path.startswith("/v1/integrations/daxview/"):
            return False
        identity = self._daxview_auth()
        if not identity:
            return True
        try:
            require_database()
            if path == "/v1/integrations/daxview/conversations":
                title = str(request.get("title") or "New conversation").strip()[:120] or "New conversation"
                conversation_id = str(uuid.uuid4())
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO daxview_conversations (id, deployment_id, company_id, user_id, title)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id, title, created_at, updated_at
                            """,
                            (conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"], title),
                        )
                        row = cur.fetchone()
                    conn.commit()
                self._send_json(
                    201,
                    {
                        "conversation_id": str(row["id"]),
                        "title": row["title"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                    },
                )
                return True
            match = re.fullmatch(r"/v1/integrations/daxview/conversations/([^/]+)/turns", path)
            if match:
                conversation_id = match.group(1)
                turn_id = str(request.get("turn_id") or "").strip()
                message = str(request.get("message") or "").strip()
                request_id = str(request.get("request_id") or uuid.uuid4())
                context = request.get("context") if isinstance(request.get("context"), dict) else {}
                if not turn_id:
                    self._send_json(400, json_error("invalid_turn", "turn_id is required.", False, request_id))
                    return True
                if not message:
                    self._send_json(400, json_error("invalid_turn", "message is required.", False, request_id))
                    return True
                job_id = safe_job_id(turn_id)
                created = False
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id FROM daxview_conversations
                            WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s AND status = 'active'
                            """,
                            (conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                        if not cur.fetchone():
                            self._send_json(404, json_error("conversation_not_found", "The conversation is unavailable.", False, request_id))
                            return True
                        cur.execute(
                            """
                            INSERT INTO daxview_turns (id, conversation_id, deployment_id, company_id, user_id, user_message, request_id, context)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                turn_id,
                                conversation_id,
                                identity["deployment_id"],
                                identity["company_id"],
                                identity["user_id"],
                                message,
                                request_id,
                                json.dumps(context),
                            ),
                        )
                        created = cur.rowcount > 0
                        cur.execute(
                            """
                            INSERT INTO daxview_jobs (id, turn_id, conversation_id, deployment_id, company_id, user_id)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (job_id, turn_id, conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                        cur.execute("UPDATE daxview_conversations SET updated_at = now() WHERE id = %s", (conversation_id,))
                    conn.commit()
                if created:
                    add_job_event(job_id, "status", {"text": "Accepted turn"})
                    DAXVIEW_JOB_EXECUTOR.submit(process_daxview_turn, job_id, turn_id, message, context, request_id, conversation_id)
                self._send_json(202, {"job_id": job_id})
                return True
            match = re.fullmatch(r"/v1/integrations/daxview/jobs/([^/]+)/cancel", path)
            if match:
                job_id = match.group(1)
                with db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE daxview_jobs
                            SET status = 'cancelled', cancelled_at = COALESCE(cancelled_at, now()), updated_at = now()
                            WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s
                            RETURNING id
                            """,
                            (job_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                        row = cur.fetchone()
                    conn.commit()
                if not row:
                    self._send_json(404, json_error("job_not_found", "The job is unavailable."))
                    return True
                add_job_event(job_id, "cancelled", {"status": "cancelled"})
                self._send_json(200, {"status": "cancelled"})
                return True
        except Exception as error:
            self._send_db_error(error)
            return True
        self._send_json(404, json_error("not_found", "Not found."))
        return True

    def handle_daxview_patch(self, path: str, request: dict) -> bool:
        if not path.startswith("/v1/integrations/daxview/"):
            return False
        identity = self._daxview_auth()
        if not identity:
            return True
        match = re.fullmatch(r"/v1/integrations/daxview/conversations/([^/]+)", path)
        if not match:
            self._send_json(404, json_error("not_found", "Not found."))
            return True
        title = str(request.get("title") or "").strip()[:120]
        if not title:
            self._send_json(400, json_error("invalid_title", "title is required."))
            return True
        try:
            require_database()
            with db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE daxview_conversations
                        SET title = %s, updated_at = now()
                        WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s AND status = 'active'
                        RETURNING id, title, created_at, updated_at
                        """,
                        (title, match.group(1), identity["deployment_id"], identity["company_id"], identity["user_id"]),
                    )
                    row = cur.fetchone()
                conn.commit()
            if not row:
                self._send_json(404, json_error("conversation_not_found", "The conversation is unavailable."))
                return True
            self._send_json(
                200,
                {
                    "conversation_id": str(row["id"]),
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                },
            )
        except Exception as error:
            self._send_db_error(error)
        return True

    def handle_daxview_delete(self, path: str) -> bool:
        if not path.startswith("/v1/integrations/daxview/"):
            return False
        identity = self._daxview_auth()
        if not identity:
            return True
        try:
            require_database()
            conversation_match = re.fullmatch(r"/v1/integrations/daxview/conversations/([^/]+)", path)
            user_match = re.fullmatch(r"/v1/integrations/daxview/lifecycle/user/([^/]+)", path)
            company_match = re.fullmatch(r"/v1/integrations/daxview/lifecycle/company/([^/]+)", path)
            with db() as conn:
                with conn.cursor() as cur:
                    if conversation_match:
                        cur.execute(
                            """
                            UPDATE daxview_conversations
                            SET status = 'deleted', deleted_at = COALESCE(deleted_at, now()), updated_at = now()
                            WHERE id = %s AND deployment_id = %s AND company_id = %s AND user_id = %s
                            """,
                            (conversation_match.group(1), identity["deployment_id"], identity["company_id"], identity["user_id"]),
                        )
                    elif user_match:
                        cur.execute(
                            """
                            UPDATE daxview_conversations
                            SET status = 'deleted', deleted_at = COALESCE(deleted_at, now()), updated_at = now()
                            WHERE deployment_id = %s AND user_id = %s
                            """,
                            (identity["deployment_id"], user_match.group(1)),
                        )
                    elif company_match:
                        cur.execute(
                            """
                            UPDATE daxview_conversations
                            SET status = 'deleted', deleted_at = COALESCE(deleted_at, now()), updated_at = now()
                            WHERE deployment_id = %s AND company_id = %s
                            """,
                            (identity["deployment_id"], company_match.group(1)),
                        )
                    else:
                        self._send_json(404, json_error("not_found", "Not found."))
                        return True
                conn.commit()
            self._send_json(200, {})
        except Exception as error:
            self._send_db_error(error)
        return True

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if self.handle_daxview_get(path, parse_qs(parsed.query)):
            return
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
            request = self._read_json()
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if self.handle_daxview_post(path, request):
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

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path
        try:
            request = self._read_json()
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, json_error("invalid_json", "Request body must be valid JSON."))
            return
        if self.handle_daxview_patch(path, request):
            return
        self._send_json(404, {"error": "Not found"})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if self.handle_daxview_delete(path):
            return
        self._send_json(404, {"error": "Not found"})

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
