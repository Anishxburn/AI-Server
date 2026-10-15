"""EMS-only chatbot API with Ollama, PostgreSQL, and pgvector RAG."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import base64
from difflib import SequenceMatcher
import hashlib
import hmac
import json
import math
import os
import re
import statistics
import time
import traceback
import uuid
from pathlib import Path
from collections import deque
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from langchain_core.runnables import RunnableLambda
import psycopg
from psycopg.rows import dict_row
from prediction_lab import PredictionService, daily_dataset
from ems_sandbox import formula_reference, prepare_sandbox_context, sample_dataset
from ems_contracts import measurement_context, metric_clarification, percentage_difference, question_metrics, result_problem
from contracts.tool_schemas import TOOL_ARGUMENT_KEYS, TOOL_SCHEMAS, validate_tool_arguments
from contracts.mcp_manifest import PUBLIC_MCP_TOOLS, build_public_calls, normalize_mcp_arguments, public_tool_for_internal
from planner.hybrid_planner import parse_planner_response, planner_json_schema, planner_system_prompt


HOST = os.getenv("CHATBOT_HOST", "127.0.0.1")
PORT = int(os.getenv("CHATBOT_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:1.5b")
CHAT_MODEL = os.getenv("CHAT_MODEL", OLLAMA_MODEL)
AI_COMPARE_MODEL_ENABLED = os.getenv("AI_COMPARE_MODEL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AI_COMPARE_MODEL = os.getenv("AI_COMPARE_MODEL", DEEPSEEK_MODEL).strip()
AI_COMPARE_MODELS = [model.strip() for model in os.getenv("AI_COMPARE_MODELS", "").split(",") if model.strip()]
if not AI_COMPARE_MODELS and AI_COMPARE_MODEL:
    AI_COMPARE_MODELS = [AI_COMPARE_MODEL]
AI_COMPARE_MODEL_SHOW_TO_USER = os.getenv("AI_COMPARE_MODEL_SHOW_TO_USER", "false").lower() in {"1", "true", "yes", "on"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_GENERATE_TIMEOUT = int(os.getenv("OLLAMA_GENERATE_TIMEOUT", "120"))
AI_SANDBOX_MODEL_TIMEOUT = int(os.getenv("AI_SANDBOX_MODEL_TIMEOUT", "180"))
AI_SANDBOX_PLANNER_TIMEOUT = int(os.getenv("AI_SANDBOX_PLANNER_TIMEOUT", "120"))
OLLAMA_EMBEDDING_TIMEOUT = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
RAG_MATCH_LIMIT = int(os.getenv("RAG_MATCH_LIMIT", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
TRACE_LIMIT = int(os.getenv("CHATBOT_TRACE_LIMIT", "25"))
TRACE_PAGE_SIZE = int(os.getenv("CHATBOT_TRACE_PAGE_SIZE", "200"))
AI_DEBUG_DASHBOARD_ENABLED = os.getenv("AI_DEBUG_DASHBOARD_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AI_REFINE_MCP_WITH_MODEL = os.getenv("AI_REFINE_MCP_WITH_MODEL", "true").lower() in {"1", "true", "yes", "on"}
AI_MCP_FALLBACKS_ENABLED = os.getenv("AI_MCP_FALLBACKS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
AI_CHARTS_ENABLED = os.getenv("AI_CHARTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
AI_DISPLAY_TIMEZONE = os.getenv("AI_DISPLAY_TIMEZONE", "Asia/Kuala_Lumpur")
AI_PLANNER_ENABLED = os.getenv("AI_PLANNER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AI_PLANNER_SHADOW_MODE = os.getenv("AI_PLANNER_SHADOW_MODE", "false").lower() in {"1", "true", "yes", "on"}
PLANNER_MODEL = os.getenv("PLANNER_MODEL", OLLAMA_MODEL)
PLANNER_TIMEOUT = int(os.getenv("PLANNER_TIMEOUT", "15"))
DEMAND_RANKING_MAX_DEVICES = int(os.getenv("DEMAND_RANKING_MAX_DEVICES", "50"))
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"
PREDICTION_SERVICE = PredictionService(
    os.getenv("AI_PREDICTION_DATA_DIR", str(Path(__file__).resolve().parent / "data")),
    os.getenv("AI_PREDICTION_SOURCE_URL", "").strip(),
    os.getenv("AI_PREDICTION_SOURCE_TOKEN", "").strip(),
    json.loads(os.getenv("AI_PREDICTION_SITES_JSON", "[]")),
)
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
DAXVIEW_ALLOWED_HISTORICAL_TOOLS = {"device_data", "alarm_data", "site_summary"}
DAXVIEW_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=AI_JOB_WORKERS)
SANDBOX_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ems-sandbox")
SANDBOX_JOBS: dict[str, dict] = {}
SANDBOX_JOBS_LOCK = Lock()
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
    "daxview", "site summary", "summary for this site", "summary of this site",
    "this site", "device", "devices", "device id", "device summary", "inventory", "open alarms",
}

DAXVIEW_TOOL_KEYWORDS = {
    "device_energy_breakdown": {
        "top consumer", "top consumers", "most energy", "highest usage",
        "highest consumption", "largest load", "biggest consumer",
        "top consuming", "energy-consuming", "energy consuming", "top devices",
    },
    "site_energy_summary": {
        "energy summary", "site energy", "usage trend", "consumption trend",
        "kwh summary", "weekly energy", "daily energy",
        "energy consumption", "consumption", "usage", "difference in energy",
        "energy difference", "highest energy consumption",
    },
    "alarm_frequency_summary": {
        "alarm frequency", "frequent alarm", "most alarms", "alarm summary",
        "alarm history", "repeated alarms", "historical alarms",
        "alarm types", "alarms occurred most often", "most common alarms",
        "which alarms occurred", "top alarms", "latest alarm", "latest alarms",
        "most recent alarm", "most recent alarms",
    },
    "site_metadata_summary": {
        "what site", "current site", "site details", "details about this site",
        "site metadata", "site context", "buildings under this site",
        "site info", "site information", "site summary", "summary details of this site",
        "summary for this site", "summary of this site", "this site summary",
        "all details", "other info",
    },
    "site_device_list": {
        "list all devices", "devices under this site", "all meters", "installed meters",
        "device list", "meters in this building", "devices have data",
        "list devices", "list me out devices", "list me out all the device",
        "list me out all the devices", "list device id", "list all device id",
        "which device", "which one is best", "best to use for voltage",
        "best to use for current", "voltage/current trend testing",
    },
    "telemetry_timeseries": {
        "trend", "timeseries", "time series", "chart", "plot", "hourly",
        "voltage trend", "power factor", "thd trend", "kwh trend",
    },
    "telemetry_metric_catalog": {
        "available metrics", "available metric", "supported metrics", "supported metric",
        "metric catalog", "telemetry catalog", "what data can i get",
        "what metrics", "which metrics", "which devices support", "supports voltage",
        "supports current", "can i get voltage", "can i get current",
    },
    "latest_telemetry_snapshot": {
        "latest reading", "latest readings", "latest telemetry", "current reading",
        "current value", "latest value", "now reading", "snapshot",
    },
    "active_alarm_summary": {
        "active alarm", "active alarms", "current alarm", "current alarms",
        "open alarm", "open alarms", "unresolved alarm", "critical alarms",
    },
    "meter_status_summary": {
        "offline meter", "offline meters", "meter status", "device status",
        "stale data", "reporting data", "all devices reporting",
    },
    "energy_comparison_summary": {
        "compare energy", "compare usage", "compare consumption", "this week and last week",
        "august and september", "today vs yesterday", "versus",
    },
    "data_availability_summary": {
        "data coverage", "missing data", "no energy data", "data availability",
        "data complete", "missing energy", "check data availability",
        "availability for voltage", "availability for current",
    },
    "alarm_detail_lookup": {
        "alarm id", "alarm detail", "alarm details", "explain this alarm",
        "latest voltage sag alarm", "what caused the alarm",
    },
    "power_quality_summary": {
        "power quality", "voltage sag", "voltage swell", "thd", "power factor",
        "unbalance", "imbalance",
    },
    "demand_peak_summary": {
        "peak demand", "maximum demand", "max demand", "highest demand",
        "demand limit", "exceed demand", "maximum kw", "max kw",
        "highest kw", "peak kw",
    },
    "tariff_cost_summary": {
        "tariff", "energy cost", "electricity cost", "billing cost",
        "cost summary",
    },
    "device_energy_breakdown": {
        "energy breakdown", "break down", "contribution by device",
        "energy share", "by device", "by building",
    },
    "energy_forecast": {
        "forecast", "predict", "prediction", "expected usage",
        "expected energy", "future energy",
    },
    "anomaly_detection_summary": {
        "anomaly", "abnormal", "unusual", "suspicious", "detect abnormal",
    },
    "report_summary": {
        "report", "management summary", "weekly ems report", "monthly report",
        "ems health summary",
    },
}

DAXVIEW_TOOL_DESCRIPTIONS = {
    "device_data": "Device inventory, configured metric discovery, historical metric values, and per-device energy rows. Select the operation field: list, metrics, values, or energy_rows.",
    "alarm_data": "Active alarms, historical alarm frequency, or one alarm's detail. Select the operation field: active, frequency, or detail.",
    "site_summary": "Site/building overview or site energy buckets. Select the operation field: overview or energy.",
}

DAXVIEW_SCOPE_REQUIRED_TOOLS = {
    *DAXVIEW_ALLOWED_HISTORICAL_TOOLS,
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
    "this site", "selected site", "current site",
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
    record = {"event": event, **safe_fields}
    if AI_DEBUG_DASHBOARD_ENABLED:
        trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **redact_debug_value(record),
        }
        TRACES.appendleft(trace)
        persist_debug_trace(trace)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def debug_trace_event(event: str, **fields: object) -> None:
    if not AI_DEBUG_DASHBOARD_ENABLED:
        return
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **redact_debug_value(fields),
    }
    TRACES.appendleft(trace)
    persist_debug_trace(trace)


def preview(text: object, limit: int = 240) -> str:
    if text is None:
        return ""
    normalized = " ".join(str(text).split())
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
        return [redact_debug_value(item) for item in value[:1000]]
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


def sanitize_persisted_trace(value):
    """Retain diagnostic metadata without storing free text or raw tool bodies."""
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("key", "token", "secret", "assertion", "authorization")):
                safe[key] = "[redacted]"
            elif lowered in {"message", "prompt", "answer", "reply", "error", "traceback", "raw_response", "response_body", "payload_text"}:
                safe[key] = ({"redacted": True, "length": len(item)} if isinstance(item, str) else "[redacted]")
            elif lowered in {"result", "structuredcontent", "backend_response", "mcp_response"}:
                safe[key] = mcp_result_shape(item)
            else:
                safe[key] = sanitize_persisted_trace(item)
        return safe
    if isinstance(value, list):
        return [sanitize_persisted_trace(item) for item in value[:1000]]
    if isinstance(value, str):
        return value[:2000]
    return value


def db() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def persist_debug_trace(trace: dict) -> None:
    """Persist redacted dashboard events while keeping the in-memory hot buffer."""
    if not AI_DEBUG_DASHBOARD_ENABLED or not DATABASE_URL:
        return
    try:
        safe_payload = sanitize_persisted_trace(trace)
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ai_debug_traces (request_id, event, payload, created_at) VALUES (%s, %s, %s, %s)",
                    (str(safe_payload.get("request_id") or safe_payload.get("job_id") or "unknown"),
                     str(safe_payload.get("event") or "trace"), json.dumps(safe_payload),
                     trace.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                )
            conn.commit()
    except Exception as error:
        # Debug persistence must never fail a user request; normal logs still retain the event.
        print(json.dumps({"event": "debug_trace_persist_failed", "error": str(error)}, ensure_ascii=False), flush=True)


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


def ensure_debug_trace_schema() -> None:
    if not DATABASE_URL:
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_debug_traces (
                    id BIGSERIAL PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ai_debug_traces_created_idx ON ai_debug_traces (created_at DESC, id DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS ai_debug_traces_request_idx ON ai_debug_traces (request_id, id DESC)")
            cur.execute("ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS parent_turn_id UUID REFERENCES daxview_turns(id) ON DELETE SET NULL")
            cur.execute("ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS clarification_state JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("CREATE INDEX IF NOT EXISTS daxview_turns_parent_idx ON daxview_turns (parent_turn_id) WHERE parent_turn_id IS NOT NULL")
        conn.commit()


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


def connected_load_calculation_answer(message: str) -> dict | None:
    """Calculate explicit connected loads without routing them as historical telemetry."""
    lowered = message.lower()
    if not re.search(r"\b(?:maximum demand|max demand|peak demand|connected load|connected loads)\b", lowered):
        return None
    if not re.search(r"\b(?:load|loads|lighting|socket|cooker|heater|air conditioner)\b", lowered):
        return None

    voltage_match = re.search(r"\b(?:at\s*)?(\d+(?:\.\d+)?)\s*v\b", lowered)
    voltage = float(voltage_match.group(1)) if voltage_match else None
    segments = [segment.strip(" \t-*•") for segment in re.split(r"[\r\n;]+|\s*[•]\s*", message) if segment.strip()]
    loads = []
    socket_points = None
    socket_current = None

    for segment in segments:
        text = segment.strip()
        lower_segment = text.lower()
        if "socket" in lower_segment and "point" in lower_segment:
            count_match = re.search(r"\b(\d+)\s+(?:(?:single|double)(?:\s*/\s*(?:single|double))?\s+)?(?:socket\s+)?(?:outlet\s+)?points?\b", lower_segment)
            current_match = re.search(r"\b(\d+(?:\.\d+)?)\s*a\b", lower_segment)
            if count_match and current_match:
                socket_points = int(count_match.group(1))
                socket_current = float(current_match.group(1))
                continue

        if "lighting" in lower_segment:
            count_match = re.search(r"\b(\d+)\s+(?:lighting\s+)?points?\b", lower_segment)
            watts_match = re.search(r"\b(\d+(?:\.\d+)?)\s*w\s*(?:each|per\s+point)\b", lower_segment)
            if count_match and watts_match:
                count, watts = int(count_match.group(1)), float(watts_match.group(1))
                loads.append({"name": "Lighting", "calculation": f"{count} x {watts:g} W", "kw": count * watts / 1000})
                continue

        kw_match = re.search(r"\b(\d+(?:\.\d+)?)\s*kw\b", lower_segment)
        w_match = re.search(r"\b(\d+(?:\.\d+)?)\s*w\b", lower_segment)
        if kw_match:
            label = re.split(r"[:(]", text, maxsplit=1)[0].strip(" -*•") or "Load"
            loads.append({"name": label, "calculation": f"{float(kw_match.group(1)):g} kW", "kw": float(kw_match.group(1))})
        elif w_match and not "each" in lower_segment:
            label = re.split(r"[:(]", text, maxsplit=1)[0].strip(" -*•") or "Load"
            loads.append({"name": label, "calculation": f"{float(w_match.group(1)):g} W", "kw": float(w_match.group(1)) / 1000})

    if socket_points and socket_current and voltage:
        socket_kw = socket_points * socket_current * voltage / 1000
        loads.append({
            "name": "Socket outlets",
            "calculation": f"{socket_points} points x {socket_current:g} A x {voltage:g} V",
            "kw": socket_kw,
        })
    if not loads:
        return None

    total_kw = sum(item["kw"] for item in loads)
    lines = ["Calculated from the loads you supplied (no DaxView/MCP data used):"]
    lines.extend(
        f"- {item['name']}: {item['calculation']} = {item['kw']:.2f} kW"
        for item in loads
    )
    lines.append(f"Total connected load: {total_kw:.2f} kW")
    if voltage:
        lines.append(f"At {voltage:g} V single-phase, the equivalent full-load current is {total_kw * 1000 / voltage:.2f} A (assuming power factor 1.0).")
    if socket_points and socket_current:
        lines.append(
            f"Socket assumption: each of the {socket_points} stated points is treated as one {socket_current:g} A outlet, "
            "all drawing full current simultaneously; single/double-point wiring details may change this."
        )
    lines.append(
        "This is the undiversified connected-load total, not a code-diversified maximum-demand estimate. "
        "A defensible diversified estimate needs the applicable local wiring standard and its diversity rules; "
        "socket-point count alone also does not establish actual simultaneous loading."
    )
    return {
        "reply": "\n".join(lines),
        "loads": loads,
        "total_kw": total_kw,
        "voltage_v": voltage,
    }


def supplied_demand_readings_answer(message: str) -> dict | None:
    """Compute a peak from explicitly supplied interval-average kW readings."""
    if not re.search(r"\b(?:demand|peak|maximum)\b", message, re.IGNORECASE):
        return None
    match = re.search(
        r"\breadings?\b[^:]{0,60}:\s*(.+?)(?=\.\s+[A-Z]|\n|$)",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    values = [
        float(value.replace(",", ""))
        for value in re.findall(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*kW\b", match.group(1), re.IGNORECASE)
    ]
    if not values:
        return None
    peak = max(values)
    peak_index = values.index(peak) + 1
    target_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*kW\s+(?:cap|limit|target)\b|"
        r"\b(?:cap|limit|target)(?:\s+of|\s+at)?\s+(\d+(?:\.\d+)?)\s*kW\b",
        message,
        re.IGNORECASE,
    )
    lines = [
        f"Maximum demand from the supplied interval-average readings: {peak:.2f} kW.",
        f"It occurred at reading {peak_index} of {len(values)} in the order provided.",
        f"Calculation: max({', '.join(f'{value:g}' for value in values)}) = {peak:g} kW.",
    ]
    if target_match:
        target = float(next(group for group in target_match.groups() if group is not None))
        reduction = max(peak - target, 0.0)
        percentage = reduction / peak * 100 if peak > 0 else 0.0
        if reduction:
            lines.append(f"To meet a {target:g} kW cap, reduce demand by {reduction:.2f} kW ({percentage:.2f}% of the peak).")
        else:
            lines.append(f"The peak is already at or below the {target:g} kW cap; no reduction is required.")
    lines.append("This uses the readings as supplied; the interval length and meter configuration determine whether it matches the utility's billing-demand method.")
    return {"reply": "\n".join(lines), "peak_kw": peak, "peak_reading_index": peak_index, "readings_kw": values}


def requested_top_limit(message: str) -> int | None:
    match = re.search(
        r"\btop\s+(\d{1,2}|five|ten)\b"
        r"|\b(\d{1,2}|five|ten)\s+(?:latest|most recent)\s+alarms?\b"
        r"|\b(?:latest|most recent|last|up to)\s+(\d{1,2}|five|ten)\s+(?:(?:active|acknowledged|open)\s+)?alarms?\b",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    token = next(group for group in match.groups() if group is not None)
    value = {"five": 5, "ten": 10}.get(token.lower())
    return value if value is not None else min(max(int(token), 1), 20)


def requested_forecast_days(message: str) -> int | None:
    match = re.search(r"\b(?:next|coming|following)\s+(\d{1,2})\s+days?\b", message, re.IGNORECASE)
    if match:
        return min(max(int(match.group(1)), 1), 14)
    return 1 if "tomorrow" in message.lower() else None


def requested_energy_extrema(message: str) -> tuple[bool, bool]:
    lowered = message.lower()
    if not re.search(r"\b(?:daily|day|days|reading|readings|week)\b", lowered):
        return False, False
    return (
        bool(re.search(r"\b(?:highest|maximum|peak)\b", lowered)),
        bool(re.search(r"\b(?:lowest|minimum)\b", lowered)),
    )


def requested_question_parts(message: str) -> list[str]:
    parts = re.split(
        r"(?<=[?.!])\s+|;\s*|\s+and\s+(?=(?:what|which|how|show|list|give|compare|rank|find)\b)",
        message.strip(), flags=re.IGNORECASE,
    )
    return [part.strip(" .?;!") for part in parts if part.strip(" .?;!")]


def wants_device_usage_ranking(message: str) -> bool:
    lowered = message.lower()
    has_device = bool(re.search(r"\b(?:devices?|meters?|consumers?)\b", lowered))
    has_usage = bool(re.search(r"\b(?:energy|usage|consumption|consuming|kwh)\b", lowered))
    has_rank = bool(re.search(r"\b(?:rank|ranking|top|most|highest|largest|biggest|contribut(?:e|ed|ion))\b", lowered))
    return has_device and has_usage and has_rank


def wants_sitewide_energy_total(message: str) -> bool:
    lowered = message.lower()
    asks_kwh = bool(re.search(r"\b(?:kwh|kilowatt[- ]?hours?|energy|consumption|usage)\b", lowered))
    if not asks_kwh or wants_device_usage_ranking(message):
        return False
    asks_sitewide = bool(re.search(
        r"\b(?:overall|site[- ]wide|whole site|entire site|all devices?|all meters?|all of (?:the )?(?:our )?devices?|total site|site total|"
        r"total energy|total kwh|combined energy)\b",
        lowered,
    ))
    asks_total = bool(re.search(r"\b(?:overall|total|sum|combined)\b", lowered))
    return asks_sitewide and asks_total


DEVICE_REFERENCE_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "can", "check", "device", "devices", "for", "get",
    "give", "how", "i", "in", "is", "it", "last", "me", "meter", "meters", "of", "on", "please",
    "reading", "show", "the", "this", "to", "today", "tomorrow", "trend", "what", "when", "where",
    "which", "with", "yesterday", "voltage", "volt", "volts", "current", "amp", "amps", "amperes",
    "energy", "kwh", "mwh", "consumption", "usage", "demand", "kw", "power", "factor", "frequency",
    "hz", "thd", "parameter", "parameters", "metric", "metrics", "latest", "recent", "now", "value",
    "values", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "for", "past",
    "readings", "similar", "names", "there", "then", "also", "please", "could", "would",
    "last", "time", "series", "timeseries", "plot", "chart", "table", "available", "data", "please",
    "overall", "all", "site", "total", "sitewide", "highest", "top", "rank", "ranking", "consumer",
    "consumers", "most", "used", "use", "using", "used", "usage",
}


def has_named_device_reference(message: str) -> bool:
    if requested_device_id(message) is not None:
        return False
    tokens = re.findall(r"[a-z0-9]+", message.lower())
    return any(token not in DEVICE_REFERENCE_STOPWORDS for token in tokens)


def wants_named_device_metric_query(message: str) -> bool:
    return bool(
        question_metrics(message)
        and not wants_sitewide_energy_total(message)
        and not wants_device_usage_ranking(message)
        and not asks_for_device_inventory(message)
        and has_named_device_reference(message)
    )


def asks_for_device_inventory(message: str) -> bool:
    """Recognize inventory requests without calling other intent predicates."""
    lowered = message.lower()
    # A device-name telemetry question often starts with "get/show me" and
    # mentions the singular word "device". Do not mistake that for a request
    # to list the site's inventory.
    explicit_inventory = bool(re.search(
        r"\b(?:list|show|give)\b.*\b(?:devices|meters|device\s+ids?|meter\s+ids?|all\s+devices?|every\s+device)\b"
        r"|\bget\b.*\b(?:all|every|list of)\s+(?:devices?|meters?)\b"
        r"|\bwhich\s+(?:devices?|meters?)\b",
        lowered,
    ))
    if question_metrics(message) and has_named_device_reference(message) and not explicit_inventory:
        return False
    return bool(
        explicit_inventory
        or re.search(r"\bwhich\s+(?:devices?|meters?)\b.*\b(?:have data|available|online|offline|status|best|suitable)\b", lowered)
        or re.search(r"\b(?:devices?|meters?)\b.*\b(?:in|under|for)\s+(?:this|selected|current)\s+site\b", lowered)
        or re.search(r"\b(?:best|suitable)\b.*\b(?:voltage|current|trend|testing)\b", lowered)
    )


def normalize_device_name(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def resolve_named_device_reference(turn_id: str, message: str, context: dict, request_id: str) -> dict:
    """Resolve a user-facing device name against this site's authorized inventory."""
    if not context.get("site_id"):
        return {"status": "clarification", "reply": "Select a site first, then ask for the device by name."}
    reference_tokens = {
        token for token in normalize_device_name(message).split()
        if token not in DEVICE_REFERENCE_STOPWORDS
    }
    if not reference_tokens:
        return {"status": "clarification", "reply": "Which device name should I use?"}
    try:
        inventory_context = {key: value for key, value in context.items() if key != "device_id"}
        inventory_context["limit"] = 100
        inventory_args = build_historical_arguments("site_device_list", inventory_context)
        inventory_result = call_authorized_internal_operation(
            turn_id, "site_device_list", inventory_args, request_id, message
        )
        inventory = historical_result_data(inventory_result)
        devices = first_list(inventory, ("devices", "rows", "items", "meters"))
    except Exception as error:
        log_event("device_name_lookup_failed", request_id=request_id, error=str(error))
        return {
            "status": "unavailable",
            "reply": "I couldn't look up device names from the selected site. Please retry after the DaxView device list is available.",
        }

    matches = []
    exact_matches = []
    normalized_message = f" {normalize_device_name(message)} "
    reference_text = " ".join(
        token for token in normalize_device_name(message).split()
        if token not in DEVICE_REFERENCE_STOPWORDS
    )
    for device in devices:
        if not isinstance(device, dict):
            continue
        name = first_value(device, ("device_name", "name", "meter_name", "label"))
        device_id = first_value(device, ("device_id", "id", "meter_id"))
        if not name or device_id is None:
            continue
        normalized_name = normalize_device_name(name)
        name_tokens = set(normalized_name.split())
        if not name_tokens:
            continue
        if f" {normalized_name} " in normalized_message:
            exact_matches.append((1.0, str(name), device_id, device))
            score = 1.0
        else:
            token_score = len(name_tokens & reference_tokens) / len(name_tokens)
            fuzzy_score = SequenceMatcher(None, reference_text, normalized_name).ratio()
            score = max(token_score, fuzzy_score)
        if score >= 0.24:
            matches.append((score, str(name), device_id, device))
    # An exact full device name is stronger evidence than token overlap. This
    # matters for names such as 96S2_Freezer_3_1 and 96S2_Freezer_3_3, where
    # converting tokens to a set loses the distinguishing repeated suffix.
    if exact_matches:
        exact_matches.sort(key=lambda item: (item[1].lower(), str(item[2])))
        exact_ids = {str(item[2]) for item in exact_matches}
        if len(exact_ids) == 1:
            score, name, device_id, _device = exact_matches[0]
            try:
                numeric_id = int(device_id)
            except (TypeError, ValueError):
                numeric_id = None
            if numeric_id is not None:
                debug_trace_event(
                    "device_name_resolved_debug",
                    request_id=request_id,
                    device_name=name,
                    device_id=numeric_id,
                    match_score=score,
                    match_type="exact_name",
                )
                return {"status": "matched", "device_id": numeric_id, "device_name": name}
        # Duplicate exact names remain ambiguous, but unrelated fuzzy matches
        # must not be shown as alternatives.
        matches = exact_matches
    matches.sort(key=lambda item: (-item[0], item[1].lower()))
    if not matches:
        # Even when token matching fails, offer the closest available names so
        # the user can select one instead of restarting with site/time details.
        for device in devices:
            if not isinstance(device, dict):
                continue
            name = first_value(device, ("device_name", "name", "meter_name", "label"))
            device_id = first_value(device, ("device_id", "id", "meter_id"))
            if name and device_id is not None:
                score = SequenceMatcher(None, reference_text, normalize_device_name(name)).ratio()
                matches.append((score, str(name), device_id, device))
        matches.sort(key=lambda item: (-item[0], item[1].lower()))
        if not matches:
            return {
                "status": "clarification",
                "reply": "I couldn't find that name in the available devices. Please enter a more distinctive part of the device name.",
            }

    top_score = matches[0][0]
    close_matches = [item for item in matches if item[0] >= max(0.4, top_score - 0.12)]
    unique_top = len({str(item[2]) for item in close_matches}) == 1
    if top_score >= 0.72 and unique_top:
        score, name, device_id, device = matches[0]
        try:
            numeric_id = int(device_id)
        except (TypeError, ValueError):
            numeric_id = None
        if numeric_id is not None:
            debug_trace_event(
                "device_name_resolved_debug",
                request_id=request_id,
                device_name=name,
                device_id=numeric_id,
                match_score=round(score, 3),
            )
            return {"status": "matched", "device_id": numeric_id, "device_name": name}

    candidates = (close_matches if len(close_matches) > 1 else matches[:5])[:5]
    choices = []
    seen_device_ids = set()
    for _, name, device_id, device in candidates:
        if str(device_id) in seen_device_ids:
            continue
        seen_device_ids.add(str(device_id))
        detail = str(device.get("status") or device.get("data_status") or "")
        device_type = device.get("device_type") or device.get("type") or device.get("model")
        building_name = device.get("building_name") or device.get("building")
        if isinstance(building_name, dict):
            building_name = building_name.get("name")
        if building_name:
            detail = f"{detail} · {building_name}" if detail else str(building_name)
        if device_type:
            detail = f"{detail} · {device_type}" if detail else str(device_type)
        choices.append({"label": name, "value": str(device_id), "description": detail})
    if len(choices) > 1:
        candidate_names = [choice["label"] for choice in choices]
        candidate_list = "\n".join(f"• {name}" for name in candidate_names)
        prompt = (
            "I found more than one close device name:\n"
            + candidate_list
            + "\nWhich one do you mean? Reply with the device name and I’ll continue with your original metric and time range."
        )
    else:
        candidate_names = [choice["label"] for choice in choices]
        prompt = (
            f"I couldn't match that name exactly. The closest device is {candidate_names[0]}. "
            "Reply with its name to continue, or enter a more specific name."
            if candidate_names else
            "I couldn't match that name exactly. Please enter a more specific device name."
        )
    debug_trace_event(
        "device_name_match_ambiguous_debug",
        request_id=request_id,
        candidates=[{"name": item[1], "device_id": str(item[2]), "score": round(item[0], 3)} for item in candidates],
    )
    return {"status": "clarification", "reply": prompt, "choices": choices}


def device_metric_selection_template(message: str, use_device_id: bool = True) -> str:
    metric_labels = {
        "energy": "energy (kWh)",
        "power_factor": "power factor",
    }
    metrics = question_metrics(message)
    metric_text = " and ".join(metric_labels.get(metric, metric) for metric in metrics) or "requested metric"
    period_match = re.search(
        r"\b(?:last|past)\s+\d{1,3}\s+(?:hours?|days?|weeks?|months?)\b"
        r"|\b(?:today|yesterday|this week|last week|this month|last month)\b",
        message,
        re.IGNORECASE,
    )
    period = period_match.group(0) if period_match else "the last 7 days"
    device_ref = "device ID {value}" if use_device_id else "device {value}"
    return f"Show {metric_text} readings for {device_ref} for {period}."


def wants_lowest_consumer(message: str) -> bool:
    lowered = message.lower()
    return bool(re.search(r"\b(?:lowest|least|minimum|smallest)\b", lowered) and re.search(r"\b(?:device|meter|consumer|energy|consume|consumption|usage)\b", lowered))


def wants_top_lowest_sum(message: str) -> bool:
    lowered = message.lower()
    return wants_lowest_consumer(message) and bool(re.search(r"\b(?:sum|total|add|combined|their\s+2|both)\b", lowered))


def wants_all_devices(message: str) -> bool:
    return bool(re.search(r"\b(?:all|every)\s+(?:(?:of\s+)?the\s+)?devices?\b", message, re.IGNORECASE))


def wants_energy_breakdown_request(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\b(?:breakdown|break\s+down|split|contribution)\b", lowered)
        and re.search(r"\b(?:energy|kwh|consumption|usage|device|meter|building|value|total)\b", lowered)
    )


def wants_full_energy_breakdown(message: str) -> bool:
    lowered = message.lower()
    return wants_all_devices(message) or bool(re.search(
        r"\b(?:full|complete|entire|every)\s+(?:energy\s+)?breakdown\b"
        r"|\b(?:all|every)\s+(?:of\s+)?(?:the\s+)?devices?\b"
        r"|\ball\s+device\s+details\b",
        lowered,
    ))


def wants_device_inventory(message: str) -> bool:
    lowered = message.lower()
    if re.search(r"\b(?:alarm|alarms|alert|alerts)\b", lowered) and not any(
        term in lowered for term in ("energy", "kwh", "consumption", "usage", "demand")
    ):
        return False
    if (wants_device_usage_ranking(message) or wants_highest_demand_device(message)
            or wants_sitewide_energy_total(message)):
        return False
    if re.search(r"\b(?:top|highest|maximum|peak|rank|ranking)\b", lowered) and question_metrics(message):
        return False
    return asks_for_device_inventory(message)


def wants_device_capability_discovery(message: str) -> bool:
    lowered = message.lower()
    return wants_device_inventory(message) and bool(re.search(r"\b(?:voltage|current|telemetry|trend|data|capabilit|parameter|testing)\b", lowered))


def wants_device_metric_timeseries(message: str) -> bool:
    if requested_device_id(message) is None:
        return False
    lowered = message.lower()
    if "demand_peak_summary" in select_historical_operations_without_device_timeseries(message):
        return False
    return bool(
        question_metrics(message)
        or re.search(r"\b(?:energy|kwh|consumption|usage|demand|kw|voltage|current|power factor|frequency|thd)\b", lowered)
    )


def select_historical_operations_without_device_timeseries(message: str) -> list[str]:
    lowered = message.lower()
    operations = []
    for tool_name, phrases in DAXVIEW_TOOL_KEYWORDS.items():
        if any(phrase in lowered for phrase in phrases):
            operations.append(tool_name)
    return operations


def explicit_mcp_tool_request(message: str) -> str | None:
    lowered = message.lower()
    match = re.search(r"\b(?:test|call|run|use)\s+(?:the\s+)?(?:mcp\s+)?tool\s+([a-z_][a-z0-9_]*)\b", lowered)
    tool_name = match.group(1) if match and match.group(1) in DAXVIEW_ALLOWED_HISTORICAL_TOOLS else next(
        (name for name in DAXVIEW_ALLOWED_HISTORICAL_TOOLS if re.search(rf"\b(?:test|call|run|use)\s+{re.escape(name)}\b", lowered)),
        None,
    )
    if tool_name:
        operations = [op for op in re.split(r"\W+", lowered) if op in {"list", "metrics", "values", "energy_rows", "active", "frequency", "detail", "overview", "energy"}]
        operation = operations[0] if operations else None
        return internal_for_public_operation(tool_name, operation)
    return None


def internal_for_public_operation(tool_name: str, operation: str | None = None) -> str:
    reverse = {
        ("device_data", "list"): "site_device_list",
        ("device_data", "metrics"): "telemetry_metric_catalog",
        ("device_data", "values"): "telemetry_timeseries",
        ("device_data", "energy_rows"): "device_energy_breakdown",
        ("alarm_data", "active"): "active_alarm_summary",
        ("alarm_data", "frequency"): "alarm_frequency_summary",
        ("alarm_data", "detail"): "alarm_detail_lookup",
        ("site_summary", "overview"): "site_metadata_summary",
        ("site_summary", "energy"): "site_energy_summary",
    }
    defaults = {"device_data": "site_device_list", "alarm_data": "active_alarm_summary", "site_summary": "site_metadata_summary"}
    return reverse.get((tool_name, operation), defaults.get(tool_name, ""))


def select_public_tools(internal_operations: list[str]) -> list[str]:
    tools = []
    for intent in internal_operations:
        tool, _ = public_tool_for_internal(intent)
        if tool not in tools:
            tools.append(tool)
    return tools


def planner_default_intents(message: str, tools: list[str]) -> list[str]:
    lowered = message.lower()
    defaults = []
    if "device_data" in tools:
        if re.search(r"\b(?:maximum demand|max demand|peak demand|highest demand|maximum kw|max kw|peak kw|highest kw)\b", lowered):
            defaults.append("demand_peak_summary")
        elif wants_device_usage_ranking(message):
            defaults.append("device_energy_breakdown")
        elif any(term in lowered for term in ("available metric", "what metrics", "supports voltage", "supports current")) or (
            "metric" in lowered and any(term in lowered for term in ("available", "support", "which", "what"))
        ):
            defaults.append("telemetry_metric_catalog")
        elif question_metrics(message) or any(term in lowered for term in ("trend", "reading", "maximum demand", "max demand", "peak demand", "voltage", "current", "power factor", "thd")):
            defaults.append("telemetry_timeseries")
        elif wants_device_inventory(message):
            defaults.append("site_device_list")
        else:
            defaults.append("site_device_list")
    if "alarm_data" in tools:
        if any(term in lowered for term in ("alarm id", "event id", "alarm detail", "alarm details", "what caused", "explain this alarm")):
            defaults.append("alarm_detail_lookup")
        elif any(term in lowered for term in ("latest", "most recent", "frequency", "most common", "most often", "history", "occurred", "occurrence", "how many alarms", "alarm count", "top alarms")):
            defaults.append("alarm_frequency_summary")
        else:
            defaults.append("active_alarm_summary")
    if "site_summary" in tools:
        if any(term in lowered for term in ("overview", "site detail", "site information", "site metadata", "buildings under")):
            defaults.append("site_metadata_summary")
        else:
            defaults.append("site_energy_summary")
    return defaults


def select_historical_operations(message: str) -> list[str]:
    if is_historical_readings_clarification_request(message):
        return ["telemetry_timeseries"]
    explicit_tool = explicit_mcp_tool_request(message)
    if explicit_tool:
        return [explicit_tool]
    lowered = message.lower()
    if wants_sitewide_energy_total(message):
        return ["site_energy_summary"]
    if wants_named_device_metric_query(message):
        return ["telemetry_timeseries"]
    if re.search(r"\b(?:maximum demand|max demand|peak demand|highest demand|maximum kw|max kw|peak kw|highest kw)\b", lowered):
        return ["demand_peak_summary"]
    if re.search(r"\b(?:maximum|highest|peak)\b", lowered) and re.search(r"\b(?:power|kw|watt)\b", lowered):
        return ["demand_peak_summary"]
    if requested_device_id(message) is not None and question_metrics(message) and re.search(
        r"\b(?:latest|most recent|current)\b", lowered
    ):
        return ["latest_telemetry_snapshot"]
    # The word "list" is common in alarm requests; resolve alarm intent before
    # the generic device-inventory heuristic can claim the request.
    asks_for_other_data = any(
        term in lowered
        for term in ("energy", "kwh", "consumption", "usage", "demand", "voltage", "current", "power")
    )
    if re.search(r"\b(?:alarm|alarms|alert|alerts)\b", lowered) and not asks_for_other_data:
        if re.search(r"\b(?:detail|details|explain|cause)\b", lowered) or re.search(r"\b(?:alarm|event)\s*(?:id|#|:)\s*\d+", lowered):
            return ["alarm_detail_lookup"]
        if re.search(r"\b(?:frequency|occurred|occurrence|how many|count|most common|most frequent|rank|latest|most recent|history)\b", lowered):
            return ["alarm_frequency_summary"]
        return ["active_alarm_summary"]

    ranked_metrics = set(question_metrics(message)) - {"energy"}
    if re.search(r"\b(?:top|highest|maximum|peak|rank|ranking)\b", lowered) and ranked_metrics:
        if ranked_metrics == {"frequency"}:
            return ["telemetry_metric_ranking"]
        if "demand" in ranked_metrics and wants_highest_demand_device(message):
            return ["demand_peak_summary"]
        if requested_device_id(message) is None:
            return ["telemetry_metric_catalog"]
        return ["demand_peak_summary" if "demand" in ranked_metrics else "telemetry_timeseries"]

    operations = select_historical_operations_without_device_timeseries(message)
    if wants_device_metric_timeseries(message):
        operations = [name for name in operations if name != "site_energy_summary"]
        if "telemetry_timeseries" not in operations:
            operations.append("telemetry_timeseries")
    if wants_available_parameters_query(message):
        if "data_availability_summary" not in operations:
            operations.append("data_availability_summary")
        operations = [name for name in operations if name != "telemetry_timeseries"]
    if wants_device_inventory(message) and "site_device_list" not in operations:
        operations.append("site_device_list")
    if "current" in question_metrics(message) and "telemetry_timeseries" not in operations:
        operations.append("telemetry_timeseries")
    if wants_device_capability_discovery(message) and requested_device_id(message) is None:
        operations = [name for name in operations if name != "telemetry_timeseries"]
    if wants_device_usage_ranking(message):
        # A chart/table request for ranked consumers should visualize the
        # returned energy_rows. The generic "chart" keyword must not add a
        # per-device telemetry_timeseries request (which also requires a
        # specific device ID).
        operations = [name for name in operations if name not in {"site_device_list", "telemetry_timeseries"}]
        if "device_energy_breakdown" not in operations:
            operations.append("device_energy_breakdown")
        if not re.search(r"\b(?:daily|site energy|site-wide|overall|total site|site total|site usage)\b", lowered):
            operations = [name for name in operations if name != "site_energy_summary"]
        if any(term in lowered for term in ("offline", "online", "status")) or wants_all_devices(message):
            if "site_device_list" not in operations:
                operations.append("site_device_list")
    if wants_energy_breakdown_request(message):
        operations = [name for name in operations if name != "site_energy_summary"]
        if "device_energy_breakdown" not in operations:
            operations.append("device_energy_breakdown")
        # Device inventory supplies a safe device-to-building name map when
        # the energy_rows payload omits building metadata.
        if "site_device_list" not in operations:
            operations.append("site_device_list")
    if "tomorrow" in lowered and any(term in lowered for term in ("energy", "kwh", "consumption", "usage")):
        operations.append("energy_forecast")
    if "alarm_frequency_summary" in operations and not any(
        term in lowered for term in ("energy", "kwh", "consumption", "usage", "demand")
    ):
        operations = [name for name in operations if name not in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers", "site_energy_summary"}]
    if "site_energy_summary" in operations and any(requested_energy_extrema(message)) and not any(
        term in lowered for term in ("top consumer", "top device", "by device", "which device")
    ) and not wants_device_usage_ranking(message):
        operations = [name for name in operations if name not in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}]
    if wants_sitewide_energy_total(message):
        return ["site_energy_summary"]
    if "energy_forecast" in operations and not any(
        term in lowered for term in ("compare", "comparison", "difference", "top consumer", "energy summary")
    ):
        operations = [name for name in operations if name != "site_energy_summary"]
    if "demand_peak_summary" in operations and "site_energy_summary" in operations and not any(
        term in lowered for term in ("daily energy", "energy consumption", "energy usage", "site energy", "kwh")
    ):
        operations.remove("site_energy_summary")
    if "demand_peak_summary" in operations and any(
        phrase in lowered for phrase in ("details", "breakdown", "devices", "which meter", "which meters")
    ):
        for tool_name in ("site_device_list",):
            if tool_name not in operations:
                operations.append(tool_name)
    if "site_metadata_summary" in operations and any(
        phrase in lowered
        for phrase in (
            "all details",
            "summary details",
            "summary for this site",
            "summary of this site",
            "site summary",
            "site information",
            "devices and",
            "alarms and",
            "other info",
            "complete",
            "overview",
        )
    ):
        for tool_name in ("site_device_list", "active_alarm_summary", "meter_status_summary"):
            if tool_name not in operations:
                operations.append(tool_name)
    if (
        "site_energy_summary" not in operations
        and "energy_comparison_summary" not in operations
        and any(word in lowered for word in ("energy", "kwh", "consumption", "usage"))
        and any(word in lowered for word in ("compare", "comparison", "difference", "between"))
    ):
        operations.append("energy_comparison_summary")
    if "report_summary" in operations:
        for tool_name in ("site_energy_summary", "device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers", "alarm_frequency_summary"):
            if tool_name in operations:
                operations.remove(tool_name)
    deduped = []
    for operation in operations:
        if operation not in deduped:
            deduped.append(operation)
    return deduped[:6]


def is_historical_readings_clarification_request(message: str) -> bool:
    lowered = message.lower()
    asks_readings = bool(re.search(r"\b(?:historical|history|past)\b.{0,60}\b(?:device\s+)?readings?\b|\b(?:device\s+)?readings?\b.{0,60}\b(?:historical|history|past)\b", lowered))
    asks_to_clarify = bool(re.search(r"\b(?:please\s+)?ask\b.{0,80}\b(?:which|what)\b.{0,60}\b(?:metric|parameter|period|time\s*range|device|meter)\b|\b(?:clarify|ask me)\b.{0,80}\b(?:metric|parameter|period|time\s*range|device|meter)\b", lowered))
    before_fetch = bool(re.search(r"\b(?:before|prior to)\s+(?:you\s+)?(?:fetch|retrieve|read|query)\b", lowered))
    return asks_readings and (asks_to_clarify or before_fetch)


def plan_turn_with_llm(message: str, context: dict, request_id: str):
    messages = [
        {"role": "system", "content": planner_system_prompt(DAXVIEW_TOOL_DESCRIPTIONS)},
    ]
    messages.append({
        "role": "user",
        "content": json.dumps(
            {
                "question": message,
                "context_scope": {
                    "has_site": bool(context.get("site_id")),
                    "has_building": bool(context.get("building_id")),
                    "has_device": bool(context.get("device_id")),
                },
            },
            ensure_ascii=True,
        ),
    })
    started_at = time.perf_counter()
    log_event("planner_request", request_id=request_id, model=PLANNER_MODEL, prompt_preview=preview(str(messages)))
    data = ollama_json(
        "/api/chat",
        {
            "model": PLANNER_MODEL,
            "messages": messages,
            "stream": False,
            "format": planner_json_schema(),
            "options": {"temperature": 0},
        },
        timeout=PLANNER_TIMEOUT,
    )
    plan = parse_planner_response(data, DAXVIEW_ALLOWED_HISTORICAL_TOOLS)
    latency_ms = round((time.perf_counter() - started_at) * 1000)
    log_event("planner_response", request_id=request_id, model=PLANNER_MODEL, latency_ms=latency_ms, tools=plan.tools, confidence=plan.confidence)
    debug_trace_event(
        "planner_intention_debug",
        request_id=request_id,
        model=PLANNER_MODEL,
        latency_ms=latency_ms,
        tools=plan.tools,
        confidence=plan.confidence,
        slots=getattr(plan.slots, "__dict__", {}),
        clarification=getattr(plan.clarification, "__dict__", None) if getattr(plan, "clarification", None) else None,
    )
    return plan, latency_ms


def planner_slot_context(message: str, context: dict, plan) -> dict:
    merged = dict(context)
    slots = plan.slots
    if slots.limit is not None:
        merged["limit"] = slots.limit
    if slots.metric:
        metric_map = {"kw": "demand", "kwh": "energy", "amps": "current", "amp": "current", "a": "current"}
        merged["metric"] = metric_map.get(str(slots.metric).lower(), slots.metric)
    if slots.building_ref and str(slots.building_ref).isdigit():
        merged["building_id"] = int(slots.building_ref)
    if slots.alarm_id:
        merged["alarm_id"] = slots.alarm_id
    if slots.device_refs:
        for ref in slots.device_refs:
            # Device names commonly contain digits (e.g. 96S2_Freezer_3_1).
            # Only a reference consisting entirely of digits is an ID.
            normalized_ref = str(ref).strip()
            if normalized_ref.isdigit() and not wants_named_device_metric_query(message):
                merged["device_id"] = int(normalized_ref)
                break
    return merged


def choose_historical_operations(message: str, context: dict, request_id: str) -> tuple[list[str], dict, dict | None]:
    regex_operations = select_historical_operations(message)
    if not (AI_PLANNER_ENABLED or AI_PLANNER_SHADOW_MODE):
        debug_trace_event(
            "planner_intention_debug",
            request_id=request_id,
            planner_enabled=False,
            regex_tools=regex_operations,
            selected_tools=regex_operations,
            context=context,
        )
        return regex_operations, context, None
    if AI_PLANNER_ENABLED and not AI_PLANNER_SHADOW_MODE and wants_device_usage_ranking(message):
        # This intent has an unambiguous deterministic route. Avoid spending
        # the planner's full timeout on a simple ranking request.
        debug_trace_event(
            "planner_intention_debug",
            request_id=request_id,
            planner_skipped=True,
            reason="deterministic_top_consumer_route",
            regex_tools=regex_operations,
            selected_tools=regex_operations,
            context=context,
        )
        return regex_operations, context, None
    if AI_PLANNER_ENABLED and not AI_PLANNER_SHADOW_MODE and wants_named_device_metric_query(message):
        debug_trace_event(
            "planner_intention_debug",
            request_id=request_id,
            planner_skipped=True,
            reason="deterministic_named_device_route",
            regex_tools=regex_operations,
            selected_tools=regex_operations,
            context=context,
        )
        return regex_operations, context, None
    try:
        plan, latency_ms = plan_turn_with_llm(message, context, request_id)
        planned_context = planner_slot_context(message, context, plan)
        planned_public_tools = [tool for tool in plan.tools if tool in DAXVIEW_ALLOWED_HISTORICAL_TOOLS]
        demand_intent = "demand_peak_summary" in regex_operations
        if demand_intent and "device_data" in planned_public_tools:
            # A timed-out/ambiguous planner must not replace deterministic demand routing
            # with a generic device inventory request.
            planned_tools = ["demand_peak_summary"]
        else:
            planned_tools = [intent for intent in regex_operations if public_tool_for_internal(intent)[0] in planned_public_tools]
        if not planned_tools:
            planned_tools = planner_default_intents(message, planned_public_tools)
        if AI_PLANNER_SHADOW_MODE or not AI_PLANNER_ENABLED:
            log_event(
                "planner_shadow_compare",
                request_id=request_id,
                regex_tools=regex_operations,
                planner_tools=planned_public_tools,
                agreement=select_public_tools(regex_operations) == planned_public_tools,
                latency_ms=latency_ms,
                confidence=plan.confidence,
            )
            return regex_operations, context, {
                "status": "shadow",
                "tools": [{"tool": tool, "arguments": {}} for tool in planned_public_tools],
                "slots": planned_context,
            }
        if plan.clarification and plan.clarification.question and not regex_operations:
            debug_trace_event(
                "planner_decision_debug",
                request_id=request_id,
                status="clarification",
                regex_tools=regex_operations,
                planner_tools=planned_public_tools,
                selected_tools=[],
                context=planned_context,
                clarification=plan.clarification.question,
            )
            return [], {**planned_context, "_planner_clarification": plan.clarification.question}, {
                "status": "clarification",
                "tools": [],
                "slots": planned_context,
            }
        if plan.clarification and regex_operations:
            debug_trace_event(
                "planner_clarification_overridden_debug",
                request_id=request_id,
                regex_tools=regex_operations,
                planner_tools=planned_public_tools,
                reason="deterministic_intent_available",
            )
        selected_tools = planned_tools or regex_operations
        debug_trace_event(
            "planner_decision_debug",
            request_id=request_id,
            status="ok",
            regex_tools=regex_operations,
            planner_tools=planned_tools,
            selected_tools=selected_tools,
            context=planned_context,
            confidence=plan.confidence,
        )
        return selected_tools, planned_context, {
            "status": "ok",
            "tools": [{"tool": tool, "arguments": {}} for tool in planned_public_tools],
            "slots": planned_context,
        }
    except Exception as error:
        log_event("planner_fallback", request_id=request_id, reason=str(error), regex_tools=regex_operations)
        return regex_operations, context, None


def first_regex_int(message: str, patterns: tuple[str, ...]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def requested_device_id(message: str) -> int | None:
    return first_regex_int(message, (
        r"\bdevice\s*(?:id|#|:)?\s*(\d+)\b",
        r"\bmeter\s*(?:id|#|:)?\s*(\d+)\b",
        r"\(\s*ID\s*[:#-]?\s*(\d+)\s*[,)]",
        r"^\s*ID\s*[:#-]?\s*(\d+)\b",
    ))


def requested_metric(message: str) -> str:
    if metric_clarification(message):
        raise ValueError(metric_clarification(message))
    metrics = question_metrics(message)
    return metrics[0] if metrics else "energy"


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
    match = re.search(
        r"\b(?:last\s+|past\s+|for\s+)?(\d{1,3})\s*[- ]?\s*"
        r"(minute|minutes|min|mins|hour|hours|hr|hrs|day|days|week|weeks|month|months)\b",
        lowered,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit in {"minute", "minutes", "min", "mins"}:
        duration = timedelta(minutes=amount)
    elif unit in {"hour", "hours", "hr", "hrs"}:
        duration = timedelta(hours=amount)
    elif unit.startswith("week"):
        days = amount * 7
        duration = timedelta(days=days)
    elif unit.startswith("month"):
        days = amount * 30
        duration = timedelta(days=days)
    else:
        days = amount
        duration = timedelta(days=days)
    return end - max(duration, timedelta(minutes=1)), end


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
    explicit_range = requested_explicit_date_range(message, end.year)
    requested_dates = requested_comparison_dates(message, end.year)
    requested_months = [] if requested_dates else requested_month_ranges(message, end.year)
    relative_range = relative_historical_range(message)
    local_now = end.astimezone(timezone(timedelta(hours=8)))
    if explicit_range:
        first_day, last_day = explicit_range
        start = datetime(first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc) - timedelta(hours=8)
        end = datetime(last_day.year, last_day.month, last_day.day, tzinfo=timezone.utc) + timedelta(hours=16)
    elif requested_dates and relative_range:
        start, end = relative_range
    elif requested_dates:
        parsed_dates = [datetime.fromisoformat(date).date() for date in requested_dates]
        start_date = min(parsed_dates)
        end_date = max(parsed_dates)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) + timedelta(hours=16)
    elif requested_months:
        start_date = min(month_start for month_start, _ in requested_months)
        end_date = max(month_end for _, month_end in requested_months)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
    elif "last month" in lowered:
        current_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        previous_month = (current_month - timedelta(days=1)).replace(day=1)
        start, end = previous_month.astimezone(timezone.utc), current_month.astimezone(timezone.utc)
    elif "this month" in lowered:
        start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    elif "last week" in lowered:
        this_week = (local_now - timedelta(days=local_now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        start, end = (this_week - timedelta(days=7)).astimezone(timezone.utc), this_week.astimezone(timezone.utc)
    elif "this week" in lowered:
        start = (local_now - timedelta(days=local_now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    elif "today" in lowered:
        start = local_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    elif "yesterday" in lowered:
        today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
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


def completed_daily_range(message: str) -> dict | None:
    lowered = message.lower()
    match = re.search(r"\blast\s+(\d{1,3})\s+days?\b", lowered)
    if not match or not any(term in lowered for term in ("daily", "each day", "per day")):
        return None
    if "including today" in lowered or "include today" in lowered:
        return None
    local_midnight = datetime.now(timezone(timedelta(hours=8))).replace(hour=0, minute=0, second=0, microsecond=0)
    days = max(int(match.group(1)), 1)
    return {
        "start": (local_midnight - timedelta(days=days)).astimezone(timezone.utc).isoformat(),
        "end": local_midnight.astimezone(timezone.utc).isoformat(),
        "timezone": "Asia/Kuala_Lumpur",
    }


def build_historical_arguments(operation_id: str, context: dict, message: str = "") -> dict:
    site_id = context.get("site_id")
    if not site_id:
        raise ValueError("site_id is required for Daxview historical data")
    args = {"site_id": int(site_id)}
    if context.get("building_id"):
        args["building_id"] = int(context["building_id"])

    device_id = requested_device_id(message) or context.get("device_id")
    alarm_id = first_regex_int(
        message,
        (
            r"\balarm\s*(?:id|#|:)?\s*(\d+)\b",
            r"\bevent\s*(?:id|#|:)?\s*(\d+)\b",
        ),
    ) or context.get("alarm_id")

    if device_id:
        args["device_id"] = int(device_id)

    range_args = context.get("_time_window") or requested_historical_range(message)
    if operation_id in {
        "device_energy_ranking",
        "telemetry_top_consumers",
        "site_energy_summary",
        "alarm_frequency_summary",
        "telemetry_timeseries",
        "telemetry_metric_ranking",
        "telemetry_metric_catalog",
        "energy_comparison_summary",
        "data_availability_summary",
        "power_quality_summary",
        "demand_peak_summary",
        "tariff_cost_summary",
        "device_energy_breakdown",
        "energy_forecast",
        "anomaly_detection_summary",
        "report_summary",
    }:
        args.update(range_args)

    if operation_id in {"device_energy_ranking", "telemetry_top_consumers"}:
        if operation_id == "device_energy_ranking":
            args["start_time"] = args.pop("start")
            args["end_time"] = args.pop("end")
            args["metric"] = str(context.get("metric") or "energy")
        wants_all = wants_all_devices(message)
        args["limit"] = 100 if wants_all else requested_top_limit(message) or int(context.get("limit") or 5)
    elif operation_id == "alarm_frequency_summary":
        args["limit"] = requested_top_limit(message) or int(context.get("limit") or 10)
    elif operation_id == "site_energy_summary":
        args["bucket"] = str(context.get("bucket") or "day")
        if args["bucket"] == "day":
            args.update(completed_daily_range(message) or {})
    elif operation_id == "site_device_list":
        default_limit = 500 if wants_energy_breakdown_request(message) else 100
        args["limit"] = int(context.get("limit") or default_limit)
    elif operation_id == "telemetry_metric_catalog":
        if "start" in args:
            args["start_time"] = args.pop("start")
            args["end_time"] = args.pop("end")
        args["limit"] = int(context.get("limit") or 100)
    elif operation_id == "latest_telemetry_snapshot":
        metrics = question_metrics(message)
        if context.get("metric"):
            metrics = [str(context["metric"])]
        if metrics:
            args["metrics"] = metrics
        args["limit"] = int(context.get("limit") or 100)
    elif operation_id == "telemetry_timeseries":
        if not device_id:
            raise ValueError("device_id is required for telemetry timeseries")
        args["metric"] = context.get("metric") or requested_metric(message)
        args["phase"] = str(context.get("phase") or "all")
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
        args["bucket"] = str(context.get("bucket") or ("1h" if "hour" in message.lower() else "1d"))
        args["aggregation"] = str(context.get("aggregation") or "auto")
        if context.get("value_mode"):
            args["value_mode"] = str(context["value_mode"])
        args["limit"] = int(context.get("limit") or 500)
    elif operation_id == "telemetry_metric_ranking":
        args["metric"] = str(context.get("metric") or requested_metric(message) or "frequency")
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
        start_dt = parse_datetime(args["start_time"])
        end_dt = parse_datetime(args["end_time"])
        duration = (end_dt - start_dt) if start_dt and end_dt else timedelta(days=8)
        # Preserve the timestamp of the actual maximum for short windows. Larger
        # windows use hourly maxima to stay within the telemetry point limit.
        args["bucket"] = "raw" if duration <= timedelta(days=7) else "1h"
        args["aggregation"] = "max"
        args["value_mode"] = "measured"
        args["limit"] = min(int(context.get("limit") or 1000), 2000)
    elif operation_id == "active_alarm_summary":
        args["limit"] = requested_top_limit(message) or int(context.get("limit") or 50)
    elif operation_id == "meter_status_summary":
        args["limit"] = int(context.get("limit") or 100)
    elif operation_id == "energy_comparison_summary":
        requested_dates = requested_comparison_dates(message, datetime.now(timezone.utc).year)
        requested_months = requested_month_ranges(message, datetime.now(timezone.utc).year)
        if len(requested_dates) >= 2:
            first_date = datetime.fromisoformat(requested_dates[0]).date()
            second_date = datetime.fromisoformat(requested_dates[1]).date()
            first_start = datetime(first_date.year, first_date.month, first_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)
            first_end = datetime(first_date.year, first_date.month, first_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
            second_start = datetime(second_date.year, second_date.month, second_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)
            second_end = datetime(second_date.year, second_date.month, second_date.day, tzinfo=timezone.utc) - timedelta(hours=8)
            args["period_a_start"] = first_start.isoformat()
            args["period_a_end"] = first_end.isoformat()
            args["period_b_start"] = second_start.isoformat()
            args["period_b_end"] = second_end.isoformat()
            args.pop("start", None)
            args.pop("end", None)
        elif len(requested_months) >= 2:
            first_start_date, first_end_date = requested_months[0]
            second_start_date, second_end_date = requested_months[1]
            args["period_a_start"] = (datetime(first_start_date.year, first_start_date.month, first_start_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)).isoformat()
            args["period_a_end"] = (datetime(first_end_date.year, first_end_date.month, first_end_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)).isoformat()
            args["period_b_start"] = (datetime(second_start_date.year, second_start_date.month, second_start_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)).isoformat()
            args["period_b_end"] = (datetime(second_end_date.year, second_end_date.month, second_end_date.day, tzinfo=timezone.utc) - timedelta(days=1, hours=8)).isoformat()
            args.pop("start", None)
            args.pop("end", None)
        elif "start" in args:
            start = args.pop("start")
            end = args.pop("end")
            args["period_b_start"] = start
            args["period_b_end"] = end
            prev_start = datetime.fromisoformat(start) - (datetime.fromisoformat(end) - datetime.fromisoformat(start))
            args["period_a_start"] = prev_start.isoformat()
            args["period_a_end"] = start
    elif operation_id == "data_availability_summary":
        if context.get("metric") or question_metrics(message):
            args["metric"] = context.get("metric") or requested_metric(message)
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
    elif operation_id == "alarm_detail_lookup":
        if not alarm_id:
            raise ValueError("alarm_id is required for alarm details")
        args = {"site_id": int(site_id), "alarm_id": int(alarm_id)}
    elif operation_id in {"power_quality_summary", "demand_peak_summary"}:
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
        if operation_id == "power_quality_summary":
            args["limit"] = int(context.get("limit") or 100)
    elif operation_id == "tariff_cost_summary":
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
    elif operation_id == "device_energy_breakdown":
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
        args["group_by"] = str(context.get("group_by") or "device")
        ranking_limit = requested_top_limit(message) if wants_device_usage_ranking(message) else None
        if wants_energy_breakdown_request(message):
            # Fetch enough rows to calculate building totals, even when prose
            # only displays the top five devices by default.
            args["limit"] = min(int(context.get("limit") or 500), 500)
        else:
            args["limit"] = 100 if wants_all_devices(message) else ranking_limit or int(context.get("limit") or 20)
    elif operation_id == "energy_forecast":
        forecast_days = requested_forecast_days(message) or int(context.get("forecast_days") or 7)
        forecast_days = min(max(forecast_days, 1), 14)
        history_days = min(max(int(context.get("history_days") or 35), 28), 90)
        forecast_start = datetime.now(timezone.utc)
        args.pop("start", None)
        args.pop("end", None)
        args["forecast_start"] = forecast_start.isoformat()
        args["forecast_end"] = (forecast_start + timedelta(days=forecast_days)).isoformat()
        args["training_days"] = history_days
    elif operation_id == "anomaly_detection_summary":
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
        args["limit"] = int(context.get("limit") or 20)
    elif operation_id == "report_summary":
        args["start_time"] = args.pop("start")
        args["end_time"] = args.pop("end")
    return validate_tool_arguments(operation_id, args)


def request_daxview_data_plan(turn_id: str, operation_id: str, arguments: dict, request_id: str) -> dict:
    if not DAXVIEW_CALLBACK_BASE_URL or not DAXVIEW_CALLBACK_KEY:
        raise RuntimeError("DAXVIEW_CALLBACK_BASE_URL and DAXVIEW_CALLBACK_KEY are required for historical data")
    arguments = normalize_mcp_arguments(operation_id, arguments)
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
        if key in {
            "site_id",
            "site_name",
            "building_id",
            "device_id",
            "alarm_id",
            "alarm_event_id",
            "start",
            "end",
            "start_time",
            "end_time",
            "period_a_start",
            "period_a_end",
            "period_b_start",
            "period_b_end",
            "timezone",
            "bucket",
            "limit",
            "metric",
            "phase",
            "aggregation",
            "value_mode",
            "forecast_start",
            "forecast_end",
            "training_days",
            "operation",
            "metric",
            "metrics",
            "device_type",
            "meter_model",
            "manufacturer",
            "status",
            "has_data",
            "search",
            "severity",
        }
    }
    log_event(
        "daxview_data_plan_request",
        request_id=request_id,
        turn_id=turn_id,
        operation_id=operation_id,
        operation=arguments.get("operation"),
        arguments=safe_arguments,
    )
    debug_trace_event(
        "daxview_fetch_stage_debug",
        request_id=request_id,
        turn_id=turn_id,
        operation_id=operation_id,
        stage="data_plan_request",
        arguments=safe_arguments,
    )
    try:
        with urlopen(request, timeout=DAXVIEW_MCP_TIMEOUT) as response:
            plan = json.loads(response.read().decode("utf-8"))
            normalized_arguments = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else {}
            log_event(
                "daxview_data_plan_response",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                operation=arguments.get("operation"),
                has_authorization=bool(plan.get("authorization_id")),
                arguments=redact_debug_value(normalized_arguments),
            )
            debug_trace_event(
                "daxview_data_plan_response_debug",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                operation=arguments.get("operation"),
                has_authorization=bool(plan.get("authorization_id")),
                arguments=redact_debug_value(normalized_arguments),
                changed_arguments={
                    key: {"requested": safe_arguments.get(key), "planned": normalized_arguments.get(key)}
                    for key in sorted(set(safe_arguments) | set(normalized_arguments))
                    if safe_arguments.get(key) != normalized_arguments.get(key)
                    and key not in {"authorization_id"}
                },
            )
            debug_trace_event(
                "daxview_fetch_stage_debug",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                operation=arguments.get("operation"),
                stage="data_plan_authorized" if plan.get("authorization_id") else "data_plan_missing_authorization",
                has_authorization=bool(plan.get("authorization_id")),
                arguments=redact_debug_value(normalized_arguments),
            )
            return plan
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
        debug_trace_event(
            "daxview_fetch_stage_debug",
            request_id=request_id,
            turn_id=turn_id,
            operation_id=operation_id,
            stage="data_plan_error",
            status=error.code,
            response=safe_body,
        )
        if isinstance(safe_body, dict) and safe_body.get("error_code") == "TURN_NOT_ACTIVE":
            raise RuntimeError(
                "DaxView rejected the data plan because this turn was no longer active (TURN_NOT_ACTIVE)."
            ) from error
        raise


def call_authorized_historical_tool(operation_id: str, authorization_id: str, arguments: dict, request_id: str) -> dict:
    if operation_id not in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        raise ValueError("historical operation is not allowlisted")
    arguments = normalize_mcp_arguments(operation_id, arguments)
    debug_trace_event(
        "daxview_fetch_stage_debug",
        request_id=request_id,
        operation_id=operation_id,
        operation=arguments.get("operation"),
        stage="mcp_tool_call_start",
        arguments=redact_debug_value(arguments),
    )
    result = call_daxview_mcp_tool(operation_id, {"authorization_id": authorization_id, **arguments}, request_id)
    structured = mcp_structured_result(result)
    data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
    debug_trace_event(
        "daxview_mcp_structured_result_debug",
        request_id=request_id,
        operation_id=operation_id,
        operation=arguments.get("operation"),
        status=structured.get("status"),
        error_code=structured.get("error_code") or structured.get("code"),
        message=structured.get("message") or structured.get("detail") or structured.get("error"),
        data_keys=sorted(data.keys()) if isinstance(data, dict) else [],
        row_count=len(first_list(data, ("rows", "items", "results", "buckets", "series", "rankings", "devices"))) if isinstance(data, dict) else None,
    )
    issue = result_problem(structured) or result_problem(result)
    if issue:
        debug_trace_event(
            "daxview_fetch_stage_debug",
            request_id=request_id,
            operation_id=operation_id,
            stage="mcp_tool_invalid_data",
            issue=issue,
        )
        raise RuntimeError(f"{operation_id} could not return valid data: {issue}")
    debug_trace_event(
        "daxview_fetch_stage_debug",
        request_id=request_id,
        operation_id=operation_id,
        operation=arguments.get("operation"),
        stage="mcp_tool_valid_data",
    )
    if AI_DEBUG_DASHBOARD_ENABLED and operation_id == "site_summary" and arguments.get("operation") == "energy" and arguments.get("bucket") == "day":
        try:
            data = historical_result_data(result)
            rows = first_list(data, ("buckets", "rows", "series", "data"))
            PREDICTION_SERVICE.store.save_dataset(daily_dataset({
                "name": f"Site {arguments.get('site_id')} - authorized energy history",
                "site_id": arguments.get("site_id"), "device_id": arguments.get("device_id"),
                "unit": data.get("unit") or "kWh", "rows": rows, "source": "authorized_mcp_snapshot",
            }))
        except Exception as error:
            log_event("prediction_snapshot_skipped", request_id=request_id, reason=str(error))
    return result


def execute_authorized_internal_operation(
    turn_id: str, internal_operation: str, arguments: dict, request_id: str, message: str = "",
) -> list[dict]:
    """Run an internal EMS intent only through a plan-authorized public MCP tool."""
    records = []
    for index, (tool_name, requested_arguments) in enumerate(build_public_calls(internal_operation, arguments, message)):
        plan = request_daxview_data_plan(turn_id, tool_name, requested_arguments, request_id)
        authorization_id = plan.get("authorization_id")
        planned = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else requested_arguments
        normalized = normalize_mcp_arguments(tool_name, planned)
        if normalized.get("operation") != requested_arguments.get("operation"):
            raise RuntimeError("DaxView changed the authorized MCP operation")
        if not authorization_id:
            raise RuntimeError(f"DaxView did not return a data authorization for {tool_name}.{normalized['operation']}")
        result = call_authorized_historical_tool(tool_name, str(authorization_id), normalized, request_id)
        record_operation = internal_operation
        if internal_operation in {"energy_comparison_summary", "report_summary"}:
            record_operation = {
                ("site_summary", "energy"): "site_energy_summary",
                ("device_data", "energy_rows"): "device_energy_breakdown",
                ("alarm_data", "frequency"): "alarm_frequency_summary",
                ("site_summary", "overview"): "site_metadata_summary",
                ("device_data", "list"): "site_device_list",
                ("alarm_data", "active"): "active_alarm_summary",
            }.get((tool_name, normalized["operation"]), internal_operation)
        records.append({
            "operation_id": record_operation,
            "intent_id": internal_operation,
            "public_tool": tool_name,
            "operation": normalized["operation"],
            "arguments": normalized,
            "result": result,
            **({"period_label": "A" if index == 0 else "B"} if internal_operation == "energy_comparison_summary" else {}),
        })
    return records


def call_authorized_internal_operation(turn_id: str, internal_operation: str, arguments: dict, request_id: str, message: str = "") -> dict:
    records = execute_authorized_internal_operation(turn_id, internal_operation, arguments, request_id, message)
    if not records:
        raise RuntimeError(f"No public MCP operation was produced for {internal_operation}")
    return records[0]["result"]


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
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        nested = dict(data["data"])
        for key in ("site", "building", "device", "success"):
            if key in data and key not in nested:
                nested[key] = data[key]
        return nested
    if isinstance(data, dict) and (data or any(isinstance(data.get(key), list) for key in list_keys)):
        return data
    found = first_dict_with_list(structured, list_keys)
    return found if found else {}


def predict_daily_energy(summary: dict, forecast_days: int, timezone_name: str = "Asia/Kuala_Lumpur") -> dict:
    rows = first_list(summary, ("buckets", "rows", "series", "data"))
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).date() if timezone_name == "Asia/Kuala_Lumpur" else datetime.now(timezone.utc).date()
    readings = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        stamp = row.get("timestamp") or row.get("bucket") or row.get("start") or row.get("date")
        day = local_bucket_date(stamp, timezone_name) if isinstance(stamp, str) else None
        value = row.get("value") if row.get("value") is not None else row.get("kwh")
        if not day or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value) and value >= 0 and day < today.isoformat():
            readings[date.fromisoformat(day)] = float(value)
    days = sorted(readings)[-35:]
    if len(days) < 21:
        raise ValueError(f"Prediction needs at least 21 complete daily readings; {len(days)} were available.")
    latest = days[-1]
    if (today - latest).days > 2:
        raise ValueError(f"The latest complete energy reading is {latest.isoformat()}; refresh historical data before forecasting.")
    span = (latest - days[0]).days + 1
    coverage = len(days) / span
    if coverage < 0.8:
        raise ValueError(f"Daily energy coverage is {coverage:.0%}; at least 80% is needed for a forecast.")

    def estimate(target: date, training: list[date]) -> float:
        same_weekday = [readings[day] for day in training if day.weekday() == target.weekday()]
        values = same_weekday[-4:] if len(same_weekday) >= 2 else [readings[day] for day in training[-7:]]
        return statistics.mean(values)

    validation = days[-7:]
    errors = [abs(readings[day] - estimate(day, [earlier for earlier in days if earlier < day])) for day in validation]
    horizon = min(max(int(forecast_days), 1), 14)
    forecast = []
    first_forecast_day = max(latest + timedelta(days=1), today)
    for offset in range(horizon):
        target = first_forecast_day + timedelta(days=offset)
        forecast.append({"date": target.isoformat(), "value": round(estimate(target, days), 2)})
    return {
        "status": "ok",
        "source_tool": "site_energy_summary",
        "method": "recent same-weekday average (last four matching days)",
        "unit": summary.get("unit") or "kWh",
        "history_start": days[0].isoformat(),
        "history_end": latest.isoformat(),
        "observed_days": len(days),
        "coverage_percent": round(100 * coverage, 1),
        "backtest_mae": round(statistics.mean(errors), 2),
        "forecast": forecast,
    }


def format_number(value, precision: int = 2) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{precision}f}"
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
    if not any(term in lowered for term in (
        "standard", "compliance", "compliant", "certif", "iso", "iec", "ieee",
        "protocol", "modbus", "bacnet", "snmp", "accuracy", "calibration", "enpi", "baseline",
    )):
        return []
    wants_compliance = any(term in lowered for term in ("standard", "compliance", "compliant", "certif"))
    wants_protocol = any(term in lowered for term in ("protocol", "modbus", "bacnet", "snmp"))
    wants_management = any(term in lowered for term in ("iso", "enpi", "baseline"))
    wants_accuracy = any(term in lowered for term in ("accuracy", "calibration", "iec 62053"))
    wants_quality = any(term in lowered for term in ("power quality", "sag", "swell", "thd", "ieee", "iec 61000"))
    requested = {
        "Data protocol": wants_protocol,
        "Energy management": wants_management or wants_compliance,
        "Metering accuracy": wants_accuracy or wants_compliance,
        "Power quality context": wants_quality,
    }
    return [item for item in ENERGY_COMPLIANCE_CONTEXT if requested.get(item["label"], False)]


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


def format_display_timestamp(value, timezone_name: str = AI_DISPLAY_TIMEZONE) -> str:
    if not isinstance(value, str):
        return str(value) if value is not None else ""
    if not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value):
        return value
    parsed = parse_datetime(value)
    if not parsed:
        return value
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError):
        zone = timezone(timedelta(hours=8)) if timezone_name == "Asia/Kuala_Lumpur" else timezone.utc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    local = parsed.astimezone(zone)
    suffix = "MYT" if timezone_name == "Asia/Kuala_Lumpur" else local.tzname() or timezone_name
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.day:02d} {local:%b %Y}, {clock} {suffix}"


def format_answer_timestamps(answer: str, timezone_name: str = AI_DISPLAY_TIMEZONE) -> str:
    pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
    )
    return pattern.sub(lambda match: format_display_timestamp(match.group(0), timezone_name), answer)


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


def ranked_consumer_rows(data: dict) -> tuple[list[tuple[dict, float]], list[dict]]:
    measured = []
    unmeasured = []
    for row in first_list(data, ("rankings", "rows", "items", "results", "top_consumers", "consumers", "devices")):
        if not isinstance(row, dict):
            continue
        value = first_value(row, (
            "value", "kwh", "total_kwh", "consumption", "energy",
            "energy_kwh", "total_energy", "total_energy_kwh", "device_energy_kwh",
            "contribution_kwh", "usage_kwh", "consumption_delta",
            "stored_consumption_delta_sum",
        ))
        try:
            amount = float(value)
        except (TypeError, ValueError):
            unmeasured.append(row)
            continue
        if not math.isfinite(amount):
            unmeasured.append(row)
            continue
        factor = {"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0}.get(row.get("unit") or data.get("unit") or "kWh")
        if factor is None or amount < 0:
            unmeasured.append(row)
            continue
        measured.append(({**row, "unit": "kWh"}, amount * factor))
    measured.sort(key=lambda item: item[1], reverse=True)
    return measured, unmeasured


def summarize_top_consumers(data: dict, arguments: dict | None = None, message: str = "") -> str:
    rows = first_list(data, ("rankings", "rows", "items", "results", "top_consumers", "consumers", "devices"))
    measured, unmeasured = ranked_consumer_rows(data)
    unit = data.get("unit") or "kWh"
    time_window = format_time_window(data, arguments)
    lines = []
    if not rows:
        lines.append("No consuming devices were returned for this site and time range.")
    limit = min(max(int((arguments or {}).get("limit") or 10), 1), 100)
    if measured and (wants_lowest_consumer(message) or wants_top_lowest_sum(message)):
        top_row, top_amount = measured[0]
        low_row, low_amount = measured[-1]
        top_name = top_row.get("device_name") or top_row.get("name") or top_row.get("label") or "Unnamed device"
        low_name = low_row.get("device_name") or low_row.get("name") or low_row.get("label") or "Unnamed device"
        lines.append(f"Direct answer for {time_window}:")
        lines.append(f"- Highest energy-consuming device: {top_name} at {format_number(top_amount, 2)} kWh.")
        lines.append(f"- Lowest energy-consuming device: {low_name} at {format_number(low_amount, 2)} kWh.")
        if wants_top_lowest_sum(message):
            lines.append(f"- Combined total of those two devices: {format_number(top_amount + low_amount, 2)} kWh.")
        if len(measured) > 1 and top_amount:
            gap = percentage_difference(top_amount, low_amount)
            if gap is not None:
                lines.append(f"- The lowest device is {format_number(gap, 1)}% below the highest device.")
        lines.append("")
    lines.extend([
        f"Device ranking for this site over the {time_window}:",
        "",
    ])
    for index, (row, amount) in enumerate(measured[:limit], 1):
        name = (
            row.get("device_name")
            or row.get("name")
            or row.get("label")
            or "Unnamed device"
        )
        precision = int(row.get("precision") if isinstance(row.get("precision"), int) else 2)
        value = format_number(amount, precision)
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
    if measured:
        lines.append(f"Showing {min(len(measured), limit)} measured device(s). Ranking is based on total consumption over the {time_window}.")
    if unmeasured:
        lines.append(f"{len(unmeasured)} returned device(s) had no numeric usage value and could not be ranked.")
    return "\n".join(lines)


def summarize_energy_breakdown(
    data: dict,
    arguments: dict | None = None,
    message: str = "",
    inventory_data: dict | None = None,
) -> str:
    """Summarize per-device energy as building totals plus a short device ranking."""
    measured, unmeasured = ranked_consumer_rows(data)
    raw_rows = first_list(data, ("rankings", "rows", "items", "results", "top_consumers", "consumers", "devices"))
    if not measured:
        return "DaxView returned no numeric device energy rows for this site and time range."

    inventory = inventory_data if isinstance(inventory_data, dict) else {}
    inventory_rows = first_list(inventory, ("devices", "rows", "items", "meters"))
    inventory_by_id = {
        str(first_value(row, ("device_id", "id", "meter_id"))): row
        for row in inventory_rows
        if isinstance(row, dict) and first_value(row, ("device_id", "id", "meter_id")) is not None
    }

    located = []
    building_totals: dict[str, dict] = {}
    for row, amount_kwh in measured:
        row = dict(row)
        device_id = first_value(row, ("device_id", "id", "meter_id"))
        device = inventory_by_id.get(str(device_id), {}) if device_id is not None else {}
        building_name = first_value(row, ("building_name", "building", "floor_building_name"))
        if isinstance(building_name, dict):
            building_name = first_value(building_name, ("name", "building_name", "label"))
        if not building_name:
            building_name = first_value(device, ("building_name", "building", "building_label"))
            if isinstance(building_name, dict):
                building_name = first_value(building_name, ("name", "building_name", "label"))
        building_id = first_value(row, ("building_id",)) or first_value(device, ("building_id",))
        row["_building_name"] = str(building_name).strip() if building_name else ""
        row["_device_name"] = first_value(row, ("device_name", "name", "label", "meter_name")) or first_value(device, ("device_name", "name", "meter_name", "label")) or "Unnamed device"
        located.append((row, amount_kwh))
        if row["_building_name"]:
            key = row["_building_name"].casefold()
            group = building_totals.setdefault(key, {"name": row["_building_name"], "values": [], "total": 0.0})
            group["values"].append(amount_kwh)
            group["total"] += amount_kwh
        elif building_id is not None:
            # Keep distinct unnamed buildings separate internally without
            # exposing raw building IDs to the user.
            key = f"unnamed:{building_id}"
            group = building_totals.setdefault(key, {"name": "Building name unavailable", "values": [], "total": 0.0})
            group["values"].append(amount_kwh)
            group["total"] += amount_kwh

    time_window = format_time_window(data, arguments)
    full = wants_full_energy_breakdown(message)
    shown = located if full else located[:5]
    lines = [f"Energy breakdown for {time_window}:"]
    if building_totals:
        lines.append("Building totals (sum of returned device energy rows):")
        for group in sorted(building_totals.values(), key=lambda item: item["total"], reverse=True):
            lines.append(f"- {group['name']}: {format_number(group['total'], 2)} kWh across {len(group['values'])} device row(s).")
    else:
        lines.append("Building totals are unavailable because DaxView and the device list returned no building names for these energy rows.")

    lines.append("All returned devices:" if full else "Top 5 devices:")
    for rank, (row, amount) in enumerate(shown, 1):
        detail = f"{rank}. {row['_device_name']}: {format_number(amount, 2)} kWh"
        if row.get("_building_name"):
            detail += f" — {row['_building_name']}"
        lines.append(detail)
    if not full and len(located) > len(shown):
        lines.append(f"Showing the top {len(shown)} of {len(located)} returned device rows. Ask for the full breakdown to see every returned device.")

    returned_total = sum(amount for _, amount in located)
    lines.append(f"Sum of returned device rows: {format_number(returned_total, 2)} kWh across {len(located)} measured device row(s).")
    top_five = located[:5]
    top_five_sum = sum(amount for _, amount in top_five)
    equation = " + ".join(format_number(amount, 2) for _, amount in top_five)
    lines.append(f"Calculation: top-device subtotal = {equation} = {format_number(top_five_sum, 2)} kWh.")
    lines.append("Building total formula: add the kWh values for all returned devices assigned to that building, over the same time range.")

    reported_count = first_value(data, ("total_count", "device_count", "row_count", "count"))
    if reported_count is not None:
        try:
            if int(reported_count) > len(raw_rows):
                lines.append(f"Data coverage: DaxView returned {len(raw_rows)} of {int(reported_count)} device rows; building totals use only the returned rows.")
        except (TypeError, ValueError):
            pass
    if unmeasured:
        lines.append(f"{len(unmeasured)} returned device row(s) had no usable numeric energy value and were excluded from totals and ranking.")
    coverage_note = format_coverage_note(data)
    if coverage_note:
        lines.append(coverage_note)
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


def requested_explicit_date_range(message: str, fallback_year: int) -> tuple[date, date] | None:
    text = message.lower().replace("\u2011", "-")
    month_pattern = r"january|february|march|april|may|june|july|august|september|october|november|december"
    day_range = re.search(
        rf"\b(?:from\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:-|\u2013|\u2014|to|through|until)\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})(?:\s+(\d{{4}}))?\b",
        text,
    )
    if day_range:
        first_day, last_day = int(day_range.group(1)), int(day_range.group(2))
        month = MONTH_NAMES[day_range.group(3)]
        year = int(day_range.group(4) or fallback_year)
        try:
            first = date(year, month, first_day)
            last = date(year, month, last_day)
            return (first, last) if first <= last else None
        except ValueError:
            return None

    full_range = re.search(
        rf"\b(?:from\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})(?:\s+(\d{{4}}))?\s+"
        rf"(?:-|\u2013|\u2014|to|through|until)\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})(?:\s+(\d{{4}}))?\b",
        text,
    )
    if not full_range:
        return None
    first_day, first_month, first_year = int(full_range.group(1)), MONTH_NAMES[full_range.group(2)], int(full_range.group(3) or fallback_year)
    last_day, last_month = int(full_range.group(4)), MONTH_NAMES[full_range.group(5)]
    last_year = int(full_range.group(6) or first_year)
    try:
        first, last = date(first_year, first_month, first_day), date(last_year, last_month, last_day)
        return (first, last) if first <= last else None
    except ValueError:
        return None


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
    explicit_range = requested_explicit_date_range(message, fallback_year)
    requested_dates = requested_comparison_dates(message, fallback_year)
    if explicit_range:
        first_day, last_day = explicit_range
        requested_dates = [
            (first_day + timedelta(days=offset)).isoformat()
            for offset in range((last_day - first_day).days + 1)
        ]
    requested_months = [] if requested_dates else requested_comparison_months(message, fallback_year)
    requested_date_set = set(requested_dates)
    has_specific_dates = bool(requested_date_set)
    has_specific_months = bool(requested_months)
    include_range_summary = wants_relative_summary(message)
    total_only = wants_sitewide_energy_total(message)
    lines = ["Overall site energy for the requested period:" if total_only else "Site energy summary:"]
    summary_keys = ("total", "total_value", "value", "total_kwh", "consumption", "energy")
    total_value = next((data.get(key) for key in summary_keys if data.get(key) is not None), None)
    if total_value is not None and not has_specific_dates and not has_specific_months:
        title = "Total for the requested range" if total_only else "Total"
        lines.append(f"{title}: {format_energy_value(total_value, conversion_factor, display_unit, precision)}.")
    buckets = data.get("buckets") or data.get("rows") or data.get("series") or data.get("data")
    daily_values = {}
    daily_kwh_values = {}
    complete_window = completed_daily_range(message) is not None and bool(arguments and arguments.get("bucket") == "day")
    if isinstance(buckets, list) and buckets:
        if total_only:
            pass
        elif has_specific_months:
            lines.append("Requested monthly values:")
        else:
            lines.append("Daily values (complete local days):" if complete_window else "Requested daily values:" if has_specific_dates else "Daily values:")
        shown_rows = 0
        for item in buckets:
            if not isinstance(item, dict):
                continue
            timestamp = item.get("timestamp") or item.get("bucket") or item.get("start") or item.get("date")
            label = local_bucket_date(timestamp, timezone_name) if isinstance(timestamp, str) else None
            label = label or timestamp or "period"
            value = item.get("value") if item.get("value") is not None else item.get("kwh")
            numeric_value = chart_number(value)
            if numeric_value is not None:
                bucket_unit = str(item.get("unit") or unit).lower()
                to_kwh = {"wh": 0.001, "kwh": 1.0, "mwh": 1000.0, "gwh": 1000000.0}.get(bucket_unit)
                if to_kwh is not None:
                    daily_kwh_values[str(label)] = daily_kwh_values.get(str(label), 0.0) + numeric_value * to_kwh
            if complete_window and arguments:
                start_day = parse_datetime(arguments.get("start"))
                end_day = parse_datetime(arguments.get("end"))
                if start_day and end_day:
                    first_day = start_day.astimezone(timezone(timedelta(hours=8))).date().isoformat()
                    last_day = end_day.astimezone(timezone(timedelta(hours=8))).date().isoformat()
                    if not first_day <= str(label) < last_day:
                        continue
            if numeric_value is not None:
                daily_values[str(label)] = numeric_value
            if has_specific_months:
                continue
            if has_specific_dates and not include_range_summary and str(label) not in requested_date_set:
                continue
            if shown_rows < 31 and not total_only:
                row_unit = item.get("unit") or unit
                row_factor, row_display_unit, row_precision = requested_energy_unit(message, row_unit)
                lines.append(f"- {label}: {format_energy_value(value, row_factor, row_display_unit, row_precision if row_factor != 1.0 else precision)}")
                shown_rows += 1
        if not has_specific_months and len(daily_values) > 31:
            lines.append(f"Showing 31 of {len(daily_values)} daily row(s).")
    elif total_value is None:
        lines.append("No energy values were returned for this site and time range.")
    if total_only and total_value is None and daily_kwh_values:
        total_kwh = sum(daily_kwh_values.values())
        total_factor, total_unit, total_precision = requested_energy_unit(message, "kWh")
        lines.append(
            f"Total across returned readings: "
            f"{format_energy_value(total_kwh, total_factor, total_unit, total_precision)}."
        )
    if data.get("complete") is False:
        lines.append("Completeness: the MCP marked this requested energy range incomplete; treat totals as partial evidence.")
    wants_highest, wants_lowest = requested_energy_extrema(message)
    if wants_highest or wants_lowest:
        complete_values = dict(daily_values)
        requested_start = parse_datetime((arguments or {}).get("start"))
        requested_end = parse_datetime((arguments or {}).get("end"))
        local_zone = timezone(timedelta(hours=8)) if timezone_name == "Asia/Kuala_Lumpur" else timezone.utc
        if requested_start and requested_end:
            local_start = requested_start.astimezone(local_zone)
            local_end = requested_end.astimezone(local_zone)
            first_complete_day = local_start.date() + timedelta(days=bool(
                local_start.hour or local_start.minute or local_start.second or local_start.microsecond
            ))
            complete_values = {
                label: value for label, value in complete_values.items()
                if first_complete_day.isoformat() <= label < local_end.date().isoformat()
            }
        if complete_values:
            for wanted, title, extreme in (
                (wants_highest, "Highest daily energy", max),
                (wants_lowest, "Lowest daily energy", min),
            ):
                if wanted:
                    amount = extreme(complete_values.values())
                    days = ", ".join(sorted(day for day, value in complete_values.items() if value == amount))
                    lines.append(f"{title}: {days} at {format_energy_value(amount, conversion_factor, display_unit, precision)}.")
            if wants_highest and wants_lowest and any(term in message.lower() for term in ("percent", "%", "difference", "compare")):
                highest, lowest = max(complete_values.values()), min(complete_values.values())
                change = percentage_difference(highest, lowest)
                lines.append(f"Highest minus lowest: {format_energy_value(highest - lowest, conversion_factor, display_unit, precision)}.")
                lines.append(f"The highest day was {change:.2f}% above the lowest day (reference: lowest day)." if change is not None else "Percentage difference is unavailable because the lowest-day reference is zero.")
        else:
            lines.append("Highest and lowest daily energy are unavailable because no complete daily readings were returned for this period.")
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
    compliance_note = "" if total_only else format_compliance_context(build_compliance_context(message, ["site_energy_summary"]))
    if compliance_note:
        lines.append("")
        lines.append(compliance_note)
    return "\n".join(lines)


def summarize_alarm_frequency(data: dict, arguments: dict | None = None, message: str = "") -> str:
    rows = data.get("rows") or data.get("alarms") or data.get("items")
    rows = rows if isinstance(rows, list) else []
    latest_requested = bool(re.search(r"\b(?:latest|most recent)\b", message, re.IGNORECASE))
    lines = [
        "Latest alarm occurrences:"
        if latest_requested
        else "Alarm frequency summary:"
    ]
    if not rows:
        lines.append("No alarm rows were returned for this site and time range.")
    limit = min(max(int((arguments or {}).get("limit") or 10), 1), 20)
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if latest_requested:
        ranked_rows = sorted(
            valid_rows,
            key=lambda row: parse_datetime(
                row.get("latest_occurrence") or row.get("last_seen") or row.get("latest_at")
            ) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
    else:
        ranked_rows = sorted(
            valid_rows,
            key=lambda row: chart_number(row.get("event_count", row.get("count", row.get("frequency", 0)))) or 0,
            reverse=True,
        )
    for index, row in enumerate(ranked_rows[:limit], 1):
        if not isinstance(row, dict):
            continue
        name = row.get("alarm_name") or row.get("name") or row.get("type") or row.get("severity") or "Alarm"
        count = next((row[key] for key in ("event_count", "count", "frequency", "total", "value") if row.get(key) is not None), None)
        detail = f"{index}. {name} - {format_number(count, 0)} occurrence(s)"
        device_name = row.get("device_name")
        if device_name:
            detail += f"; device {device_name}"
        if row.get("severity"):
            detail += f", severity {row['severity']}"
        first_seen = row.get("first_seen")
        latest = row.get("latest_occurrence") or row.get("last_seen") or row.get("latest_at")
        if first_seen and latest and first_seen != latest:
            detail += f"; first {first_seen}; latest {latest}"
        elif latest:
            detail += f"; latest {latest}"
        lifecycle = [
            f"{row[key]} {label}" for key, label in (
                ("active_count", "active"), ("acknowledged_count", "acknowledged"),
                ("resolved_count", "resolved"), ("closed_count", "closed"),
            ) if isinstance(row.get(key), (int, float)) and row[key] > 0
        ]
        if lifecycle:
            detail += "; " + ", ".join(lifecycle)
        lines.append(detail)
    if data.get("row_count") is not None:
        lines.append(f"Rows returned: {data.get('row_count')}.")
    return "\n".join(lines)


def summarize_active_alarms(data: dict) -> str:
    site = data.get("site") if isinstance(data.get("site"), dict) else {}
    alarms = data.get("alarms") if isinstance(data.get("alarms"), list) else []
    lines = []
    if site.get("name"):
        lines.append(f"Site: {site.get('name')} (ID {site.get('id', 'unknown')})")
    lines.append(
        "Summary: "
        f"{format_number(data.get('active_count'), 0)} active alarm(s), "
        f"{format_number(data.get('critical_count'), 0)} critical, "
        f"{format_number(data.get('warning_count'), 0)} warning"
    )
    if alarms:
        lines.append("Latest active alarms:")
        for index, alarm in enumerate(alarms[:8], 1):
            if not isinstance(alarm, dict):
                continue
            lines.append(
                f"- {index}. {alarm.get('alarm_name', 'Alarm')} on "
                f"{alarm.get('device_name') or 'unknown device'} "
                f"- {alarm.get('severity', 'unknown')} since {alarm.get('started_at') or 'unknown time'}"
            )
        if len(alarms) > 8:
            lines.append(f"Showing 8 of {len(alarms)} alarm(s).")
    return "\n".join(lines)


def summarize_site_metadata(data: dict) -> str:
    site = data.get("site") if isinstance(data.get("site"), dict) else data
    buildings = data.get("buildings") if isinstance(data.get("buildings"), list) else []
    lines = []
    site_name = site.get("name") or site.get("site_name")
    if site_name:
        lines.append(f"Site: {site_name} (ID {site.get('id', site.get('site_id', 'unknown'))})")
    details = []
    detail_labels = {
        "timezone": "Timezone",
        "country": "Country",
        "currency": "Currency",
        "address": "Address",
        "market": "Market",
        "building_count": "Buildings",
        "device_count": "Devices",
        "meter_count": "Meters",
    }
    for key in ("timezone", "country", "currency", "address", "market"):
        if site.get(key):
            details.append(f"{detail_labels[key]}: {site.get(key)}")
    for key in ("building_count", "device_count", "meter_count"):
        if data.get(key) is not None:
            details.append(f"{detail_labels[key]}: {data.get(key)}")
    if details:
        lines.append("Key details:")
        for detail in details:
            lines.append(f"- {detail}")
    if buildings:
        lines.append("Buildings:")
        for index, building in enumerate(buildings[:10], 1):
            if isinstance(building, dict):
                building_details = []
                for key, label in (("device_count", "devices"), ("meter_count", "meters"), ("floor_count", "floors")):
                    if building.get(key) is not None:
                        building_details.append(f"{building.get(key)} {label}")
                detail_suffix = f" - {', '.join(building_details)}" if building_details else ""
                lines.append(
                    f"- {index}. {building.get('name') or building.get('building_name')} "
                    f"(ID {building.get('id', building.get('building_id', 'unknown'))}){detail_suffix}"
                )
    return "\n".join(lines)


def summarize_site_devices(data: dict, message: str = "") -> str:
    devices = first_list(data, ("devices", "rows", "items", "meters"))
    lines = []
    total = data.get("device_count") or data.get("meter_count") or data.get("row_count") or len(devices)
    if data.get("site_name") or data.get("building_name"):
        scope = data.get("building_name") or data.get("site_name")
        lines.append(f"Scope: {scope}")
    if total:
        lines.append(f"{total} device(s) found.")
    if not devices:
        lines.append("No devices were returned.")
        return "\n".join(lines)

    lowered = message.lower()
    wants_status = bool(re.search(r"\b(?:status|online|offline|stale|reporting|last seen)\b", lowered))
    wants_parameters = bool(re.search(r"\b(?:parameter|parameters|metric|metrics|capabilities|supported data|available data|what data)\b", lowered))
    status_counts: dict[str, int] = {}
    for device in devices:
        if isinstance(device, dict):
            status = str(device.get("status") or device.get("connection_status") or device.get("state") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts and wants_status:
        lines.append("Status counts:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status.title()}: {count}")

    shown = min(len(devices), 20)
    lines.append("Devices:")
    for index, device in enumerate(devices[:shown], 1):
        if not isinstance(device, dict):
            lines.append(f"- {index}. {device}")
            continue
        status = str(device.get("status") or device.get("connection_status") or device.get("state") or "unknown")
        name = device.get("device_name") or device.get("name") or device.get("meter_name") or "Unnamed device"
        last_seen = device.get("last_updated") or device.get("last_seen") or device.get("last_telemetry_at")
        metrics = first_value(device, (
            "supported_metrics", "metrics", "metric_names", "telemetry_metrics",
            "parameters", "parameter_names", "available_parameters", "capabilities",
        ))
        detail = f" — {status.title()}" if wants_status else ""
        if wants_status and last_seen and "last seen" in lowered:
            detail += f", last seen {format_display_timestamp(str(last_seen), AI_DISPLAY_TIMEZONE)}"
        lines.append(f"- {name}{detail}")
        if wants_parameters and metrics:
            if isinstance(metrics, list):
                metric_text = ", ".join(str(item) for item in metrics[:12])
                if len(metrics) > 12:
                    metric_text += f", +{len(metrics) - 12} more"
            else:
                metric_text = str(metrics)
            lines.append(f"  Available parameters: {metric_text}")

    if len(devices) > shown:
        lines.append(f"Showing {shown} of {len(devices)} device(s).")
    return "\n".join(lines)


def summarize_meter_status(data: dict) -> str:
    rows = first_list(data, ("rows", "devices", "items", "meters"))
    lines = ["Meter communication status:"]
    counts = []
    for key, label in (("online_count", "online"), ("offline_count", "offline"), ("stale_count", "stale"), ("unknown_count", "unknown")):
        if data.get(key) is not None:
            counts.append(f"{format_number(data.get(key), 0)} {label}")
    if counts:
        lines.append("- " + ", ".join(counts))
    if not rows:
        return "\n".join(lines)

    offline_or_unknown = [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("status") or row.get("data_status") or "unknown").lower() in {"offline", "unknown", "stale"}
    ]
    focus_rows = offline_or_unknown[:8] or rows[:8]
    lines.append("Devices needing attention:")
    for index, row in enumerate(focus_rows, 1):
        if not isinstance(row, dict):
            lines.append(f"- {index}. {row}")
            continue
        name = row.get("device_name") or row.get("name") or "Unnamed device"
        status = row.get("status") or row.get("data_status") or "unknown"
        last_seen = row.get("last_seen") or row.get("last_updated") or row.get("last_telemetry_at")
        last_seen_text = f", last seen {last_seen}" if last_seen else ""
        lines.append(f"- {index}. {name} - {status}{last_seen_text}")
    if len(rows) > len(focus_rows):
        lines.append(f"Showing {len(focus_rows)} of {len(rows)} device(s).")
    return "\n".join(lines)


def summarize_generic_tool(operation_id: str, data: dict) -> str:
    title = operation_id.replace("_", " ").title()
    lines = [f"{title}:"]
    if not data:
        lines.append("No structured data was returned.")
        return "\n".join(lines)

    scalar_keys = (
        "status",
        "site_name",
        "building_name",
        "device_name",
        "metric",
        "unit",
        "total",
        "count",
        "row_count",
        "active_count",
        "online_count",
        "offline_count",
        "stale_count",
        "coverage_percent",
        "peak_kw",
        "peak_time",
        "total_cost",
        "currency",
        "confidence",
        "method",
    )
    for key in scalar_keys:
        value = data.get(key)
        if value is not None:
            lines.append(f"{key}: {value}")

    rows = first_list(
        data,
        (
            "rows",
            "items",
            "devices",
            "metrics",
            "available_metrics",
            "readings",
            "alarms",
            "series",
            "forecast",
            "anomalies",
            "missing_intervals",
            "buildings",
            "sections",
        ),
    )
    if rows:
        lines.append("Rows:")
        preferred_row_keys = (
            "device_name",
            "name",
            "metric",
            "canonical_metric",
            "source_metric",
            "value",
            "reading",
            "latest_value",
            "latest",
            "kwh",
            "total_kwh",
            "energy_kwh",
            "kw",
            "demand_kw",
            "peak_kw",
            "current",
            "current_a",
            "voltage",
            "voltage_v",
            "unit",
            "reading_unit",
            "timestamp",
            "time",
            "last_seen",
            "last_timestamp",
            "sample_count",
            "coverage_percent",
        )
        for index, row in enumerate(rows[:10], 1):
            if not isinstance(row, dict):
                lines.append(f"{index}. {row}")
                continue
            parts = []
            used_keys = set()
            for key in preferred_row_keys:
                value = row.get(key)
                if value is None or isinstance(value, (dict, list)):
                    continue
                parts.append(f"{key}={value}")
                used_keys.add(key)
                if len(parts) >= 10:
                    break
            if len(parts) < 10:
                for key, value in row.items():
                    if key in used_keys or key.lower() in {"device_id", "meter_id", "deviceid", "meterid", "site_id", "building_id", "id"} or value is None or isinstance(value, (dict, list)):
                        continue
                    parts.append(f"{key}={value}")
                    if len(parts) >= 10:
                        break
            lines.append(f"{index}. " + ", ".join(parts))
        if len(rows) > 10:
            lines.append(f"Showing 10 of {len(rows)} row(s).")

    coverage_note = format_coverage_note(data)
    if coverage_note:
        lines.append(coverage_note)
    return "\n".join(lines)


def summarize_metric_ranking(data: dict, message: str, arguments: dict | None = None) -> str:
    rows = data.get("ranked_devices") if isinstance(data.get("ranked_devices"), list) else []
    period = format_time_window(data, arguments)
    lines = [f"Frequency ranking by highest recorded value ({period}):"]
    if not rows:
        lines.append("No devices returned numeric frequency readings with a recognized Hz or kHz unit.")
    show_all = bool(re.search(r"\b(?:all|every)\b", message, re.IGNORECASE))
    limit = len(rows) if show_all else requested_top_limit(message) or 10
    for rank, row in enumerate(rows[:limit], 1):
        lines.append(
            f"{rank}. {row.get('device_name') or 'Unnamed device'}: "
            f"{format_number(row.get('value'), 4)} Hz"
            + (f"; highest at {row['peak_time']}" if row.get("peak_time") else "")
            + (f"; {row['status']}" if row.get("status") else "")
        )
    if rows:
        lines.append(f"Ranked {min(len(rows), limit)} of {len(rows)} device(s) with usable readings, from {data.get('device_count', len(rows))} devices scanned. Values are ranked by the highest measured frequency in the period.")
    excluded = data.get("excluded_devices") if isinstance(data.get("excluded_devices"), list) else []
    if excluded:
        reasons = {}
        for row in excluded:
            reason = str(row.get("reason") or "Readings unavailable")
            reasons[reason] = reasons.get(reason, 0) + 1
        reason_summary = "; ".join(f"{count} {reason.lower()}" for reason, count in reasons.items())
        lines.append(f"{len(excluded)} device(s) were excluded: {reason_summary}.")
        for row in excluded[:5]:
            name = row.get("device_name") or "Unnamed device"
            lines.append(f"- {name}: {row.get('reason') or 'readings unavailable'}.")
    if data.get("truncated_device_scan"):
        lines.append("The device list was truncated, so this may not include every device in the site.")
    if data.get("truncated_readings"):
        lines.append("Some readings were truncated, so a device's highest value may be understated.")
    return "\n".join(lines)


def summarize_latest_telemetry(data: dict, arguments: dict | None = None) -> str:
    requested_metrics = (arguments or {}).get("metrics")
    metric = str((requested_metrics[0] if isinstance(requested_metrics, list) and requested_metrics else None) or (arguments or {}).get("metric") or "requested metric")
    device_id = (arguments or {}).get("device_id")
    series_rows = first_list(data, ("series", "rows", "items", "readings", "results"))
    candidates = []
    for series in series_rows:
        if not isinstance(series, dict):
            continue
        series_metric = str(first_value(series, ("metric", "source_metric", "name")) or metric)
        points = series.get("points") if isinstance(series.get("points"), list) else [series]
        for point in points:
            if not isinstance(point, dict):
                continue
            value = chart_value(point, ("value", "reading", "last", "latest", "frequency", "frequency_hz"))
            if value is None:
                continue
            timestamp = first_value(point, ("timestamp", "time", "datetime", "recorded_at", "last_seen"))
            parsed = parse_datetime(timestamp) if timestamp else None
            candidates.append((parsed or datetime.min.replace(tzinfo=timezone.utc), value, timestamp, series_metric, point.get("unit") or series.get("unit")))
    if not candidates:
        device_name = (arguments or {}).get("_device_name")
        scope = f" for {device_name}" if device_name else ""
        return f"DaxView returned no {metric} readings{scope} in the latest telemetry result."
    _, value, timestamp, series_metric, unit = max(candidates, key=lambda item: item[0])
    unit = unit or data.get("unit")
    label = f"{series_metric}{f' ({unit})' if unit else ''}"
    device_name = (arguments or {}).get("_device_name")
    scope = f" for {device_name}" if device_name else ""
    return f"Latest {label} reading{scope}: {format_number(value, 4)}{f' {unit}' if unit else ''}" + (f", recorded at {format_display_timestamp(timestamp)}." if timestamp else "; timestamp not returned.")


def summarize_demand_peak(data: dict, message: str, arguments: dict | None = None) -> str:
    ranked_devices = data.get("ranked_devices") if isinstance(data.get("ranked_devices"), list) else []
    if ranked_devices or "device_count" in data:
        lines = [f"Highest demand devices ({format_time_window(data, arguments)}):"]
        top_peak = ranked_devices[0].get("peak_kw") if ranked_devices else None
        if not ranked_devices:
            lines.append("No devices returned a comparable, unit-qualified demand or active-power reading for this period.")
        for index, row in enumerate(ranked_devices[:requested_top_limit(message) or 5], 1):
            diff = ""
            if index > 1 and top_peak not in (None, 0):
                pct = percentage_difference(top_peak, row.get("peak_kw"))
                if pct is not None:
                    diff = f" ({format_number(pct, 1)}% below top)"
            status = f", {row.get('status')}" if row.get("status") else ""
            lines.append(
                f"{index}. {row.get('device_name') or 'Unnamed device'}: "
                f"{format_number(row.get('peak_kw'), 2)} kW{diff}{status}"
                + (f", peak at {row['peak_time']}" if row.get("peak_time") else "")
                + (f", metric {row['metric']}" if row.get("metric") else "")
            )
        if data.get("device_count") is not None:
            lines.append(
                f"Scanned {data.get('device_count')} device(s); compared {data.get('eligible_device_count', len(ranked_devices))} with usable demand/power readings."
            )
        excluded = data.get("excluded_devices") if isinstance(data.get("excluded_devices"), list) else []
        if excluded:
            lines.append(f"Excluded {len(excluded)} device(s) with unsupported metrics, missing readings, or unusable units.")
            for row in excluded[:5]:
                name = row.get("device_name") or "Unnamed device"
                lines.append(f"- {name}: {row.get('reason') or 'not eligible'}.")
        if data.get("truncated_device_scan"):
            lines.append("The device inventory was truncated; this is not a complete site-wide ranking.")
        if data.get("truncated_readings"):
            lines.append("At least one device's returned readings were truncated; its peak may be understated.")
        return "\n".join(lines)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
    rows = first_list(data, ("rows", "items", "results", "devices"))
    source = summary
    peak_keys = ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw")
    if first_value(source, peak_keys) is None:
        source = next((row for row in rows if isinstance(row, dict) and first_value(row, peak_keys) is not None), summary)
    device_id = first_value(source, ("device_id", "meter_id")) or (arguments or {}).get("device_id")
    device_name = first_value(source, ("device_name", "meter_name", "name"))
    device_name = device_name or (arguments or {}).get("_device_name")
    heading = f"Max demand for {device_name}" if device_name else "Max demand"
    lines = [f"{heading} (historical, {format_time_window(data, arguments)}):"]
    peak = first_value(source, peak_keys)
    if peak is None:
        lines.append("Peak demand: not returned by DaxView for this device and period.")
    else:
        lines.append(f"Peak demand: {format_number(peak, 2)} kW")
        peak_time = first_value(source, ("peak_time", "peak_at", "timestamp", "max_time", "max_at"))
        if peak_time:
            lines.append(f"Peak time: {peak_time}")
    for keys, label, unit in (
        (("average_kw", "avg_kw"), "Average demand", "kW"),
        (("minimum_kw", "min_kw"), "Minimum demand", "kW"),
        (("energy_kwh", "total_kwh"), "Energy in period", "kWh"),
        (("coverage_percent",), "Data coverage", "%"),
    ):
        value = first_value(source, keys)
        if value is not None:
            lines.append(f"{label}: {format_number(value, 2)} {unit}")
    latest = first_value(source, ("latest_kw", "latest_demand_kw"))
    latest_time = first_value(source, ("latest_at", "latest_timestamp", "last_seen"))
    if latest is not None:
        lines.append(f"Latest returned demand: {format_number(latest, 2)} kW" + (f" at {latest_time}" if latest_time else " (timestamp unavailable)"))
    if source.get("source") == "manual calculation from telemetry_timeseries":
        sample_count = source.get("sample_count")
        lines.append(f"Calculation source: manual max from demand telemetry" + (f" ({sample_count} samples)." if sample_count else "."))
    if any(term in message.lower() for term in ("live", "real-time", "realtime", "current value")):
        lines.append("Live value: not provided by this historical demand tool; the latest returned value is not verified live.")
    return "\n".join(lines)


def summarize_energy_forecast(data: dict) -> str:
    rows = data.get("forecast") if isinstance(data.get("forecast"), list) else []
    if not rows:
        return "A forecast could not be calculated from the available daily energy readings."
    unit = data.get("unit") or "kWh"
    lines = [f"Daily energy forecast ({unit}):"]
    for row in rows:
        if isinstance(row, dict):
            lines.append(f"- {row.get('date')}: {format_number(row.get('value'), 2)} {unit}")
    lines.append(
        f"Method: {data.get('method')}. Based on {data.get('observed_days')} complete days "
        f"({data.get('coverage_percent')}% coverage), through {data.get('history_end')}."
    )
    if data.get("backtest_mae") is not None:
        lines.append(f"Recent 7-day backtest mean absolute error: {data['backtest_mae']} {unit}.")
    lines.append("This is a historical-pattern estimate, not a guaranteed future reading.")
    return "\n".join(lines)


def telemetry_series_points(data: dict) -> list[dict]:
    """Return time-sorted, independently labeled telemetry series."""
    output = []
    for index, source in enumerate(first_list(data, ("series", "rows", "buckets", "data")), 1):
        if not isinstance(source, dict):
            continue
        points = source.get("points") if isinstance(source.get("points"), list) else [source]
        normalized = []
        for point in points:
            if not isinstance(point, dict):
                continue
            stamp = first_value(point, ("timestamp", "date", "bucket", "time"))
            value = chart_number(first_value(point, ("value", "average", "avg", "reading")))
            parsed = parse_datetime(stamp) if isinstance(stamp, str) else None
            if parsed is None or value is None:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            key = parsed.astimezone(timezone.utc).isoformat()
            normalized.append((key, value))
        normalized.sort(key=lambda item: item[0])
        if not normalized:
            continue
        name = first_value(source, ("phase", "source_metric", "metric", "name")) or f"Series {index}"
        output.append({
            "name": str(name),
            "unit": str(source.get("unit") or data.get("unit") or ""),
            "aggregation": str(source.get("aggregation") or ""),
            "points": normalized,
        })
    return output


def format_chart_timestamp(value: str, timezone_name: str = AI_DISPLAY_TIMEZONE) -> str:
    parsed = parse_datetime(value)
    if not parsed:
        return value
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError):
        zone = timezone(timedelta(hours=8)) if timezone_name == "Asia/Kuala_Lumpur" else timezone.utc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(zone)
    # Short category labels stay legible on dense hourly charts. The chart's
    # summary gives the full date range and timezone for context.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{local.day} {local:%b %Y}"
    hour = local.strftime("%I %p").lstrip("0")
    return f"{local.day} {local:%b}, {hour}"


def summarize_telemetry_timeseries(
    data: dict,
    arguments: dict | None = None,
) -> str:
    series = telemetry_series_points(data)
    if not series:
        return "DaxView returned no numeric telemetry points for this device and time range."

    arguments = arguments or {}
    timezone_name = arguments.get("timezone") or AI_DISPLAY_TIMEZONE
    device = data.get("device") if isinstance(data.get("device"), dict) else {}
    device_name = first_value(device, ("device_name", "name", "label")) or arguments.get("_device_name")
    metric = str(arguments.get("metric") or first_value(series[0], ("metric",)) or "Telemetry").replace("_", " ").title()
    all_points = [point for item in series for point in item["points"]]
    first_stamp = min(stamp for stamp, _ in all_points)
    last_stamp = max(stamp for stamp, _ in all_points)
    period = f"{format_display_timestamp(first_stamp, timezone_name)} to {format_display_timestamp(last_stamp, timezone_name)}"
    lines = [f"{metric} trend{f' for {device_name}' if device_name else ''}:", f"Period: {period}."]

    overall_values = []
    aggregation_labels = set()
    for item in series:
        values = [value for _, value in item["points"]]
        overall_values.extend(values)
        if item["aggregation"]:
            aggregation_labels.add(item["aggregation"].lower())
        average = sum(values) / len(values)
        first_value_reading = values[0]
        last_value_reading = values[-1]
        delta = last_value_reading - first_value_reading
        sign = "+" if delta > 0 else ""
        unit = f" {item['unit']}" if item["unit"] else ""
        lines.append(
            f"- {item['name']}: average {format_number(average, 2)}{unit}; "
            f"range {format_number(min(values), 2)} to {format_number(max(values), 2)}{unit}; "
            f"first-to-last {format_number(first_value_reading, 2)} to "
            f"{format_number(last_value_reading, 2)}{unit} ({sign}{format_number(delta, 2)}{unit})."
        )

    unit = next((item["unit"] for item in series if item["unit"]), "")
    unit_text = f" {unit}" if unit else ""
    lines.append(
        f"Summary: {len(series)} plotted series with {sum(len(item['points']) for item in series):,} "
        f"returned point(s); values ranged from {format_number(min(overall_values), 2)} to "
        f"{format_number(max(overall_values), 2)}{unit_text}."
    )
    if aggregation_labels:
        labels = ", ".join(sorted(aggregation_labels))
        bucket = str(arguments.get("bucket") or "").lower()
        bucket_label = {"1h": "hourly", "1d": "daily", "15m": "15-minute", "raw": "raw"}.get(bucket, bucket)
        if bucket_label:
            lines.append(f"Sampling: {bucket_label} buckets with {labels} aggregation.")
        else:
            lines.append(f"Reading aggregation returned by DaxView: {labels}.")
    return "\n".join(lines)


def summarize_historical_answer(
    message: str,
    operation_id: str,
    mcp_result: dict,
    request_id: str,
    arguments: dict | None = None,
) -> str:
    issue = result_problem(mcp_structured_result(mcp_result)) or result_problem(mcp_result)
    if issue:
        metric = (arguments or {}).get("metric") or ("demand" if operation_id == "demand_peak_summary" else "energy")
        scope = "device" if (arguments or {}).get("device_id") else "building" if (arguments or {}).get("building_id") else "site"
        code = str(issue).upper()
        if "NO_DATA" in code:
            return f"DaxView has no {metric} data for this {scope} in {format_time_window({}, arguments)}."
        if "UNAVAILABLE" in code:
            return f"DaxView historical data is currently unavailable for this {scope}."
        return f"DaxView could not return {operation_id}: {issue}."
    data = historical_result_data(mcp_result)
    if operation_id == "demand_peak_summary" and not any(
        data.get(key) is not None for key in ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw")
    ):
        calculated = calculate_demand_peak_from_telemetry(mcp_result, arguments or {})
        if calculated and isinstance(calculated.get("data"), dict):
            data = {**data, **calculated["data"]}
    if operation_id == "device_energy_breakdown" and wants_energy_breakdown_request(message):
        return summarize_energy_breakdown(data, arguments, message)
    if operation_id in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"} and wants_device_usage_ranking(message):
        return summarize_top_consumers(data, arguments, message)
    if operation_id == "telemetry_metric_ranking":
        return summarize_metric_ranking(data, message, arguments)
    if operation_id == "latest_telemetry_snapshot":
        return summarize_latest_telemetry(data, arguments)
    if operation_id == "telemetry_timeseries":
        return summarize_telemetry_timeseries(data, arguments)
    if operation_id == "site_energy_summary":
        return summarize_site_energy(data, message, arguments)
    if operation_id == "alarm_frequency_summary":
        return summarize_alarm_frequency(data, arguments, message)
    if operation_id == "energy_forecast":
        return summarize_energy_forecast(data)
    if operation_id == "active_alarm_summary":
        return summarize_active_alarms(data)
    if operation_id == "site_metadata_summary":
        return summarize_site_metadata(data)
    if operation_id == "site_device_list":
        return summarize_site_devices(data, message)
    if operation_id in {"meter_status_summary", "site_device_list"}:
        return summarize_meter_status(data)
    if operation_id == "demand_peak_summary":
        return summarize_demand_peak(data, message, arguments)
    # MCP tool results have already supplied the facts. Keep a deterministic
    # summary as the fallback for any internal operation without a specialized
    # formatter; the Ollama call is optional answer polishing, not a dependency.
    return summarize_generic_tool(operation_id, data)


def summarize_result_takeaway(results: list[dict]) -> str:
    """Add a short, data-grounded overview for a multi-part site answer."""
    by_operation = {
        item.get("operation_id"): item
        for item in results
        if isinstance(item, dict) and item.get("operation_id")
    }
    lines = []
    energy_item = by_operation.get("site_energy_summary")
    energy_data = historical_result_data(energy_item.get("result") or {}) if energy_item else {}
    energy_days = []
    if energy_data:
        buckets = first_list(energy_data, ("buckets", "rows", "series", "data"))
        for row in buckets:
            if not isinstance(row, dict):
                continue
            stamp = first_value(row, ("timestamp", "bucket", "start", "date"))
            value = chart_value(row, ("value", "kwh", "energy", "total_kwh"))
            label = local_bucket_date(stamp, "Asia/Kuala_Lumpur") if isinstance(stamp, str) else None
            if label and value is not None:
                energy_days.append((label, value))
    energy_days.sort()
    if len(energy_days) >= 2:
        low_day, low_value = min(energy_days, key=lambda item: item[1])
        high_day, high_value = max(energy_days, key=lambda item: item[1])
        latest_day, latest_value = energy_days[-1]
        previous_day, previous_value = energy_days[-2]
        change = percentage_difference(latest_value, previous_value)
        change_text = (
            f" The latest returned day ({latest_day}) is {format_number(abs(change), 1)}% "
            f"{'above' if latest_value > previous_value else 'below' if latest_value < previous_value else 'the same as'} {previous_day}."
            if change is not None else ""
        )
        lines.append(
            f"- Energy: highest daily use was {format_energy_value(high_value, 1.0, 'kWh', 2)} on {high_day}; "
            f"lowest was {format_energy_value(low_value, 1.0, 'kWh', 2)} on {low_day}.{change_text}"
        )

    alarm_item = by_operation.get("active_alarm_summary")
    alarm_data = historical_result_data(alarm_item.get("result") or {}) if alarm_item else {}
    has_offline_alarm = False
    if alarm_data:
        active = first_value(alarm_data, ("active_count", "total_active"))
        critical = first_value(alarm_data, ("critical_count", "critical"))
        warning = first_value(alarm_data, ("warning_count", "warning"))
        alarm_rows = first_list(alarm_data, ("alarms", "rows", "items", "active_alarms"))
        has_offline_alarm = any(
            "offline" in str(first_value(row, ("alarm_name", "name", "title", "message", "description", "alarm_type")) or "").lower()
            for row in alarm_rows if isinstance(row, dict)
        )
        if active is not None:
            alarm_line = f"- Alarms: {format_number(active, 0)} active"
            if critical is not None:
                alarm_line += f", including {format_number(critical, 0)} critical"
            if warning is not None:
                alarm_line += f" and {format_number(warning, 0)} warning"
            alarm_line += ". Prioritize the listed critical/offline alarms for review." if (critical or 0) > 0 else "."
            lines.append(alarm_line)

    meter_item = by_operation.get("meter_status_summary")
    meter_data = historical_result_data(meter_item.get("result") or {}) if meter_item else {}
    meter_offline = first_value(meter_data, ("offline_count", "offline", "devices_offline")) if meter_data else None
    meter_rows = first_list(meter_data, ("devices", "meters", "rows", "items", "needing_attention")) if meter_data else []
    has_offline_meter = bool(isinstance(meter_offline, (int, float)) and meter_offline > 0)
    has_offline_meter = has_offline_meter or any(
        "offline" in str(first_value(row, ("status", "communication_status", "data_status")) or "").lower()
        for row in meter_rows if isinstance(row, dict)
    )

    metadata_item = by_operation.get("site_metadata_summary")
    metadata_data = historical_result_data(metadata_item.get("result") or {}) if metadata_item else {}
    site_name = first_value(metadata_data, ("site_name", "name", "label")) if metadata_data else None
    device_list_item = by_operation.get("site_device_list")
    device_data = historical_result_data(device_list_item.get("result") or {}) if device_list_item else {}
    device_count = first_value(device_data, ("total_count", "total", "device_count", "count")) if device_data else None
    if device_count is None and device_data:
        device_rows = first_list(device_data, ("devices", "rows", "items", "meters"))
        if device_rows:
            device_count = len(device_rows)

    overview_parts = []
    if site_name:
        overview_parts.append(f"{site_name}")
    if device_count is not None:
        overview_parts.append(f"{format_number(device_count, 0)} devices")
    if alarm_data and active is not None:
        alarm_overview = f"{format_number(active, 0)} active alarms"
        if critical is not None:
            alarm_overview += f" ({format_number(critical, 0)} critical)"
        overview_parts.append(alarm_overview)
    if overview_parts:
        lines.insert(0, "- " + "; ".join(overview_parts) + ".")
    if has_offline_alarm or has_offline_meter:
        lines.append(
            "- Suggested checks for offline devices: verify device power and indicators, check the network/gateway link and communication settings, then confirm fresh readings resume. These are troubleshooting steps, not a confirmed root cause."
        )

    ranking_item = next((
        item for item in results
        if item.get("operation_id") in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}
    ), None)
    ranking_data = historical_result_data(ranking_item.get("result") or {}) if ranking_item else {}
    if ranking_data:
        measured, _ = ranked_consumer_rows(ranking_data)
        if measured:
            row, value = measured[0]
            name = row.get("device_name") or row.get("name") or row.get("label") or "Unnamed device"
            lines.append(f"- Consumption leader: {name} is highest among the returned devices at {format_number(value, 2)} kWh for the selected period.")
            total = first_value(energy_data, ("total", "total_value", "value", "total_kwh", "consumption", "energy"))
            top_five_sum = sum(amount for _, amount in measured[:5])
            if isinstance(total, (int, float)) and total > 0 and top_five_sum > 0 and total / top_five_sum > 100:
                lines.append(
                    f"- Data check: the site total ({format_number(total, 2)} kWh) is much larger than the returned top-five total ({format_number(top_five_sum, 2)} kWh). Confirm the time range, meter scope, and units before comparing them directly."
                )

    telemetry_item = by_operation.get("telemetry_timeseries")
    if telemetry_item and energy_days:
        telemetry_data = historical_result_data(telemetry_item.get("result") or {})
        telemetry_rows = first_list(telemetry_data, ("series", "rows", "items"))
        timestamps = [
            parse_datetime(first_value(row, ("last_seen", "last_timestamp", "timestamp")))
            for row in telemetry_rows if isinstance(row, dict)
        ]
        timestamps = [stamp for stamp in timestamps if stamp]
        if timestamps:
            latest_telemetry_day = max(timestamps).astimezone(timezone(timedelta(hours=8))).date().isoformat()
            if energy_days[-1][0] > latest_telemetry_day:
                lines.append(
                    f"- Coverage check: the daily energy list reaches {energy_days[-1][0]}, while the detailed telemetry timestamp reaches {latest_telemetry_day}; verify the newest day's completeness."
                )

    return "At a glance:\n" + "\n".join(lines) if lines else ""


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

    comparison_results = [item for item in results if item.get("intent_id") == "energy_comparison_summary"]
    if len(comparison_results) == 2:
        totals = []
        for item in comparison_results:
            data = historical_result_data(item.get("result", {}))
            total = first_value(data, ("total_kwh", "total", "total_value", "value"))
            if total is None:
                buckets = first_list(data, ("buckets", "rows", "series"))
                numeric = [chart_value(row, ("value", "kwh", "energy")) for row in buckets if isinstance(row, dict)]
                numeric = [value for value in numeric if value is not None]
                total = sum(numeric) if numeric else None
            totals.append((item.get("period_label", "period"), total))
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for _, value in totals):
            (label_a, value_a), (label_b, value_b) = totals
            delta = value_b - value_a
            percent = (delta / value_a * 100) if value_a else None
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
            lines = [f"Energy comparison: period {label_a} = {format_number(value_a, 2)} kWh; period {label_b} = {format_number(value_b, 2)} kWh."]
            lines.append(f"Period {label_b} was {format_number(abs(delta), 2)} kWh {direction} than period {label_a}" + (f" ({format_number(abs(percent), 2)}%)." if percent is not None else ". Percentage change is unavailable because the reference period is zero."))
            return "\n".join(lines)

    top_result = next((item for item in results if item["operation_id"] in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}), None)
    device_result = next((item for item in results if item["operation_id"] == "site_device_list"), None)
    if top_result and top_result.get("operation_id") == "device_energy_breakdown" and wants_energy_breakdown_request(message):
        top_data = historical_result_data(top_result["result"])
        device_data = historical_result_data(device_result["result"]) if device_result else {}
        return summarize_energy_breakdown(
            top_data,
            top_result.get("arguments"),
            message,
            inventory_data=device_data,
        )
    if top_result and device_result and wants_device_usage_ranking(message):
        top_data = historical_result_data(top_result["result"])
        device_data = historical_result_data(device_result["result"])
        devices = first_list(device_data, ("devices", "rows", "items", "meters"))
        by_id = {
            str(device.get("device_id") or device.get("id") or device.get("meter_id")): device
            for device in devices if isinstance(device, dict)
        }
        measured, _ = ranked_consumer_rows(top_data)
        limit = int(top_result.get("arguments", {}).get("limit") or 5)
        lines = [summarize_top_consumers(top_data, top_result.get("arguments"), message), "Current status of ranked devices:"]
        ranked_ids = set()
        for index, (row, _) in enumerate(measured[:limit], 1):
            device_id = row.get("device_id") or row.get("id") or row.get("meter_id")
            matched = by_id.get(str(device_id)) if device_id is not None else None
            if device_id is not None:
                ranked_ids.add(str(device_id))
            name = row.get("device_name") or row.get("name") or "Unnamed device"
            status = (matched or {}).get("status") or (matched or {}).get("connection_status") or row.get("status")
            lines.append(f"{index}. {name}: {status or 'status unavailable'}")
        if not measured:
            lines.append("No measured devices could be matched to current status.")
        if wants_all_devices(message):
            unranked = [device for device in devices if isinstance(device, dict)
                        and str(device.get("device_id") or device.get("id") or device.get("meter_id")) not in ranked_ids]
            if unranked:
                lines.append("Other devices (usage not returned, so not ranked):")
                for device in unranked:
                    name = device.get("device_name") or device.get("name") or "Unnamed device"
                    lines.append(f"- {name}")
            total = device_data.get("device_count") or device_data.get("row_count") or len(devices)
            if str(total).isdigit() and int(total) > len(devices):
                lines.append(f"Device inventory returned {len(devices)} of {total} devices; the rest are not shown.")
        remaining = [item for item in results if item not in (top_result, device_result)]
        if not remaining:
            return "\n".join(lines)
        return "\n".join(lines) + "\n\n" + summarize_historical_answers(message, remaining, request_id)

    sections = []
    for item in results:
        operation_id = item["operation_id"]
        title = {
            "device_energy_ranking": "Top energy-consuming devices",
            "device_energy_breakdown": "Top energy-consuming devices" if wants_device_usage_ranking(message) else "Device energy breakdown",
            "telemetry_top_consumers": "Top energy-consuming devices",
            "site_energy_summary": "Site energy summary",
            "alarm_frequency_summary": "Alarm frequency summary",
            "site_metadata_summary": "Site details",
            "site_device_list": "Devices",
            "active_alarm_summary": "Active alarms",
            "meter_status_summary": "Meter status",
        }.get(operation_id, operation_id)
        summary = summarize_historical_answer(
            message,
            operation_id,
            item["result"],
            request_id,
            item.get("arguments"),
        )
        sections.append(f"{title}\n{summary}")
    takeaway = summarize_result_takeaway(results)
    if takeaway:
        sections.insert(0, takeaway)
    return "\n\n".join(sections)


def refine_historical_answer_with_model(message: str, results: list[dict], deterministic_answer: str, request_id: str) -> str:
    display_timezone = next((
        item.get("arguments", {}).get("timezone")
        for item in results
        if isinstance(item.get("arguments"), dict) and item["arguments"].get("timezone")
    ), AI_DISPLAY_TIMEZONE)
    if not AI_REFINE_MCP_WITH_MODEL:
        return format_answer_timestamps(deterministic_answer, display_timezone)
    prompt = build_mcp_refine_prompt(message, results, deterministic_answer, request_id)
    if wants_sitewide_energy_total(message):
        return format_answer_timestamps(deterministic_answer, display_timezone)
    primary_answer = ensure_historical_answer_coverage(
        message, results, deterministic_answer,
        run_mcp_refine_model(prompt, request_id, CHAT_MODEL, deterministic_answer, role="primary"),
        request_id,
    )
    if not AI_COMPARE_MODEL_ENABLED:
        return format_answer_timestamps(primary_answer, display_timezone)
    comparison_answers = []
    for model in AI_COMPARE_MODELS:
        if not model or model == CHAT_MODEL:
            continue
        compare_answer = run_mcp_refine_model(prompt, request_id, model, "", role="compare")
        if not compare_answer:
            continue
        compare_answer = ensure_historical_answer_coverage(message, results, deterministic_answer, compare_answer, request_id)
        debug_trace_event("mcp_answer_compare_selected", request_id=request_id, model=model, role="compare")
        comparison_answers.append((model, compare_answer))
    if comparison_answers:
        log_event(
            "mcp_answer_compare_hidden",
            request_id=request_id,
            primary_model=CHAT_MODEL,
            compare_models=[model for model, _ in comparison_answers],
        )
    return format_answer_timestamps(primary_answer, display_timezone)


def ensure_historical_answer_coverage(
    message: str, results: list[dict], deterministic_answer: str, model_answer: str, request_id: str,
) -> str:
    if not model_answer:
        return deterministic_answer
    generic_failure_patterns = (
        r"\bmessage did(?: not|n['’]t) come through\b",
        r"\bempty input\b",
        r"\btesting the system\b",
        r"\bseems like you might have intended to ask\b",
        r"\bsent a message with just\b",
        r"\bhello!?\s+how can i assist you today\b",
        r"\bhow can i assist you today\b",
        r"\bplease let me know if you have any questions\b",
        r"\bneed help with something specific\b",
        r"\bno specific question\b",
        r"\bclarify what you(?:'|’)d like to ask\b",
        r"\bwhat you(?:'|’)d like to ask\b",
        r"\bi(?:'|’)m here to help\b",
        r"\bcould you clarify\b",
    )
    if results and any(re.search(pattern, model_answer, flags=re.IGNORECASE) for pattern in generic_failure_patterns):
        log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="generic_model_reply")
        return deterministic_answer

    # Successful tool data is the factual source of truth. If the model's rewrite
    # omits the returned entities entirely, show the deterministic summary.
    for item in results:
        operation_id = item.get("operation_id")
        data = historical_result_data(item.get("result") or {})
        if operation_id == "latest_telemetry_snapshot":
            latest_match = re.search(r"^Latest .* reading[^:]*:\s*([\d,]+(?:\.\d+)?)", deterministic_answer, re.MULTILINE)
            expected_value = latest_match.group(1) if latest_match else None
            if expected_value and expected_value not in model_answer:
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="latest_telemetry_value")
                return deterministic_answer
        elif operation_id == "telemetry_metric_ranking":
            rows = first_list(data, ("ranked_devices", "rankings", "devices"))
            expected = rows[:min(len(rows), requested_top_limit(message) or 3)]
            for row in expected:
                name = str(row.get("device_name") or "").strip()
                value = format_number(row.get("value"), 4)
                if (name and name.casefold() not in model_answer.casefold()) or value not in model_answer:
                    log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="frequency_ranking")
                    return deterministic_answer
            excluded = first_list(data, ("excluded_devices",))
            if excluded and str(len(excluded)) not in model_answer:
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="frequency_exclusions")
                return deterministic_answer
        elif operation_id == "site_device_list":
            rows = first_list(data, ("devices", "rows", "items", "meters"))
            expected = [
                str(row.get("device_name") or row.get("name") or row.get("meter_name") or "").strip()
                for row in rows[:3] if isinstance(row, dict)
            ]
            expected = [name for name in expected if name]
            if expected and not any(name.casefold() in model_answer.casefold() for name in expected):
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="device_inventory")
                return deterministic_answer
        elif operation_id in {"alarm_frequency_summary", "active_alarm_summary"}:
            rows = first_list(data, ("rows", "alarms", "items"))
            expected = [
                str(row.get("alarm_name") or row.get("name") or row.get("alarm_type") or "").strip()
                for row in rows[:3] if isinstance(row, dict)
            ]
            expected = [name for name in expected if name]
            if expected and not any(name.casefold() in model_answer.casefold() for name in expected):
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="alarm_rows")
                return deterministic_answer
        elif operation_id == "telemetry_timeseries":
            telemetry = telemetry_series_points(data)
            device = data.get("device") if isinstance(data.get("device"), dict) else {}
            expected_device = first_value(device, ("device_name", "name", "label"))
            if expected_device and str(expected_device).casefold() not in model_answer.casefold():
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="telemetry_device")
                return deterministic_answer
            for series in telemetry:
                values = [value for _, value in series["points"]]
                expected_average = format_number(sum(values) / len(values), 2)
                if expected_average not in model_answer or (series["unit"] and series["unit"] not in model_answer):
                    log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="telemetry_summary")
                    return deterministic_answer
    top_result = next((item for item in results if item.get("operation_id") in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}), None)
    if top_result and wants_device_usage_ranking(message):
        top_data = historical_result_data(top_result["result"])
        measured, _ = ranked_consumer_rows(top_data)
        limit = int(top_result.get("arguments", {}).get("limit") or 5)
        for row, amount in measured[:limit]:
            name = str(row.get("device_name") or row.get("name") or row.get("label") or "")
            value = str(int(amount)) if amount.is_integer() else str(amount)
            if (name and name.lower() not in model_answer.lower()) or value not in model_answer:
                log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="device_ranking")
                return deterministic_answer
        if any(term in message.lower() for term in ("offline", "online", "status")):
            status_lines = [line for line in deterministic_answer.splitlines() if re.match(r"^\d+\. .+\(ID .+\): ", line)]
            for line in status_lines:
                status = line.rsplit(": ", 1)[-1]
                if status.lower() not in model_answer.lower():
                    log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="device_status")
                    return deterministic_answer
    missing = []
    lowered = model_answer.lower()
    has_energy = any(item.get("operation_id") == "site_energy_summary" for item in results)
    wants_highest, wants_lowest = requested_energy_extrema(message) if has_energy else (False, False)
    wants_daily = has_energy and any(term in message.lower() for term in ("daily", "each day", "per day"))
    for line in deterministic_answer.splitlines():
        date_match = re.match(r"^- (\d{4}-\d{2}-\d{2}):", line)
        value_match = re.search(r"\b[\d,]+(?:\.\d+)?\s*(?:kWh|MWh|GWh)\b", line, re.IGNORECASE)
        if wants_daily and date_match and (
            date_match.group(1) not in model_answer
            or (value_match and value_match.group(0) not in model_answer)
        ):
            missing.append(line)
        if wants_highest and line.startswith("Highest daily energy:") and (
            not re.search(r"\b(?:highest|maximum|peak)\b", lowered)
            or not any(day in model_answer for day in re.findall(r"\d{4}-\d{2}-\d{2}", line))
            or (value_match and value_match.group(0) not in model_answer)
        ):
            missing.append(line)
        if wants_lowest and line.startswith("Lowest daily energy:") and (
            not re.search(r"\b(?:lowest|minimum)\b", lowered)
            or not any(day in model_answer for day in re.findall(r"\d{4}-\d{2}-\d{2}", line))
            or (value_match and value_match.group(0) not in model_answer)
        ):
            missing.append(line)
    if (wants_highest or wants_lowest) and not any(
        line.startswith(("Highest daily energy:", "Lowest daily energy:")) for line in deterministic_answer.splitlines()
    ):
        unavailable = next(
            (line for line in deterministic_answer.splitlines() if "unavailable because no complete daily readings" in line),
            None,
        )
        if unavailable and unavailable not in model_answer:
            missing.append(unavailable)
    if len(results) > 1:
        markers = {
            "site_energy_summary": r"\b(?:energy|usage|consumption|kwh|mwh)\b",
            "alarm_frequency_summary": r"\b(?:alarm|alert)\b",
            "active_alarm_summary": r"\b(?:alarm|alert)\b",
            "device_energy_ranking": r"\b(?:device|meter|consumer|umg)\b",
            "device_energy_breakdown": r"\b(?:device|meter|consumer|umg|energy)\b",
            "telemetry_top_consumers": r"\b(?:device|meter|consumer|umg)\b",
            "demand_peak_summary": r"\b(?:demand|peak)\b",
            "energy_forecast": r"\b(?:forecast|predict|estimate)\b",
        }
        for item in results:
            operation_id = item.get("operation_id")
            marker = markers.get(operation_id)
            if marker and not re.search(marker, lowered):
                missing.append(summarize_historical_answer(
                    message, operation_id, item["result"], request_id, item.get("arguments")
                ))
    if not missing:
        return model_answer
    log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items=len(missing))
    return f"{model_answer.rstrip()}\n\nRequested details:\n" + "\n".join(dict.fromkeys(missing))


def build_mcp_refine_prompt(message: str, results: list[dict], deterministic_answer: str, request_id: str = "") -> str:
    mcp_context = {
        "enabled": True,
        "tools": [item.get("operation_id") for item in results],
        "results": [
            {"tool": item.get("operation_id"), "result": item.get("result") or {}}
            for item in results
        ],
        "errors": [],
    }
    raw_context = format_daxview_context(mcp_context)
    if wants_named_device_metric_query(message) and not context.get("device_id"):
        raw_context = re.sub(r"\b(?:device_id|meter_id)\s*[:=]\s*\d+", "", raw_context, flags=re.IGNORECASE)
    question_parts = requested_question_parts(message)
    question_plan = "\n".join(f"{index}. {part}" for index, part in enumerate(question_parts, 1))
    library_section = ""
    if needs_ems_library(message):
        try:
            library_rows = retrieve_context(message)[:3]
            if library_rows:
                library_section = "\nRelevant EMS library excerpts:\n" + "\n".join(
                    f"- {row.get('title')}: {str(row.get('chunk_text') or '')[:900]}"
                    for row in library_rows
                )
        except Exception as error:
            log_event("ems_library_retrieval_error", request_id=request_id, error=str(error))
    prompt = f"""You are an EMS operations assistant.
Use DaxView MCP data for site-specific facts. Use library excerpts only for relevant general EMS explanations.
Make it human, clean, and practical.
Do not invent values, devices, alarms, timestamps, causes, or recommendations not supported by the MCP data.
Conceptual EMS model (logical relationships only; not a verified physical database schema): a site scopes buildings and devices; buildings contain floors; devices may have locations; each device has configured metrics; readings belong to a configured metric and include time, numeric value, and unit; alarms belong to a site and may optionally reference a device. Use device names for people and IDs only to connect returned records internally. Never infer physical joins, missing metric configuration, or a device's supported parameters from this model.
Render timestamps as concise local times in the requested/site timezone (for example, "30 Sep 2026, 3:53 PM MYT"). Do not repeat raw ISO timestamps or microseconds in user-facing prose.
Answer each part of the user's question using the matching tool result. Never present energy consumption as an alarm count.
Include only the requested metric, device, time range, and necessary data-quality caveat. Do not add supported-parameter lists, unrelated status fields, IDs, or other returned fields unless the user asks for them.
For energy, preserve the source value, unit, time range, and scope exactly. Do not add cumulative register readings as if they were interval consumption. Only describe a value as interval consumption when the returned data establishes that meaning; otherwise say DaxView returned the value and avoid claiming the measurement semantics are verified. Do not silently recompute or "correct" a backend total.
For telemetry trends, explain what the chart shows in plain language. Keep each phase or metric series separate, retain its returned unit, and preserve the per-series averages and ranges from the deterministic draft. A successful chart with numeric readings is a valid answer; never replace it with a greeting or say the user's message did not come through.
When valid site-specific data is present, answer from that data; do not claim the user gave no question or ask them to clarify unless the returned data genuinely lacks a required scope or value.
Do not include standards, protocols, or library background unless the user asks for them.
For predictions, label every future value as an estimate and retain the method, coverage, and backtest error.
Answer every numbered request below. If a value was not returned, state that for the affected request.
When daily values are requested, retain every date and its value from the deterministic draft.
Do not show internal device IDs in user-facing text or tables. Refer to devices by name; use a device-name choice when the name is ambiguous.
Keep the answer concise but useful:
- Start with the direct answer.
- Use short sections and bullets.
- For a single-metric question, answer with that metric only; do not turn the response into a device inventory or capability report.
- For a named-device metric question, answer with that device's requested metric/time range. If the data fetch fails, report the failure clearly; do not substitute an inventory response.
- For telemetry trends, summarize the returned period, aggregation, plotted point count, per-series average and range, and the overall observed range. State the displayed timezone; do not infer whether values are normal or abnormal without a threshold or request.
- For a request asking for an energy breakdown, use the per-device energy result for the site and period explicitly present in the current request or authorized request context. Explain that it is a device-level breakdown and preserve returned values and units.
- For energy breakdowns, state building totals and the top five device values by default. If the user asks for the full/complete breakdown, include every returned device and say if DaxView returned fewer rows than the site contains. Show only calculations present in the deterministic draft; explain building totals as the sum of returned device energy rows assigned to that building.
- Mention the tool/data limitation only if a requested value is missing.
- Hide backend/internal terms such as MCP, row_count, source_count, aggregation, authorization_id, request_id, and JSON field names.
- If the user asks for max demand, explain whether the result is site-level, building-level, or device-level based on the data. If device-level detail is missing, state what additional device-level data is needed.
- If demand data and device-list data are both present, do not say the device list caused or contributed to the max demand unless the data explicitly links demand to those devices. Present devices as "devices to inspect" or "candidate meters to check".
- If the actual max-demand value is missing but device counts are present, say the max-demand value was not returned and then list the useful device/status context separately.

User question:
{message}

Measurement definitions (preserve units and aggregation):
{json.dumps(measurement_context(message))}

Requested parts:
{question_plan}

Deterministic draft:
{deterministic_answer}

DaxView data summary:
{raw_context}
{library_section}

Final answer:"""


def run_mcp_refine_model(prompt: str, request_id: str, model: str, fallback_answer: str, role: str = "primary") -> str:
    started_at = time.perf_counter()
    log_event("mcp_answer_refine_request", request_id=request_id, model=model, role=role, prompt_preview=preview(prompt))
    try:
        data = ollama_json(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.15},
            },
            timeout=OLLAMA_GENERATE_TIMEOUT,
        )
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        log_event("mcp_answer_refine_error", request_id=request_id, model=model, role=role, error=str(error))
        debug_trace_event("mcp_answer_refine_fallback", request_id=request_id, model=model, role=role, reason="model_request_failed")
        return fallback_answer
    reply = clean_final_answer(str(data.get("message", {}).get("content") or data.get("response", "")).strip())
    if not reply:
        debug_trace_event("mcp_answer_refine_fallback", request_id=request_id, model=model, role=role, reason="empty_model_answer")
        return fallback_answer
    log_event("mcp_answer_refine_response", request_id=request_id, model=model, role=role, duration_ms=round((time.perf_counter() - started_at) * 1000))
    debug_trace_event(
        "mcp_answer_refine_response_debug",
        request_id=request_id,
        model=model,
        role=role,
        duration_ms=round((time.perf_counter() - started_at) * 1000),
        answer=reply,
    )
    return reply


def chart_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def chart_label(row: dict, fallback: str = "item") -> str:
    for key in ("device_name", "alarm_name", "name", "label", "timestamp", "bucket", "date", "started_at"):
        value = row.get(key)
        if value:
            return str(value)
    for key in ("device_id", "alarm_id", "id"):
        value = row.get(key)
        if value is not None:
            return f"{fallback} {value}"
    return fallback


def chart_value(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = chart_number(row.get(key))
        if value is not None:
            return value
    return None


def build_chart_spec(chart_type: str, title: str, labels: list[str], values: list[float], unit: str = "", series_name: str = "value") -> dict | None:
    if not labels or not values or len(labels) != len(values):
        return None
    return {
        "type": chart_type,
        "title": title,
        "labels": labels,
        "series": [{"name": series_name, "unit": unit, "data": values}],
    }


def chart_from_historical_result(operation_id: str, result: dict, arguments: dict | None = None, message: str = "") -> dict | None:
    data = historical_result_data(result)
    if not data:
        return None
    if operation_id == "site_energy_summary":
        rows = first_list(data, ("series", "buckets", "rows", "data"))
        labels = []
        values = []
        for row in rows[:60]:
            if not isinstance(row, dict):
                continue
            label = str(row.get("timestamp") or row.get("bucket") or row.get("date") or row.get("start") or "")
            value = chart_value(row, ("value", "kwh", "energy", "consumption"))
            if label and value is not None:
                labels.append(label)
                values.append(value)
        unit = data.get("unit") or (data.get("display") or {}).get("unit") if isinstance(data.get("display"), dict) else data.get("unit") or "kWh"
        return build_chart_spec("bar", "Daily Site Energy", labels, values, unit or "kWh", "Energy")
    if operation_id in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}:
        ranked, _ = ranked_consumer_rows(data)
        rows = [dict(row, value=value) for row, value in ranked]
        labels = []
        values = []
        chart_limit = 100 if wants_full_energy_breakdown(message) else 5 if wants_energy_breakdown_request(message) else 10
        for row in rows[:chart_limit]:
            if not isinstance(row, dict):
                continue
            value = chart_value(row, (
                "value", "kwh", "total_kwh", "consumption", "energy",
                "energy_kwh", "total_energy", "total_energy_kwh",
                "device_energy_kwh", "contribution_kwh", "usage_kwh",
                "consumption_delta",
            ))
            if value is not None:
                labels.append(chart_label(row, "device"))
                values.append(value)
        return build_chart_spec("bar", "Top Energy Consumers", labels, values, "kWh", "Consumption")
    if operation_id == "alarm_frequency_summary":
        rows = first_list(data, ("rows", "alarms", "items"))
        labels = []
        values = []
        for row in rows[:10]:
            if not isinstance(row, dict):
                continue
            value = chart_value(row, ("event_count", "occurrence_count", "alarm_count", "count", "frequency", "total", "value"))
            if value is not None:
                labels.append(chart_label(row, "alarm"))
                values.append(value)
        return build_chart_spec("bar", "Alarm Frequency", labels, values, "occurrence(s)", "Alarms")
    if operation_id == "active_alarm_summary":
        labels = []
        values = []
        for key, label in (("critical_count", "Critical"), ("warning_count", "Warning"), ("active_count", "Active")):
            value = chart_number(data.get(key))
            if value is not None:
                labels.append(label)
                values.append(value)
        return build_chart_spec("bar", "Active Alarm Counts (total includes severities)", labels, values, "alarm(s)", "Alarms")
    if operation_id in {"meter_status_summary", "site_device_list"}:
        labels = []
        values = []
        for key, label in (("online_count", "Online"), ("offline_count", "Offline"), ("stale_count", "Stale"), ("unknown_count", "Unknown")):
            value = chart_number(data.get(key))
            if value is not None:
                labels.append(label)
                values.append(value)
        if not values:
            rows = first_list(data, ("rows", "devices", "items", "meters"))
            counts: dict[str, float] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or row.get("data_status") or "unknown").title()
                counts[status] = counts.get(status, 0) + 1
            labels = list(counts.keys())
            values = list(counts.values())
        return build_chart_spec("donut", "Meter Status", labels, values, "device(s)", "Devices")
    if operation_id == "demand_peak_summary":
        ranked_devices = first_list(data, ("ranked_devices", "rankings", "devices"))
        if ranked_devices:
            labels = []
            values = []
            for row in ranked_devices[:10]:
                if not isinstance(row, dict):
                    continue
                value = chart_value(row, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "max_kw"))
                if value is not None:
                    labels.append(chart_label(row, "device"))
                    values.append(value)
            if labels:
                return build_chart_spec("bar", "Peak Demand by Device", labels, values, "kW", "Maximum demand")
        rows = first_list(data, ("rows", "items", "peaks", "series"))
        labels = []
        values = []
        for row in rows[:20]:
            if not isinstance(row, dict):
                continue
            value = chart_value(row, ("peak_kw", "max_kw", "demand_kw", "value", "peak", "maximum"))
            if value is not None:
                labels.append(chart_label(row, "peak"))
                values.append(value)
        if not values:
            for series in first_list(data, ("series",)):
                if not isinstance(series, dict):
                    continue
                factor = {"w": 0.001, "kw": 1.0, "mw": 1000.0}.get(str(series.get("unit") or "kw").lower(), 1.0)
                for point in (series.get("points") if isinstance(series.get("points"), list) else []):
                    if not isinstance(point, dict):
                        continue
                    value = chart_value(point, ("value", "demand_kw", "kw", "reading"))
                    label = point.get("timestamp") or point.get("time") or point.get("bucket")
                    if value is not None and label:
                        labels.append(str(label))
                        values.append(value * factor)
        if not values:
            value = chart_value(data, ("peak_kw", "max_kw", "demand_kw", "value", "peak", "maximum"))
            if value is not None:
                labels = [str(data.get("peak_time") or data.get("timestamp") or "Peak demand")]
                values = [value]
        return build_chart_spec("bar", "Peak Demand", labels, values, data.get("unit") or "kW", "Demand")
    if operation_id == "telemetry_metric_ranking":
        rows = first_list(data, ("ranked_devices", "rankings", "devices"))
        labels = []
        values = []
        for row in rows[:50]:
            if not isinstance(row, dict):
                continue
            value = chart_value(row, ("value", "reading", "frequency_hz"))
            if value is not None:
                labels.append(chart_label(row, "device"))
                values.append(value)
        return build_chart_spec("bar", "Highest Frequency by Device", labels, values, data.get("unit") or "Hz", "Frequency")
    if operation_id == "telemetry_timeseries":
        telemetry_series = telemetry_series_points(data)
        if not telemetry_series:
            return None
        timestamps = sorted({stamp for item in telemetry_series for stamp, _ in item["points"]})
        timezone_name = (arguments or {}).get("timezone") or AI_DISPLAY_TIMEZONE
        labels = [format_chart_timestamp(stamp, timezone_name) for stamp in timestamps]
        chart_series = []
        for item in telemetry_series:
            values_by_time = dict(item["points"])
            chart_series.append({
                "name": item["name"],
                "unit": item["unit"],
                "data": [values_by_time.get(stamp) for stamp in timestamps],
            })
        metric = (arguments or {}).get("metric", "Telemetry")
        device = data.get("device") if isinstance(data.get("device"), dict) else {}
        device_name = first_value(device, ("device_name", "name", "label")) or (arguments or {}).get("_device_name")
        title = f"{device_name} - {str(metric).title()} Trend" if device_name else f"{str(metric).title()} Trend"
        summary_lines = summarize_telemetry_timeseries(data, arguments).splitlines()
        chart_summary = next((line for line in summary_lines if line.startswith("Summary:")), summary_lines[0])
        return {
            "type": "line",
            "title": title,
            "labels": labels,
            "series": chart_series,
            "summary": chart_summary,
            "beginAtZero": False,
        }
    if operation_id == "energy_forecast":
        rows = first_list(data, ("forecast", "rows", "series", "items"))
        labels = []
        values = []
        for row in rows[:30]:
            if not isinstance(row, dict):
                continue
            value = chart_value(row, ("value", "forecast", "kwh", "energy", "predicted_value"))
            if value is not None:
                labels.append(chart_label(row, "forecast"))
                values.append(value)
        return build_chart_spec("line", "Energy Forecast", labels, values, data.get("unit") or "kWh", "Forecast")
    return None


def build_charts_from_historical_results(results: list[dict], preview: bool = False, message: str = "") -> list[dict]:
    if not AI_CHARTS_ENABLED and not preview:
        return []
    charts = []
    for item in results:
        chart = chart_from_historical_result(
            str(item.get("operation_id")),
            item.get("result") if isinstance(item.get("result"), dict) else {},
            item.get("arguments") if isinstance(item.get("arguments"), dict) else None,
            message,
        )
        if chart:
            charts.append(chart)
    return charts


def build_tables_from_historical_results(results: list[dict], message: str = "") -> list[dict]:
    """Build debug-only tables from verified MCP rows; never ask the model to invent cells."""
    tables = []
    latest_requested = bool(re.search(r"\b(?:latest|most recent)\b", message, re.IGNORECASE))
    inventory_item = next((item for item in results if item.get("operation_id") == "site_device_list"), None)
    inventory_data = historical_result_data(inventory_item.get("result") or {}) if inventory_item else {}
    inventory_by_id = {
        str(first_value(row, ("device_id", "id", "meter_id"))): row
        for row in first_list(inventory_data, ("devices", "rows", "items", "meters"))
        if isinstance(row, dict) and first_value(row, ("device_id", "id", "meter_id")) is not None
    }
    for item in results:
        operation_id = str(item.get("operation_id") or "")
        if operation_id == "site_device_list" and wants_energy_breakdown_request(message) and any(
            result.get("operation_id") == "device_energy_breakdown" for result in results
        ):
            # This inventory call only enriches energy rows with building names;
            # do not expose the helper list as an unrelated second table.
            continue
        data = historical_result_data(item.get("result") if isinstance(item.get("result"), dict) else {})
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        rows = []
        title = ""
        columns = []
        extra_tables = []

        if operation_id == "alarm_frequency_summary":
            source_rows = [row for row in first_list(data, ("rows", "alarms", "items")) if isinstance(row, dict)]
            if latest_requested:
                source_rows = sorted(
                    source_rows,
                    key=lambda row: parse_datetime(first_value(row, (
                        "latest_occurrence", "last_occurrence", "last_seen", "last_seen_at", "latest_at",
                    ))) or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True,
                )
            else:
                source_rows = sorted(
                    source_rows,
                    key=lambda row: chart_number(first_value(row, (
                        "event_count", "occurrence_count", "alarm_count", "count", "frequency",
                    ))) or 0,
                    reverse=True,
                )
            table_limit = min(max(int(arguments.get("limit") or 10), 1), 100)
            for rank, row in enumerate(source_rows[:table_limit], 1):
                device_name = first_value(row, ("device_name", "meter_name", "device"))
                rows.append({
                    "rank": rank,
                    "alarm": first_value(row, ("alarm_name", "alarm_type", "alarm", "name", "type")) or "Alarm",
                    "device": device_name or "Unnamed device",
                    "severity": first_value(row, ("severity", "priority")) or "",
                    "occurrences": first_value(row, ("event_count", "occurrence_count", "alarm_count", "count", "frequency")),
                    "first_seen": first_value(row, ("first_seen", "first_occurrence", "started_at")),
                    "latest": first_value(row, ("latest_occurrence", "last_occurrence", "last_seen", "last_seen_at", "latest_at")),
                    "active": first_value(row, ("active_count", "active")),
                    "acknowledged": first_value(row, ("acknowledged_count", "acknowledged")),
                    "resolved": first_value(row, ("resolved_count", "resolved")),
                    "closed": first_value(row, ("closed_count", "closed")),
                    "alarm_id": first_value(row, ("alarm_event_id", "alarm_id", "event_id")),
                })
            title = "Latest Alarm Occurrences" if latest_requested else "Alarm Frequency"
            columns = [
                ("rank", "#"), ("alarm", "Alarm"), ("device", "Device"), ("severity", "Severity"),
                ("first_seen", "First occurrence"), ("latest", "Latest occurrence"),
                ("occurrences", "Occurrences"), ("active", "Active"), ("acknowledged", "Acknowledged"),
                ("resolved", "Resolved"), ("closed", "Closed"), ("alarm_id", "Alarm ID"),
            ]
        elif operation_id in {"device_energy_breakdown", "device_energy_ranking", "telemetry_top_consumers"}:
            measured, _ = ranked_consumer_rows(data)
            is_energy_breakdown = operation_id == "device_energy_breakdown" and wants_energy_breakdown_request(message)
            if is_energy_breakdown:
                limit = len(measured) if wants_full_energy_breakdown(message) else min(len(measured), 5)
            else:
                limit = min(max(int(arguments.get("limit") or 20), 1), 100)
            for rank, (row, amount) in enumerate(measured[:limit], 1):
                device_id = first_value(row, ("device_id", "id", "meter_id"))
                device = inventory_by_id.get(str(device_id), {}) if device_id is not None else {}
                building_name = first_value(row, ("building_name", "building", "building_label"))
                if isinstance(building_name, dict):
                    building_name = first_value(building_name, ("name", "building_name", "label"))
                if not building_name:
                    building_name = first_value(device, ("building_name", "building", "building_label"))
                    if isinstance(building_name, dict):
                        building_name = first_value(building_name, ("name", "building_name", "label"))
                table_row = {
                    "rank": rank,
                    "device": row.get("device_name") or row.get("name") or row.get("label") or first_value(device, ("device_name", "name", "meter_name", "label")) or "Unnamed device",
                    "value": amount,
                    "unit": "kWh" if is_energy_breakdown else row.get("unit") or data.get("unit") or "kWh",
                    "status": row.get("status") or row.get("device_status") or "",
                }
                if is_energy_breakdown:
                    table_row["building"] = building_name or ""
                rows.append(table_row)
            if is_energy_breakdown:
                building_totals = {}
                for row, amount in measured:
                    device_id = first_value(row, ("device_id", "id", "meter_id"))
                    device = inventory_by_id.get(str(device_id), {}) if device_id is not None else {}
                    building_name = first_value(row, ("building_name", "building", "building_label"))
                    if isinstance(building_name, dict):
                        building_name = first_value(building_name, ("name", "building_name", "label"))
                    if not building_name:
                        building_name = first_value(device, ("building_name", "building", "building_label"))
                        if isinstance(building_name, dict):
                            building_name = first_value(building_name, ("name", "building_name", "label"))
                    if building_name:
                        group = building_totals.setdefault(str(building_name), {"total": 0.0, "devices": 0})
                        group["total"] += amount
                        group["devices"] += 1
                if building_totals:
                    building_rows = [
                        {"building": name, "device_count": values["devices"], "energy_kwh": values["total"]}
                        for name, values in sorted(building_totals.items(), key=lambda item: item[1]["total"], reverse=True)
                    ]
                    extra_tables.append({
                        "title": "Energy by Building",
                        "columns": [
                            {"key": "building", "label": "Building"},
                            {"key": "device_count", "label": "Devices included"},
                            {"key": "energy_kwh", "label": "Energy (kWh)"},
                        ],
                        "rows": building_rows,
                        "row_count": len(building_rows),
                        "truncated": False,
                        "source_tool": item.get("public_tool") or operation_id,
                        "operation": item.get("operation") or operation_id,
                    })
            title = "Building and Device Energy Breakdown" if is_energy_breakdown else "Device Energy Ranking"
            columns = [("rank", "#")]
            if is_energy_breakdown:
                columns.append(("building", "Building"))
            columns.extend([("device", "Device"), ("value", "Energy"), ("unit", "Unit"), ("status", "Status")])
        elif operation_id == "demand_peak_summary":
            source_rows = first_list(data, ("ranked_devices", "rankings", "devices", "rows", "peaks"))
            for rank, row in enumerate(source_rows[:100], 1):
                if not isinstance(row, dict):
                    continue
                peak = first_value(row, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw", "value"))
                if peak is None:
                    continue
                device_id = first_value(row, ("device_id", "id", "meter_id"))
                rows.append({
                    "rank": rank,
                    "device": first_value(row, ("device_name", "name", "meter_name")) or "Unnamed device",
                    "maximum_demand": peak,
                    "unit": row.get("unit") or "kW",
                    "peak_time": first_value(row, ("peak_time", "peak_at", "timestamp", "max_time", "max_at")),
                })
            if not rows:
                peak = first_value(data, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw"))
                if peak is not None:
                    rows.append({
                        "rank": 1,
                        "device": first_value(data, ("device_name", "name")) or "",
                        "maximum_demand": peak,
                        "unit": data.get("unit") or "kW",
                        "peak_time": first_value(data, ("peak_time", "peak_at", "timestamp", "max_time", "max_at")),
                    })
            title = "Maximum Demand Ranking" if len(rows) > 1 else "Maximum Demand"
            columns = [("rank", "#"), ("device", "Device"), ("maximum_demand", "Maximum demand"), ("unit", "Unit"), ("peak_time", "Peak time")]
        elif operation_id == "telemetry_metric_ranking":
            source_rows = first_list(data, ("ranked_devices", "rankings", "devices"))
            for rank, row in enumerate(source_rows[:100], 1):
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "rank": rank,
                    "device": row.get("device_name") or "Unnamed device",
                    "frequency": row.get("value"),
                    "unit": row.get("unit") or "Hz",
                    "highest_at": row.get("peak_time") or "",
                    "status": row.get("status") or "",
                    "data_status": row.get("data_status") or "",
                    "last_seen": row.get("last_seen") or "",
                    "samples": row.get("sample_count"),
                    "result": "Ranked",
                    "reason": "",
                })
            excluded_rows = first_list(data, ("excluded_devices",))
            for row in excluded_rows[:max(0, 100 - len(rows))]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "rank": "",
                    "device": row.get("device_name") or "Unnamed device",
                    "frequency": "",
                    "unit": "",
                    "highest_at": "",
                    "status": row.get("status") or "",
                    "data_status": row.get("data_status") or "",
                    "last_seen": row.get("last_seen") or "",
                    "samples": "",
                    "result": "Excluded",
                    "reason": row.get("reason") or "Readings unavailable",
                })
            title = "Frequency Ranking"
            columns = [("rank", "Rank"), ("device", "Device"), ("frequency", "Highest frequency"), ("unit", "Unit"), ("highest_at", "Occurred at"), ("samples", "Samples"), ("status", "Device status"), ("data_status", "Data freshness"), ("last_seen", "Last seen"), ("result", "Ranking result"), ("reason", "Exclusion reason")]
        elif operation_id == "latest_telemetry_snapshot":
            source_rows = first_list(data, ("series", "rows", "items", "readings", "results"))
            for series in source_rows[:100]:
                if not isinstance(series, dict):
                    continue
                points = series.get("points") if isinstance(series.get("points"), list) else [series]
                valid_points = [point for point in points if isinstance(point, dict) and chart_value(point, ("value", "reading", "last", "latest", "frequency", "frequency_hz")) is not None]
                if not valid_points:
                    continue
                point = max(valid_points, key=lambda p: parse_datetime(first_value(p, ("timestamp", "time", "datetime", "recorded_at"))) or datetime.min.replace(tzinfo=timezone.utc))
                rows.append({
                    "metric": first_value(series, ("metric", "source_metric", "name")) or "",
                    "value": chart_value(point, ("value", "reading", "last", "latest", "frequency", "frequency_hz")),
                    "unit": point.get("unit") or series.get("unit") or "",
                    "timestamp": first_value(point, ("timestamp", "time", "datetime", "recorded_at")) or "",
                })
            title = "Latest Telemetry Reading"
            columns = [("metric", "Metric"), ("value", "Value"), ("unit", "Unit"), ("timestamp", "Recorded at")]
        elif operation_id == "site_device_list":
            source_rows = first_list(data, ("devices", "rows", "items", "meters"))
            for row in source_rows[:100]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "device": row.get("device_name") or row.get("name") or row.get("device") or "",
                    "type": row.get("device_type") or row.get("type") or "",
                    "status": row.get("status") or row.get("data_status") or "",
                    "last_seen": row.get("last_seen"),
                })
            title = "Site Devices"
            columns = [("device", "Device"), ("type", "Type"), ("status", "Status"), ("last_seen", "Last seen")]
        elif operation_id == "active_alarm_summary":
            source_rows = first_list(data, ("alarms", "rows", "items"))
            for rank, row in enumerate(source_rows[:100], 1):
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "rank": rank,
                    "alarm": row.get("alarm_name") or row.get("name") or "Alarm",
                    "device": row.get("device_name") or "",
                    "severity": row.get("severity") or "",
                    "status": row.get("status") or "",
                    "started_at": row.get("started_at") or row.get("first_seen"),
                })
            title = "Active Alarms"
            columns = [("rank", "#"), ("alarm", "Alarm"), ("device", "Device"), ("severity", "Severity"), ("status", "Status"), ("started_at", "Started")]
        elif operation_id == "site_energy_summary":
            source_rows = first_list(data, ("buckets", "rows", "series", "data"))
            for row in source_rows[:100]:
                if not isinstance(row, dict):
                    continue
                value = chart_value(row, ("value", "kwh", "energy", "consumption"))
                if value is None:
                    continue
                rows.append({
                    "period": row.get("timestamp") or row.get("bucket") or row.get("date") or row.get("start") or "",
                    "value": value,
                    "unit": row.get("unit") or data.get("unit") or "kWh",
                })
            title = "Site Energy by Period"
            columns = [("period", "Period"), ("value", "Energy"), ("unit", "Unit")]
        elif operation_id == "site_metadata_summary":
            site = data.get("site") if isinstance(data.get("site"), dict) else data
            fields = (
                ("Site", first_value(site, ("name", "site_name"))),
                ("Timezone", first_value(site, ("timezone", "timezone_name"))),
                ("Location", first_value(site, ("location", "address", "city"))),
                ("Buildings", first_value(data, ("building_count", "total_buildings"))),
                ("Devices", first_value(data, ("device_count", "total_devices"))),
            )
            rows = [{"detail": label, "value": value} for label, value in fields if value not in (None, "")]
            title = "Site Overview"
            columns = [("detail", "Detail"), ("value", "Value")]

        if rows:
            display_timezone = arguments.get("timezone") or AI_DISPLAY_TIMEZONE
            for row in rows:
                for key in ("first_seen", "latest", "started_at", "last_seen", "period"):
                    if row.get(key):
                        row[key] = format_display_timestamp(row[key], display_timezone)
            tables.append({
                "title": title,
                "columns": [{"key": key, "label": label} for key, label in columns],
                "rows": rows,
                "row_count": data.get("row_count", len(rows)),
                "truncated": bool(data.get("truncated")),
                "source_tool": (
                    f"{item.get('public_tool')}.{item.get('operation')}"
                    if item.get("public_tool") and item.get("operation")
                    else item.get("public_tool") or operation_id
                ),
                "operation": item.get("operation") or operation_id,
                "time_range": {
                    key: format_display_timestamp(arguments.get(key), display_timezone)
                    for key in ("start_time", "end_time", "timezone")
                    if arguments.get(key) is not None
                },
            })
            tables.extend(extra_tables)
    return tables


def wants_available_parameters_query(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\b(?:other parameter|other parameters|available parameter|available parameters|what else|anything else)\b", lowered)
        and re.search(r"\b(?:device|parameter|metric|voltage|current|telemetry|available)\b", lowered)
    )


def answer_presentation(message: str, operation_ids: list[str], results: list[dict], charts: list[dict], tables: list[dict]) -> dict:
    """Classify the response so clients can present it as an answer, ranking, or report."""
    lowered = message.lower()
    result_operations = {str(item.get("operation_id") or "") for item in results}
    if "report_summary" in operation_ids or any(term in lowered for term in ("report", "management summary", "site health summary")):
        kind = "report"
    elif result_operations & {"demand_peak_summary", "telemetry_metric_ranking", "device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"} or re.search(r"\b(?:top|rank|ranking)\b", lowered):
        kind = "ranking"
    elif result_operations & {"active_alarm_summary", "alarm_frequency_summary", "alarm_detail_lookup"}:
        kind = "alarm_summary"
    elif result_operations & {"site_metadata_summary", "site_device_list", "meter_status_summary", "site_energy_summary"}:
        kind = "site_summary"
    elif result_operations & {"telemetry_timeseries", "power_quality_summary"}:
        kind = "trend"
    else:
        kind = "answer"
    return {
        "type": kind,
        "show_chart": bool(charts),
        "show_table": bool(tables),
        "sections": list(dict.fromkeys(
            str(table.get("title") or "Data") for table in tables if isinstance(table, dict)
        )),
    }


def historical_argument_clarification(error: ValueError, operation_id: str) -> str:
    detail = str(error).lower()
    if "device_id" in detail:
        return "I need a device name and time range before I can request a telemetry time series. Try: `Show the voltage trend for the 96S2 Freezer for the last 24 hours.`"
    if "alarm_id" in detail:
        return "I need the alarm ID before I can look up alarm details. Try: `Show details for alarm ID 123`."
    if "site_id" in detail:
        return "Choose a DaxView site before I access historical data."
    if operation_id == "telemetry_timeseries":
        return "Choose a device name and time range before I request telemetry time-series data."
    return "Choose a site and time range before I access DaxView historical data."


def readable_historical_errors(errors: list[dict]) -> str:
    if not errors:
        return ""
    parts = []
    for item in errors:
        operation = str(item.get("operation_id") or "DaxView tool")
        metric = item.get("metric")
        error = str(item.get("error") or "")
        label = f"{operation} for {metric}" if metric else operation
        if "INVALID_METRIC" in error:
            parts.append(f"{label} was rejected by DaxView because that metric name is not supported by the telemetry tool.")
        elif "UNSUPPORTED_METRIC" in error and operation == "demand_peak_summary":
            parts.append(
                "DaxView has no configured demand series for the selected device and period. "
                "Maximum demand must be calculated from measured active-power or demand readings in kW; "
                "current in amperes alone is not enough to determine kW without the required voltage, phase, and power-factor data."
            )
        elif "UNSUPPORTED_METRIC" in error:
            parts.append(f"{label} is not configured for this device or site in DaxView.")
        elif "NO_DATA" in error:
            parts.append(f"{label} returned no data for the requested scope and time range.")
        elif "BACKEND_UNAVAILABLE" in error:
            parts.append(
                f"{label} reached the DaxView MCP, but its EMS data backend returned HTTP 503. "
                "The MCP operation was accepted; retry after the DaxView data service is available."
            )
        elif "TURN_NOT_ACTIVE" in error:
            parts.append(
                f"{label} was selected, but DaxView rejected its authorization request because the turn was no longer active. "
                "This is a DaxView turn-lifecycle issue; retry the request and check that the turn remains active through authorization."
            )
        elif "UNAVAILABLE" in error:
            parts.append(f"{label} is currently unavailable from DaxView.")
        elif "404" in error or "NOT FOUND" in error.upper():
            parts.append("The DaxView MCP endpoint returned HTTP 404. Check that DAXVIEW_MCP_URL points to the deployed /mcp endpoint and that the MCP service is available.")
        else:
            parts.append(f"{label} could not be retrieved: {error}")
    return " ".join(parts)


def device_selection_prompt_from_result(mcp_result: dict) -> tuple[str, list[dict]]:
    data = historical_result_data(mcp_result)
    devices = first_list(data, ("devices", "rows", "items", "meters"))
    choices = []
    lines = [
        "Which device should I use for the max-demand check?",
        "Select a device by name, or type part of its name.",
        "",
        "Available devices:",
    ]
    for index, device in enumerate(devices[:100], 1):
        if not isinstance(device, dict):
            continue
        name = device.get("device_name") or device.get("name") or device.get("meter_name") or "Unnamed device"
        device_id = device.get("device_id") or device.get("id") or device.get("meter_id")
        if device_id is None:
            continue
        status = device.get("status") or device.get("connection_status") or device.get("state") or "unknown"
        device_type = device.get("device_type") or device.get("type") or device.get("model")
        detail = f"{name} ({str(status).title()}"
        if device_type:
            detail += f", {device_type}"
        detail += ")"
        if index <= 12:
            lines.append(f"{index}. {detail}")
        description = str(status)
        if device_type:
            description += f" | {device_type}"
        choices.append({"label": name, "value": str(device_id), "description": description})
    total = data.get("device_count") or data.get("meter_count") or data.get("row_count") or len(devices)
    if total and len(devices) > 12:
        lines.append(f"Showing the first 12 of {len(devices)} returned devices here; {len(choices)} are selectable.")
    if not choices:
        lines.append("No selectable devices were returned. Please check the site device list.")
    return "\n".join(lines), choices


def build_device_choice_response(turn_id: str, message: str, context: dict, request_id: str) -> dict:
    try:
        arguments = build_historical_arguments("site_device_list", context, message)
        result = call_authorized_internal_operation(turn_id, "site_device_list", arguments, request_id, message)
        prompt, choices = device_selection_prompt_from_result(result)
    except Exception as error:
        log_event("daxview_device_choice_failed", request_id=request_id, turn_id=turn_id, error=str(error))
        prompt = (
            "Which device should I use for the max-demand check? "
            "Reply with the device name, or a distinctive part of it."
        )
        choices = []
    window = re.search(
        r"\b(?:last|past)\s+\d{1,3}\s+days?\b|\b(?:today|yesterday|this week|last week|this month|last month)\b",
        message, re.IGNORECASE,
    )
    time_scope = window.group(0) if window else "the last 7 days"
    return {
        "provider": "daxview-question-filter",
        "reply": prompt,
        "fields": ["device_id"],
        "input_type": "select",
        "choices": choices,
        "submit_template": f"max demand for {{value}} for {time_scope}",
    }


def run_highest_demand_device_ranking(turn_id: str, message: str, context: dict, request_id: str) -> list[dict]:
    device_args = validate_tool_arguments("site_device_list", {
        "site_id": int(context["site_id"]),
        **({"building_id": int(context["building_id"])} if context.get("building_id") else {}),
        **({"device_id": int(context["device_id"])} if context.get("device_id") else {}),
        "limit": DEMAND_RANKING_MAX_DEVICES + 1,
    })
    normalized_args = build_public_calls("site_device_list", device_args, message)[0][1]
    device_plan = request_daxview_data_plan(turn_id, "device_data", normalized_args, request_id)
    device_result = call_authorized_historical_tool(
        "device_data", str(device_plan["authorization_id"]), normalized_args, request_id
    )
    device_data = historical_result_data(device_result)
    devices = first_list(device_data, ("devices", "rows", "items", "meters"))
    selectable = [device for device in devices if isinstance(device, dict) and (device.get("device_id") or device.get("id") or device.get("meter_id"))]
    total = device_data.get("device_count") or device_data.get("row_count") or len(selectable)
    if len(selectable) > DEMAND_RANKING_MAX_DEVICES or (str(total).isdigit() and int(total) > DEMAND_RANKING_MAX_DEVICES):
        prompt, choices = device_selection_prompt_from_result(device_result)
        return [{
            "operation_id": "site_device_list",
            "arguments": normalized_args,
            "result": device_result,
            "clarification": (
                f"The site has more than {DEMAND_RANKING_MAX_DEVICES} devices, so I need a smaller scope before ranking demand. "
                f"{prompt}"
            ),
            "choices": choices,
        }]
    ranked = []
    excluded = []
    for device in selectable:
        device_id = device.get("device_id") or device.get("id") or device.get("meter_id")
        device_name = device.get("device_name") or device.get("name") or f"Device {device_id}"
        try:
            device_context = {**context, "device_id": int(device_id)}
            metric_args = build_historical_arguments("telemetry_metric_catalog", device_context, message)
            public_metric_args = build_public_calls("telemetry_metric_catalog", metric_args, message)[0][1]
            metric_plan = request_daxview_data_plan(turn_id, "device_data", public_metric_args, request_id)
            metric_result = call_authorized_historical_tool(
                "device_data", str(metric_plan["authorization_id"]), public_metric_args, request_id
            )
            metric_data = historical_result_data(metric_result)
            metric_rows = first_list(metric_data, ("metrics", "rows", "items", "series"))
            eligible = []
            for metric_row in metric_rows:
                if not isinstance(metric_row, dict):
                    continue
                canonical = str(metric_row.get("metric") or "").strip().lower().replace(" ", "_")
                source_name = str(metric_row.get("source_metric") or metric_row.get("name") or "").strip().lower()
                unit = str(metric_row.get("unit") or "").strip().lower()
                status = str(metric_row.get("status") or "ok").strip().lower()
                if status not in {"ok", "available", "supported"} or metric_row.get("enabled") is False:
                    continue
                if canonical in {"demand", "maximum_demand", "max_demand"} or "maximum demand" in source_name:
                    eligible.append(("demand", metric_row))
                elif (canonical in {"active_power", "power", "active_power_total"} or "active power" in source_name) and unit in {"w", "kw", "mw"}:
                    eligible.append(("active_power", metric_row))
                elif canonical in {"energy", "consumption_delta"} or ("energy" in source_name and unit in {"wh", "kwh", "mwh"}):
                    eligible.append(("energy", metric_row))
            if not eligible:
                excluded.append({"device_id": int(device_id), "device_name": device_name, "reason": "No configured demand, active-power, or energy-delta metric"})
                continue

            # Prefer the meter's explicit demand series. Otherwise use measured
            # active power in a compatible unit; never infer kW from current.
            metric_kind, metric_row = next(
                (item for item in eligible if item[0] == "demand"),
                next((item for item in eligible if item[0] == "active_power"), eligible[0]),
            )
            source_metric = metric_row.get("metric") or metric_row.get("source_metric") or "demand"
            if metric_kind == "demand":
                peak_args = build_historical_arguments("demand_peak_summary", device_context, message)
                per_args = build_public_calls("demand_peak_summary", peak_args, message)[0][1]
                per_plan = request_daxview_data_plan(turn_id, "device_data", per_args, request_id)
                per_args = normalize_mcp_arguments("device_data", per_plan.get("arguments") or per_args)
                per_result = call_authorized_historical_tool("device_data", str(per_plan["authorization_id"]), per_args, request_id)
            else:
                is_energy_delta = metric_kind == "energy"
                peak_args = build_historical_arguments("telemetry_timeseries", {
                    **device_context,
                    "metric": str(source_metric),
                    "bucket": "15m",
                    "aggregation": "sum" if is_energy_delta else "mean",
                    "value_mode": "consumption_delta" if is_energy_delta else "measured",
                    "limit": 2000,
                }, message)
                per_args = build_public_calls("telemetry_timeseries", peak_args, message)[0][1]
                per_plan = request_daxview_data_plan(turn_id, "device_data", per_args, request_id)
                per_args = normalize_mcp_arguments("device_data", per_plan.get("arguments") or per_args)
                per_result = call_authorized_historical_tool("device_data", str(per_plan["authorization_id"]), per_args, request_id)

            per_data = historical_result_data(per_result)
            if metric_kind == "active_power":
                series_rows = first_list(per_data, ("series", "rows", "items"))
                has_power_unit = any(
                    isinstance(row, dict) and str(row.get("unit") or "").strip().lower() in {"w", "kw", "mw"}
                    for row in series_rows
                )
                if not has_power_unit:
                    excluded.append({"device_id": int(device_id), "device_name": device_name, "reason": "Active-power values did not include a recognized W/kW/MW unit"})
                    continue
            summary = per_data.get("summary") if isinstance(per_data.get("summary"), dict) else per_data
            peak = first_value(summary, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw"))
            peak_time = first_value(summary, ("peak_time", "peak_at", "timestamp", "max_time", "max_at"))
            calculated = None
            if peak is None:
                calculated = (
                    calculate_demand_peak_from_energy(per_result, per_args)
                    if metric_kind == "energy"
                    else calculate_demand_peak_from_telemetry(per_result, per_args)
                )
                if calculated:
                    peak_data = calculated["data"]
                    peak = peak_data["peak_kw"]
                    peak_time = peak_data.get("peak_time")
            if peak is None:
                excluded.append({"device_id": int(device_id), "device_name": device_name, "reason": "No numeric demand/power readings in the requested period"})
                continue
            ranked.append({
                "device_id": int(device_id),
                "device_name": device_name,
                "status": device.get("status") or device.get("connection_status"),
                "peak_kw": float(peak),
                "peak_time": format_display_timestamp(peak_time) if peak_time else None,
                "metric": source_metric,
                "sample_count": (calculated or {}).get("data", {}).get("sample_count") or first_value(summary, ("sample_count", "returned_count")),
                "coverage_percent": first_value(summary, ("coverage_percent",)),
                "truncated": bool(per_data.get("truncated")) or any(
                    isinstance(series, dict) and bool(series.get("truncated"))
                    for series in first_list(per_data, ("series",))
                ),
                "result": per_result,
                "arguments": per_args,
            })
        except Exception as error:
            log_event("demand_device_rank_step_failed", request_id=request_id, device_id=device_id, error=str(error))
            excluded.append({"device_id": int(device_id), "device_name": device_name, "reason": "Metric discovery or readings unavailable"})
    ranked.sort(key=lambda row: row["peak_kw"], reverse=True)
    debug_trace_event(
        "demand_site_scan_debug",
        request_id=request_id,
        device_count=len(selectable),
        eligible_count=len(ranked),
        excluded_count=len(excluded),
        excluded_devices=excluded[:50],
        range=context.get("_time_window"),
    )
    return [{
        "operation_id": "demand_peak_summary",
        "arguments": {key: context[key] for key in ("site_id", "building_id", "_time_window") if key in context},
        "result": {"structuredContent": {"data": {
            "ranked_devices": ranked,
            "device_count": len(selectable),
            "eligible_device_count": len(ranked),
            "excluded_devices": excluded,
            "truncated_device_scan": bool(device_data.get("truncated")) or len(selectable) > DEMAND_RANKING_MAX_DEVICES,
            "truncated_readings": any(row.get("truncated") for row in ranked),
        }}},
    }]


def run_device_metric_ranking(turn_id: str, message: str, context: dict, request_id: str) -> list[dict]:
    """Rank configured site devices by their highest measured frequency in Hz."""
    metric = str(context.get("metric") or requested_metric(message) or "frequency")
    unit_factors = {"hz": 1.0, "khz": 1000.0}
    if metric != "frequency":
        raise ValueError(f"Device ranking is not configured for {metric}.")

    def authorized_call(tool_name: str, arguments: dict) -> tuple[dict, dict]:
        plan = request_daxview_data_plan(turn_id, tool_name, arguments, request_id)
        authorization_id = plan.get("authorization_id")
        if not authorization_id:
            raise RuntimeError(f"No authorization was issued for {tool_name}.{arguments.get('operation')}.")
        planned = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else arguments
        normalized = normalize_mcp_arguments(tool_name, planned)
        if normalized.get("operation") != arguments.get("operation"):
            raise RuntimeError("DaxView changed the authorized MCP operation")
        return call_authorized_historical_tool(tool_name, str(authorization_id), normalized, request_id), normalized

    base = {key: context[key] for key in ("site_id", "building_id") if context.get(key) is not None}
    list_args = {"operation": "list", **base, "limit": DEMAND_RANKING_MAX_DEVICES + 1}
    device_result, list_args = authorized_call("device_data", list_args)
    device_data = historical_result_data(device_result)
    devices = first_list(device_data, ("devices", "rows", "items", "meters"))
    selectable = [row for row in devices if isinstance(row, dict) and first_value(row, ("device_id", "id", "meter_id")) is not None]
    total = device_data.get("device_count") or device_data.get("total_count") or device_data.get("row_count") or len(selectable)
    if len(selectable) > DEMAND_RANKING_MAX_DEVICES or (str(total).isdigit() and int(total) > DEMAND_RANKING_MAX_DEVICES):
        prompt, choices = device_selection_prompt_from_result(device_result)
        return [{
            "operation_id": "telemetry_metric_ranking",
            "arguments": list_args,
            "result": device_result,
            "clarification": f"There are more than {DEMAND_RANKING_MAX_DEVICES} devices in scope. Narrow the site or building before ranking frequency. {prompt}",
            "choices": choices,
        }]

    time_window = context.get("_time_window") or requested_historical_range(message)
    ranked = []
    excluded = []
    for device in selectable:
        device_id = first_value(device, ("device_id", "id", "meter_id"))
        device_name = first_value(device, ("device_name", "name", "meter_name")) or f"Device {device_id}"
        try:
            metrics_args = {"operation": "metrics", **base, "device_id": int(device_id), "limit": 100}
            metrics_result, _ = authorized_call("device_data", metrics_args)
            metric_rows = first_list(historical_result_data(metrics_result), ("metrics", "rows", "items", "series"))
            match = next((row for row in metric_rows if isinstance(row, dict) and (
                str(row.get("metric") or "").strip().lower().replace(" ", "_") == metric
                or "frequency" in str(row.get("source_metric") or row.get("name") or "").lower()
            ) and row.get("enabled") is not False and str(row.get("status") or "ok").lower() in {"ok", "available", "supported"}), None)
            if not match:
                excluded.append({"device_id": int(device_id), "device_name": device_name, "status": device.get("status") or device.get("data_status"), "last_seen": device.get("last_seen"), "data_status": device.get("data_status"), "reason": "No configured frequency metric"})
                continue

            metric_name = str(match.get("source_metric") or match.get("metric") or metric)
            start_dt = parse_datetime(time_window.get("start") or time_window.get("start_time"))
            end_dt = parse_datetime(time_window.get("end") or time_window.get("end_time"))
            range_duration = (end_dt - start_dt) if start_dt and end_dt else timedelta(days=8)
            request_args = {
                **base,
                "device_id": int(device_id),
                "metric": metric_name,
                "start_time": time_window.get("start") or time_window.get("start_time"),
                "end_time": time_window.get("end") or time_window.get("end_time"),
                "timezone": time_window.get("timezone") or "Asia/Kuala_Lumpur",
                "bucket": "raw" if range_duration <= timedelta(days=7) else "1h",
                "aggregation": "max",
                "value_mode": "measured",
                "limit": 1000,
            }
            tool_name, public_args = build_public_calls("telemetry_metric_ranking", request_args, message)[0]
            values_result, normalized = authorized_call(tool_name, public_args)
            values_data = historical_result_data(values_result)
            series_rows = first_list(values_data, ("series", "rows", "items", "buckets", "points"))
            configured_unit = str(match.get("unit") or "").strip().lower()
            candidates = []
            for series_row in series_rows:
                if not isinstance(series_row, dict):
                    continue
                points = series_row.get("points") if isinstance(series_row.get("points"), list) else [series_row]
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    value = chart_value(point, ("value", "reading", "average", "max", "maximum", "frequency", "frequency_hz"))
                    unit = str(point.get("unit") or series_row.get("unit") or configured_unit).strip().lower()
                    factor = unit_factors.get(unit)
                    if value is None or factor is None:
                        continue
                    timestamp = first_value(point, ("timestamp", "time", "bucket", "date", "start_time"))
                    candidates.append((float(value) * factor, timestamp))
            if not candidates:
                excluded.append({"device_id": int(device_id), "device_name": device_name, "status": device.get("status") or device.get("data_status"), "last_seen": device.get("last_seen"), "data_status": device.get("data_status"), "reason": "No numeric readings with a recognized Hz or kHz unit"})
                continue

            positive_candidates = [item for item in candidates if item[0] > 0]
            if not positive_candidates:
                excluded.append({"device_id": int(device_id), "device_name": device_name, "status": device.get("status") or device.get("data_status"), "last_seen": device.get("last_seen"), "data_status": device.get("data_status"), "reason": "No positive frequency readings"})
                continue
            highest, peak_time = max(positive_candidates, key=lambda item: item[0])
            ranked.append({
                "device_id": int(device_id),
                "device_name": device_name,
                "value": round(highest, 4),
                "unit": "Hz",
                "peak_time": format_display_timestamp(peak_time) if peak_time else None,
                "sample_count": len(candidates),
                "status": device.get("status") or device.get("data_status"),
                "data_status": device.get("data_status"),
                "last_seen": device.get("last_seen"),
                "age_seconds": device.get("age_seconds"),
                "expected_interval_seconds": device.get("expected_interval_seconds"),
                "metric": metric_name,
                "truncated": bool(values_data.get("truncated")) or any(bool(row.get("truncated")) for row in series_rows if isinstance(row, dict)),
            })
        except Exception as error:
            log_event("metric_device_rank_step_failed", request_id=request_id, device_id=device_id, metric=metric, error=str(error))
            excluded.append({"device_id": int(device_id), "device_name": device_name, "status": device.get("status") or device.get("data_status"), "last_seen": device.get("last_seen"), "data_status": device.get("data_status"), "reason": "Frequency readings unavailable"})

    ranked.sort(key=lambda row: row["value"], reverse=True)
    data = {
        "ranked_devices": ranked,
        "device_count": len(selectable),
        "ranked_count": len(ranked),
        "excluded_devices": excluded,
        "truncated_device_scan": bool(device_data.get("truncated")),
        "truncated_readings": any(row.get("truncated") for row in ranked),
        "metric": metric,
        "unit": "Hz",
    }
    return [{
        "operation_id": "telemetry_metric_ranking",
        "arguments": {**base, **time_window, "metric": metric},
        "result": {"structuredContent": {"data": data}},
    }]


class PredictionSourceError(RuntimeError):
    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(f"Historical energy data is unavailable: {error_code}")


def run_authorized_energy_prediction(turn_id: str, arguments: dict, request_id: str) -> dict:
    forecast_start = parse_datetime(arguments.get("forecast_start")) if arguments.get("forecast_start") else datetime.now(timezone.utc)
    training_days = int(arguments.get("training_days") or 35)
    forecast_end = parse_datetime(arguments.get("forecast_end")) if arguments.get("forecast_end") else forecast_start + timedelta(days=7)
    forecast_days = max(1, min(14, (forecast_end.date() - forecast_start.date()).days or 1))
    source_args = {
        key: arguments[key]
        for key in ("site_id", "building_id", "timezone")
        if key in arguments
    }
    source_args["start"] = (forecast_start - timedelta(days=training_days)).isoformat()
    source_args["end"] = forecast_start.isoformat()
    source_args["bucket"] = "day"
    try:
        source = call_authorized_internal_operation(turn_id, "site_energy_summary", source_args, request_id)
    except RuntimeError as error:
        raise PredictionSourceError(str(error)) from error
    structured = mcp_structured_result(source)
    if structured.get("status") == "error" or source.get("isError"):
        raise PredictionSourceError(str(structured.get("error_code") or "MCP_TOOL_ERROR"))
    prediction = predict_daily_energy(
        historical_result_data(source, ("buckets", "rows", "series", "data")),
        forecast_days,
        arguments.get("timezone") or "Asia/Kuala_Lumpur",
    )
    log_event(
        "energy_prediction_completed", request_id=request_id, tool="energy_forecast",
        source_tool="site_energy_summary", observed_days=prediction["observed_days"],
        coverage_percent=prediction["coverage_percent"], backtest_mae=prediction["backtest_mae"],
    )
    debug_trace_event("energy_prediction_debug", request_id=request_id, prediction=prediction)
    return {"structuredContent": {"data": prediction}}


def demand_peak_needs_manual_fallback(mcp_result: dict) -> bool:
    issue = result_problem(mcp_structured_result(mcp_result)) or result_problem(mcp_result)
    if issue and "NO_DATA" in str(issue).upper():
        return True
    data = historical_result_data(mcp_result)
    peak_keys = ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw")
    if first_value(data, peak_keys) is not None:
        return False
    if calculate_demand_peak_from_telemetry(mcp_result, {}):
        return False
    rows = first_list(data, ("rows", "items", "results", "data", "series", "buckets"))
    return not any(isinstance(row, dict) and first_value(row, peak_keys) is not None for row in rows)


def calculate_demand_peak_from_telemetry(mcp_result: dict, arguments: dict) -> dict | None:
    data = historical_result_data(mcp_result, ("rows", "items", "results", "data", "series", "buckets", "points"))
    rows = first_list(data, ("rows", "items", "results", "data", "series", "buckets", "points"))
    candidates = []
    value_keys = (
        "demand_kw", "kw", "value_kw", "reading_kw", "value", "average",
        "avg", "max", "maximum", "reading", "demand",
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        points = row.get("points") if isinstance(row.get("points"), list) else [row]
        for point in points:
            if not isinstance(point, dict):
                continue
            value = chart_value(point, value_keys)
            if value is None:
                continue
            unit = str(point.get("unit") or row.get("unit") or data.get("unit") or "kw").strip().lower()
            factor = {"w": 0.001, "kw": 1.0, "mw": 1000.0}.get(unit)
            if factor is not None:
                value *= factor
            candidates.append((value, {**row, **point, "unit": "kW"}))
    if not candidates:
        return None
    peak, row = max(candidates, key=lambda item: item[0])
    return {
        "status": "ok",
        "data": {
            "peak_kw": round(float(peak), 4),
            "peak_time": first_value(row, ("timestamp", "time", "bucket", "date", "start_time", "started_at")),
            "unit": "kW",
            "source": "deterministic maximum of device_data.values demand points",
            "sample_count": len(candidates),
            "calculation": "max(demand telemetry values)",
            "start_time": arguments.get("start_time"),
            "end_time": arguments.get("end_time"),
            "device_id": arguments.get("device_id"),
            "site_id": arguments.get("site_id"),
            "building_id": arguments.get("building_id"),
        },
    }


def calculate_demand_peak_from_energy(mcp_result: dict, arguments: dict) -> dict | None:
    """Convert known interval-consumption deltas into interval-average kW."""
    bucket_hours = {
        "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
        "1h": 1.0, "hour": 1.0,
    }.get(str(arguments.get("bucket") or "15m").lower())
    if not bucket_hours or bucket_hours <= 0:
        return None
    data = historical_result_data(mcp_result, ("series", "rows", "items", "buckets", "points"))
    rows = first_list(data, ("series", "rows", "items", "buckets", "points"))
    candidates = []
    energy_to_kwh = {"wh": 0.001, "kwh": 1.0, "mwh": 1000.0}
    for row in rows:
        if not isinstance(row, dict):
            continue
        points = row.get("points") if isinstance(row.get("points"), list) else [row]
        for point in points:
            if not isinstance(point, dict):
                continue
            value = chart_value(point, ("value", "energy_kwh", "consumption_kwh", "reading", "sum"))
            unit = str(point.get("unit") or row.get("unit") or data.get("unit") or "").strip().lower()
            factor = energy_to_kwh.get(unit)
            if value is None or factor is None:
                continue
            energy_kwh = float(value) * factor
            demand_kw = energy_kwh / bucket_hours
            timestamp = first_value(point, ("timestamp", "time", "bucket", "date", "start_time"))
            candidates.append((demand_kw, timestamp))
    if not candidates:
        return None
    peak_kw, peak_time = max(candidates, key=lambda item: item[0])
    return {
        "status": "ok",
        "data": {
            "peak_kw": round(peak_kw, 4),
            "peak_time": peak_time,
            "unit": "kW",
            "source": "interval consumption delta divided by interval duration",
            "calculation": f"max(interval kWh / {bucket_hours:g} h)",
            "sample_count": len(candidates),
            "bucket": arguments.get("bucket") or "15m",
            "value_mode": "consumption_delta",
            "start_time": arguments.get("start_time"),
            "end_time": arguments.get("end_time"),
            "device_id": arguments.get("device_id"),
            "site_id": arguments.get("site_id"),
            "truncated": bool(data.get("truncated")) or any(
                isinstance(row, dict) and bool(row.get("truncated")) for row in rows
            ),
        },
    }


def try_manual_demand_peak_fallback(turn_id: str, arguments: dict, request_id: str) -> dict:
    telemetry_args = dict(arguments)
    telemetry_args["metric"] = "demand"
    telemetry_args["phase"] = telemetry_args.get("phase") or "all"
    telemetry_args["bucket"] = telemetry_args.get("bucket") or "1h"
    telemetry_args["aggregation"] = telemetry_args.get("aggregation") or "auto"
    telemetry_args["limit"] = int(telemetry_args.get("limit") or 1000)
    telemetry_result = call_authorized_internal_operation(turn_id, "telemetry_timeseries", telemetry_args, request_id)
    normalized = build_public_calls("telemetry_timeseries", telemetry_args)[0][1]
    calculated = calculate_demand_peak_from_telemetry(telemetry_result, normalized)
    if not calculated:
        raise RuntimeError("telemetry_timeseries returned no demand values that can be used for manual max-demand calculation")
    return {
        "operation_id": "demand_peak_summary",
        "arguments": arguments,
        "result": {"structuredContent": calculated},
        "fallback": {
            "source_tool": "telemetry_timeseries",
            "arguments": {key: value for key, value in normalized.items() if key != "authorization_id"},
        },
    }


def try_top_consumers_breakdown_fallback(turn_id: str, arguments: dict, request_id: str) -> dict:
    breakdown_args = {
        key: arguments[key]
        for key in ("site_id", "building_id", "start", "end", "start_time", "end_time", "timezone", "limit")
        if key in arguments
    }
    if "start" in breakdown_args:
        breakdown_args["start_time"] = breakdown_args.pop("start")
    if "end" in breakdown_args:
        breakdown_args["end_time"] = breakdown_args.pop("end")
    breakdown_args["limit"] = int(breakdown_args.get("limit") or 20)
    result = call_authorized_internal_operation(turn_id, "device_energy_breakdown", breakdown_args, request_id)
    normalized = build_public_calls("device_energy_breakdown", breakdown_args)[0][1]
    return {
        "operation_id": "device_energy_ranking",
        "arguments": {**normalized, "limit": int(normalized.get("limit") or breakdown_args["limit"])},
        "result": result,
        "fallback": {
            "source_tool": "device_energy_breakdown",
            "arguments": {key: value for key, value in normalized.items() if key != "authorization_id"},
        },
    }


def run_daxview_integration_turn(turn_id: str, message: str, context: dict, request_id: str, session_id: str) -> dict:
    supplied_demand = supplied_demand_readings_answer(message)
    if supplied_demand:
        debug_trace_event(
            "local_ems_calculation_debug",
            request_id=request_id,
            calculation="maximum_demand_from_supplied_readings",
            source="user_supplied_interval_average_kw",
            peak_kw=supplied_demand["peak_kw"],
            sample_count=len(supplied_demand["readings_kw"]),
            mcp_used=False,
        )
        return {
            "provider": "local-ems-calculation",
            "reply": supplied_demand["reply"],
            "charts": [],
            "debug_charts": [],
            "debug_tables": [],
            "resolved_plan": resolved_plan_for_result("ok", [], context),
        }
    local_calculation = connected_load_calculation_answer(message)
    if local_calculation:
        debug_trace_event(
            "local_ems_calculation_debug",
            request_id=request_id,
            calculation="connected_load_total",
            source="user_supplied_values",
            total_kw=local_calculation["total_kw"],
            load_count=len(local_calculation["loads"]),
            mcp_used=False,
        )
        return {
            "provider": "local-ems-calculation",
            "reply": local_calculation["reply"],
            "charts": [],
            "debug_charts": [],
            "debug_tables": [],
            "resolved_plan": resolved_plan_for_result("ok", [], context),
        }
    clarification = metric_clarification(message)
    if clarification:
        return {"provider": "daxview-question-filter", "reply": clarification, "resolved_plan": resolved_plan_for_result("clarification", [], context)}
    if is_historical_readings_clarification_request(message) and not context.get("_clarification_reply"):
        debug_trace_event(
            "daxview_clarification_intent_debug",
            request_id=request_id,
            intent="historical_telemetry_clarification",
            planner_skipped=True,
            mcp_skipped=True,
        )
        return build_incomplete_request_clarification(message, context, request_id, turn_id=turn_id)
    operation_ids, context, planner_plan = choose_historical_operations(message, context, request_id)
    debug_trace_event(
        "daxview_tool_selection_debug",
        request_id=request_id,
        turn_id=turn_id,
        operation_ids=operation_ids,
        message=message,
    )
    if context.get("_planner_clarification"):
        return {
            "provider": "daxview-question-filter",
            "reply": str(context["_planner_clarification"]),
            "resolved_plan": planner_plan or resolved_plan_for_result("clarification", [], context),
        }
    if context.get("_clarification_result"):
        return context["_clarification_result"]
    if operation_ids and daxview_clarification_needed(message, operation_ids) and not context.get("_clarification_reply"):
        return build_incomplete_request_clarification(message, context, request_id, turn_id=turn_id)
    if not operation_ids:
        result = langchain_chat_response(message, request_id, session_id)
        result["resolved_plan"] = resolved_plan_for_result("ok", [], context)
        return result
    context = dict(context)
    if context.get("_clarification_device_unresolved"):
        return build_incomplete_request_clarification(message, context, request_id, turn_id=turn_id)
    if context.get("_clarification_device_ref") and not context.get("device_id") and context.get("site_id"):
        match = resolve_named_device_reference(turn_id, context["_clarification_device_ref"], context, request_id)
        if match.get("status") == "matched":
            context["device_id"] = match["device_id"]
            context["_device_name"] = match["device_name"]
        else:
            return build_incomplete_request_clarification(message, context, request_id, turn_id=turn_id)
    if requested_device_id(message):
        context["device_id"] = requested_device_id(message)
    if wants_named_device_metric_query(message) and not context.get("device_id"):
        device_match = resolve_named_device_reference(turn_id, message, context, request_id)
        if device_match.get("status") != "matched":
            return {
                "provider": "daxview-question-filter",
                "reply": device_match.get("reply") or "Which device name should I use?",
                "required_fields": ["device or meter"],
                "clarification_state": {
                    "status": "waiting",
                    "intent": "historical_device_readings",
                    "original_message": message,
                    "required_fields": ["device or meter"],
                    "known_slots": {
                        **({"metric": question_metrics(message)[0]} if question_metrics(message) else {}),
                    },
                    "turn_number": 2,
                },
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context),
            }
        context["device_id"] = device_match["device_id"]
        context["_device_name"] = device_match["device_name"]
        save_resolved_turn_context(turn_id, context)
    if "demand_peak_summary" in operation_ids and not context.get("device_id") and not wants_highest_demand_device(message):
        result = build_device_choice_response(turn_id, message, context, request_id)
        result["resolved_plan"] = resolved_plan_for_result("clarification", operation_ids, context)
        return result
    context.setdefault("_time_window", completed_daily_range(message) or requested_historical_range(message))
    save_resolved_turn_context(turn_id, context)
    if "telemetry_metric_ranking" in operation_ids:
        metric_results = run_device_metric_ranking(turn_id, message, context, request_id)
        if metric_results and metric_results[0].get("clarification"):
            return {
                "provider": "daxview-question-filter",
                "reply": metric_results[0]["clarification"],
                "input_type": "select",
                "choices": metric_results[0].get("choices") or [],
                "submit_template": "rank frequency across the selected scope for the same period",
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context, metric_results),
            }
        deterministic_reply = summarize_historical_answers(message, metric_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, metric_results, deterministic_reply, request_id)
        charts = build_charts_from_historical_results(metric_results, message=message)
        tables = build_tables_from_historical_results(metric_results, message)
        presentation = answer_presentation(message, operation_ids, metric_results, charts, tables)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": charts,
            "tables": tables,
            "presentation": presentation,
            "debug_charts": charts,
            "debug_tables": tables,
            "resolved_plan": resolved_plan_for_result("ok", operation_ids, context, metric_results),
        }
    if wants_sitewide_energy_total(message):
        operation_results = execute_authorized_internal_operation(
            turn_id, "site_energy_summary", build_historical_arguments("site_energy_summary", context, message), request_id, message
        )
        deterministic_reply = summarize_historical_answers(message, operation_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, operation_results, deterministic_reply, request_id)
        tables = build_tables_from_historical_results(operation_results, message)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": [],
            "tables": tables,
            "presentation": answer_presentation(message, ["site_energy_summary"], operation_results, [], tables),
            "debug_charts": [],
            "debug_tables": tables,
            "resolved_plan": resolved_plan_for_result("ok", ["site_energy_summary"], context, operation_results),
        }
    if "demand_peak_summary" in operation_ids:
        if not context.get("device_id") and not wants_highest_demand_device(message):
            result = build_device_choice_response(turn_id, message, context, request_id)
            result["resolved_plan"] = resolved_plan_for_result("clarification", operation_ids, context)
            return result
        demand_results = run_highest_demand_device_ranking(turn_id, message, context, request_id)
        if demand_results and demand_results[0].get("clarification"):
            return {
                "provider": "daxview-question-filter",
                "reply": demand_results[0]["clarification"],
                "input_type": "select",
                "choices": demand_results[0].get("choices") or [],
                "submit_template": "max demand for {value} for the same period",
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context, demand_results),
            }
        deterministic_reply = summarize_historical_answers(message, demand_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, demand_results, deterministic_reply, request_id)
        charts = build_charts_from_historical_results(demand_results, message=message)
        tables = build_tables_from_historical_results(demand_results, message)
        presentation = answer_presentation(message, operation_ids, demand_results, charts, tables)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": charts,
            "tables": tables,
            "presentation": presentation,
            "debug_charts": charts,
            "debug_tables": tables,
            "resolved_plan": resolved_plan_for_result("ok", operation_ids, context, demand_results),
        }
    if wants_highest_demand_device(message) and not context.get("device_id"):
        ranked_results = run_highest_demand_device_ranking(turn_id, message, context, request_id)
        if ranked_results and ranked_results[0].get("clarification"):
            return {
                "provider": "daxview-question-filter",
                "reply": ranked_results[0]["clarification"],
                "input_type": "select",
                "choices": ranked_results[0].get("choices") or [],
                "submit_template": "max demand for {value} for the same period",
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context, ranked_results),
            }
        deterministic_reply = summarize_historical_answers(message, ranked_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, ranked_results, deterministic_reply, request_id)
        charts = build_charts_from_historical_results(ranked_results, message=message)
        tables = build_tables_from_historical_results(ranked_results, message)
        presentation = answer_presentation(message, operation_ids, ranked_results, charts, tables)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": charts,
            "tables": tables,
            "presentation": presentation,
            "debug_charts": charts,
            "debug_tables": tables,
            "resolved_plan": resolved_plan_for_result("ok", ["demand_peak_summary"], context, ranked_results),
        }
    if "demand_peak_summary" in operation_ids and context.get("device_id"):
        device_results = run_highest_demand_device_ranking(turn_id, message, context, request_id)
        deterministic_reply = summarize_historical_answers(message, device_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, device_results, deterministic_reply, request_id)
        charts = build_charts_from_historical_results(device_results, message=message)
        tables = build_tables_from_historical_results(device_results, message)
        presentation = answer_presentation(message, operation_ids, device_results, charts, tables)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": charts,
            "tables": tables,
            "presentation": presentation,
            "debug_charts": charts,
            "debug_tables": tables,
            "resolved_plan": resolved_plan_for_result("ok", ["demand_peak_summary"], context, device_results),
        }
    results = []
    errors = []
    steps = []
    for operation_id in operation_ids:
        metrics = question_metrics(message) if operation_id == "telemetry_timeseries" else []
        steps.extend((operation_id, metric) for metric in metrics or [None])
    for operation_id, metric in steps:
        try:
            step_context = {**context, "metric": metric} if metric else context
            arguments = build_historical_arguments(operation_id, step_context, message)
            if context.get("_device_name"):
                arguments["_device_name"] = context["_device_name"]
            debug_trace_event(
                "daxview_tool_arguments_debug",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                metric=metric,
                arguments=redact_debug_value(arguments),
            )
        except ValueError as error:
            return {
                "provider": "daxview-question-filter",
                "reply": historical_argument_clarification(error, operation_id),
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context),
            }
        if operation_id == "energy_forecast":
            try:
                result = run_authorized_energy_prediction(turn_id, arguments, request_id)
                results.append({"operation_id": operation_id, "arguments": arguments, "result": result})
            except PredictionSourceError as error:
                log_event("energy_prediction_unavailable", request_id=request_id, source_tool="site_energy_summary", error_code=error.error_code)
                if len(operation_ids) == 1:
                    return {
                        "provider": "energy-prediction",
                        "reply": "I couldn't calculate a forecast because DaxView did not return historical energy readings for this site. Please retry after the data service is available.",
                        "charts": [],
                        "resolved_plan": resolved_plan_for_result("no_data", operation_ids, context, results, errors),
                    }
                errors.append({"operation_id": operation_id, "error": str(error)})
            except ValueError as error:
                log_event("energy_prediction_rejected", request_id=request_id, reason=str(error))
                if len(operation_ids) == 1:
                    return {"provider": "energy-prediction", "reply": str(error), "charts": [], "resolved_plan": resolved_plan_for_result("error", operation_ids, context, results, errors)}
                errors.append({"operation_id": operation_id, "error": str(error)})
            except Exception as error:
                errors.append({"operation_id": operation_id, "error": str(error)})
                log_event("daxview_tool_step_failed", request_id=request_id, turn_id=turn_id, operation_id=operation_id, error=str(error))
                if len(operation_ids) == 1:
                    raise
            continue
        try:
            operation_results = execute_authorized_internal_operation(turn_id, operation_id, arguments, request_id, message)
            result = operation_results[0]["result"] if operation_results else {}
            if AI_MCP_FALLBACKS_ENABLED and operation_id == "demand_peak_summary" and demand_peak_needs_manual_fallback(result):
                try:
                    results.append(try_manual_demand_peak_fallback(turn_id, arguments, request_id))
                    continue
                except Exception as fallback_error:
                    errors.append({"operation_id": "telemetry_timeseries", "metric": "demand", "error": str(fallback_error)})
            results.extend(operation_results)
        except Exception as error:
            if AI_MCP_FALLBACKS_ENABLED and operation_id in {"device_energy_ranking", "telemetry_top_consumers"}:
                try:
                    results.append(try_top_consumers_breakdown_fallback(turn_id, arguments, request_id))
                    continue
                except Exception as fallback_error:
                    errors.append({"operation_id": "device_energy_breakdown", "error": str(fallback_error)})
            if AI_MCP_FALLBACKS_ENABLED and operation_id == "demand_peak_summary":
                try:
                    results.append(try_manual_demand_peak_fallback(turn_id, arguments, request_id))
                    continue
                except Exception as fallback_error:
                    errors.append({"operation_id": "telemetry_timeseries", "metric": "demand", "error": str(fallback_error)})
            metric_label = f":{metric}" if metric else ""
            errors.append({"operation_id": operation_id, "metric": metric, "error": str(error)})
            log_event("daxview_tool_step_failed", request_id=request_id, turn_id=turn_id, operation_id=operation_id, metric=metric, error=str(error))
            if len(operation_ids) == 1 and len(steps) == 1:
                break
    if not results:
        error_detail = "; ".join(
            f"{item['operation_id']}{':' + str(item.get('metric')) if item.get('metric') else ''}: {item['error']}"
            for item in errors
        )
        return {
            "provider": "daxview-historical-mcp",
            "reply": readable_historical_errors(errors) or error_detail or (
                "No DaxView MCP tool returned data."
            ),
            "charts": [],
            "debug_charts": [],
            "resolved_plan": resolved_plan_for_result("error", operation_ids, context, results, errors),
        }
    deterministic_reply = summarize_historical_answers(message, results, request_id)
    if errors:
        failed_tools = readable_historical_errors(errors)
        deterministic_reply = f"{deterministic_reply}\n\nUnavailable detail: {failed_tools}"
    refined_reply = refine_historical_answer_with_model(message, results, deterministic_reply, request_id)
    if errors and "Unavailable detail:" not in refined_reply:
        refined_reply += f"\n\nUnavailable detail: {failed_tools}"
    charts = build_charts_from_historical_results(results, message=message)
    preview_charts = charts if charts else build_charts_from_historical_results(results, preview=True, message=message)
    tables = build_tables_from_historical_results(results, message)
    presentation = answer_presentation(message, operation_ids, results, charts, tables)
    debug_trace_event(
        "ai_final_response_debug",
        request_id=request_id,
        model=CHAT_MODEL if AI_REFINE_MCP_WITH_MODEL else "deterministic",
        answer=refined_reply,
        charts=preview_charts,
        tables=tables,
    )
    return {
        "provider": "daxview-historical-mcp",
        "reply": refined_reply,
        "charts": charts,
        "tables": tables,
        "presentation": presentation,
        "debug_charts": preview_charts,
        "debug_tables": tables,
        "resolved_plan": resolved_plan_for_result("ok" if not errors else "no_data", operation_ids, context, results, errors),
    }


def save_resolved_turn_context(turn_id: str, context: dict) -> None:
    if not DATABASE_URL:
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE daxview_turns SET context = %s WHERE id = %s", (json.dumps(context), turn_id))
        conn.commit()


def save_turn_memory(turn_id: str, reply: str, resolved_plan: dict) -> None:
    if not DATABASE_URL:
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE daxview_turns
                SET assistant_reply = %s, resolved_plan = %s
                WHERE id = %s
                """,
                (reply, json.dumps(resolved_plan or {}), turn_id),
            )
        conn.commit()


def save_turn_clarification_state(turn_id: str, state: dict) -> None:
    if not DATABASE_URL:
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE daxview_turns SET clarification_state = %s WHERE id = %s", (json.dumps(state), turn_id))
        conn.commit()


def get_pending_clarification(conversation_id: str, identity: dict, turn_id: str) -> dict | None:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.user_message, t.request_id, t.context, t.clarification_state, t.resolved_plan,
                       c.title
                FROM daxview_turns t
                JOIN daxview_conversations c ON c.id = t.conversation_id
                JOIN daxview_jobs j ON j.turn_id = t.id
                WHERE t.id = %s AND t.conversation_id = %s AND t.deployment_id = %s AND t.company_id = %s AND t.user_id = %s
                  AND c.status = 'active' AND j.status = 'completed'
                  AND t.clarification_state->>'status' = 'waiting'
                LIMIT 1
                """,
                (turn_id, conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
            )
            return cur.fetchone()


def get_latest_pending_clarification(conversation_id: str, identity: dict) -> dict | None:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.user_message, t.request_id, t.context, t.clarification_state, t.resolved_plan
                FROM daxview_turns t
                JOIN daxview_conversations c ON c.id = t.conversation_id
                JOIN daxview_jobs j ON j.turn_id = t.id
                WHERE t.conversation_id = %s AND t.deployment_id = %s AND t.company_id = %s AND t.user_id = %s
                  AND c.status = 'active' AND j.status = 'completed'
                  AND t.clarification_state->>'status' = 'waiting'
                ORDER BY t.created_at DESC LIMIT 1
                """,
                (conversation_id, identity["deployment_id"], identity["company_id"], identity["user_id"]),
            )
            return cur.fetchone()


def looks_like_clarification_reply(message: str, state: dict) -> bool:
    lowered = message.strip().lower()
    if not lowered:
        return False
    if re.search(r"\b(?:show|list|give)\b.*\b(?:available\s+)?(?:devices?|meters?)\b", lowered):
        return True
    if len(message.strip()) <= 100 and (
        question_metrics(message) or has_time_scope(message) or requested_device_id(message) is not None
        or has_named_device_reference(message)
    ):
        return True
    # Accept a complete restatement that fills the pending request's device/metric/period slots.
    return bool(
        any(term in lowered for term in ("reading", "readings", "historical", "trend"))
        and (question_metrics(message) or has_time_scope(message) or has_named_device_reference(message))
    )


def resolved_plan_for_result(status: str, operation_ids: list[str], context: dict, results: list[dict] | None = None, errors: list[dict] | None = None) -> dict:
    safe_results = results or []
    slots = {
        "site_id": context.get("site_id"),
        "building_id": context.get("building_id"),
        "device_id": context.get("device_id"),
        "metric": context.get("metric"),
        "_time_window": context.get("_time_window"),
    }
    plan_tools = []
    for item in safe_results:
        args = dict(item.get("arguments") or {})
        args.pop("authorization_id", None)
        plan_tools.append({"tool": item.get("operation_id"), "arguments": args})
    return {
        "status": status,
        "tools": plan_tools or [{"tool": tool, "arguments": {}} for tool in operation_ids],
        "slots": {key: value for key, value in slots.items() if value is not None},
        "errors": errors or [],
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
            waiting_event = {
                "prompt": result["reply"],
                "clarification_turn_id": turn_id,
                "fields": result.get("fields") or result.get("required_fields") or ["site_id", "building_id", "device", "metric", "time_range"],
                "input_type": "text",
                "example": result.get("example") or "For example: ‘Show energy readings for Main Meter from 1–7 October 2026.’",
            }
            if result.get("input_type"):
                waiting_event["input_type"] = result["input_type"]
            if result.get("choices"):
                waiting_event["choices"] = result["choices"]
            if result.get("submit_template"):
                waiting_event["submit_template"] = result["submit_template"]
            waiting_event["example"] = result.get("example") or "For example: ‘Show energy readings for Main Meter from 1–7 October 2026.’"
            add_job_event(
                job_id,
                "waiting_for_user",
                waiting_event,
            )
            update_job_status(job_id, "completed")
            add_job_event(job_id, "completed", {"status": "completed"})
            debug_trace_event("ai_outcome_debug", request_id=request_id, status="waiting_for_user", answer=result["reply"], charts=[])
            save_turn_memory(turn_id, result["reply"], result.get("resolved_plan") or {})
            state = result.get("clarification_state") or context.get("_clarification_pending_state") or {
                "status": "waiting",
                "original_message": context.get("_original_user_message") or message,
                "required_fields": result.get("required_fields") or [],
            }
            state["status"] = "waiting"
            state.setdefault("original_message", context.get("_original_user_message") or message)
            state.setdefault("required_fields", result.get("required_fields") or [])
            save_turn_clarification_state(turn_id, state)
            return
        message_event = {"text": result["reply"]}
        if result.get("charts"):
            message_event["charts"] = result["charts"]
        if result.get("tables"):
            message_event["tables"] = result["tables"]
        if result.get("presentation"):
            message_event["presentation"] = result["presentation"]
        add_job_event(job_id, "message", message_event)
        update_job_status(job_id, "completed")
        add_job_event(job_id, "completed", {"status": "completed"})
        result_tables = result.get("debug_tables") or result.get("tables") or []
        result_charts = result.get("debug_charts") or result.get("charts") or []
        presentation = result.get("presentation") or {}
        plan_tools = (result.get("resolved_plan") or {}).get("tools") or []
        generation = {
            "answer": {
                "builder": "refine_historical_answer_with_model" if AI_REFINE_MCP_WITH_MODEL else "summarize_historical_answers",
                "model": CHAT_MODEL if AI_REFINE_MCP_WITH_MODEL else "deterministic",
            },
            "inputs": {"selected_operations_and_arguments": plan_tools},
            "charts": {
                "builder": "build_charts_from_historical_results / chart_from_historical_result",
                "count": len(result_charts),
                "specs": result_charts,
            },
            "tables": {
                "builder": "build_tables_from_historical_results",
                "count": len(result_tables),
                "tables": result_tables,
            },
            "report": {
                "builder": "summarize_historical_answers / answer_presentation",
                "type": presentation.get("type") or "answer",
                "presentation": presentation,
                "report_data": result.get("report"),
            },
        }
        debug_trace_event(
            "ai_outcome_debug", request_id=request_id, status="completed",
            answer=result["reply"], charts=result_charts,
            tables=result_tables, generation=generation,
            charts_enabled=AI_CHARTS_ENABLED,
        )
        save_turn_memory(turn_id, result["reply"], result.get("resolved_plan") or {})
        clarification_parent = context.get("_clarification_parent_turn_id")
        if clarification_parent:
            with db() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE daxview_turns SET clarification_state = jsonb_set(clarification_state, '{status}', '\"resolved\"'::jsonb) WHERE id = %s", (clarification_parent,))
                conn.commit()
        save_turn_clarification_state(turn_id, {"status": "resolved"})
    except Exception as error:
        log_event(
            "daxview_job_failed",
            request_id=request_id,
            job_id=job_id,
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(limit=6),
        )
        update_job_status(job_id, "failed")
        add_job_event(job_id, "failed", {"status": "failed", "text": "AI response generation failed."})
        debug_trace_event("ai_outcome_debug", request_id=request_id, status="failed", error=str(error), charts=[])


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
    if any(phrase in lowered for phrase in (
        "which device", "highest demand device", "top demand device",
        "device with the highest demand", "device has the highest demand",
    )):
        return False
    if any(phrase in lowered for phrase in ("all devices", "every device", "top devices")):
        return False
    return requested_device_id(message) is None and not re.search(r"\bumg[\s-]?\d+\b", lowered)


def llm_clarification_prompt(original_message: str, missing_fields: list[str], device_names: list[str] | None = None, known_slots: dict | None = None, turn_number: int = 1) -> str:
    fields = ", ".join(missing_fields)
    devices = device_names or []
    known = known_slots or {}
    instruction = (
        "Continue one EMS historical telemetry request across chat turns. Understand its intent and known slots. "
        "Ask only for missing details and never ask again for a supplied detail. Do not claim readings were fetched. "
        "On the first clarification, ask whether the operator wants the available device list or already knows the meter. "
        "Show authorized device names as readable text. Later, ask only unresolved fields. Never invent device names or IDs. "
        "Return concise user-facing text, with a precise example when useful."
    )
    prompt_data = {"original_question": original_message, "intent": "retrieve historical device readings after required details are clear", "missing_details": missing_fields, "known_slots": known, "available_devices": devices[:30], "clarification_turn_number": turn_number}
    try:
        result = ollama_json(
            "/api/chat",
            {"model": CHAT_MODEL, "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
            ], "stream": False, "options": {"temperature": 0.2}},
            timeout=min(OLLAMA_GENERATE_TIMEOUT, 30),
        )
        content = result.get("message", {}).get("content", "")
        prompt = str(content).strip()
        if prompt:
            if turn_number == 1 and devices and devices[0].lower() not in prompt.lower():
                prompt += "\n\nAvailable devices: " + "; ".join(devices[:30]) + "."
            return prompt[:1800]
    except Exception as error:
        log_event("clarification_prompt_model_fallback", error=str(error))
    if turn_number == 1 and devices:
        prompt = "I can retrieve historical readings for this site. Would you like the available device list, or do you already know the meter? Reply ‘show available devices’ or enter its name. Available devices: " + "; ".join(devices[:30]) + "."
    elif turn_number == 1 and any("device" in field or "meter" in field for field in missing_fields):
        prompt = "I can retrieve historical readings for this site. Would you like the available device list, or do you already know the meter? Reply ‘show available devices’ or enter its name."
    else:
        prompt = f"I understand you want historical readings. Please provide only the {fields}."
    if devices and devices[0].lower() not in prompt.lower():
        prompt += " Available devices: " + "; ".join(devices[:15]) + "."
    prompt += " For example: ‘Show energy readings for Main Meter from 1–7 October 2026.’"
    device = known.get("device_name") or known.get("device_ref")
    metric = known.get("metric")
    period = known.get("time_phrase") or "the last 7 days (example)"
    if device and metric:
        prompt += f" Example: ‘Show {metric} readings for {device} for {period}.’"
    elif turn_number > 1:
        prompt += " Example: ‘Show energy (kWh) readings for 96S2_Freezer_3_1 for the last 7 days.’"
    return prompt


def extract_clarification_details(original_message: str, reply: str, context: dict, request_id: str, available_devices: list[str] | None = None) -> tuple[str, dict, dict]:
    extraction = {"original_question": original_message, "operator_reply": reply,
                  "known_context": {key: context.get(key) for key in ("site_id", "building_id", "device_id", "metric") if context.get(key) is not None}}
    try:
        data = ollama_json("/api/chat", {"model": PLANNER_MODEL, "messages": [
            {"role": "system", "content": "Extract only details explicitly stated in operator_reply that refine original_question. Never invent IDs. Return JSON keys query (standalone complete question), metric, device_ref, time_phrase, site_ref, building_ref. Use null when absent."},
            {"role": "user", "content": json.dumps(extraction, ensure_ascii=False)},
        ], "stream": False, "format": {"type": "json"}, "options": {"temperature": 0}}, timeout=PLANNER_TIMEOUT)
        content = data.get("message", {}).get("content", "{}")
        slots = json.loads(content) if isinstance(content, str) else {}
        if not isinstance(slots, dict): slots = {}
    except Exception as error:
        log_event("clarification_extract_fallback", request_id=request_id, error=str(error))
        slots = {}
    if re.search(r"\b(?:show|list|give)\b.*\b(?:available\s+)?(?:devices?|meters?)\b", reply, re.IGNORECASE):
        slots["list_devices"] = True
    # Deterministic fallbacks keep short replies useful when the planner model
    # is unavailable or returns malformed JSON.
    metrics = question_metrics(reply)
    if metrics:
        slots["metric"] = metrics[0]
    # Prefer an exact authorized inventory name over an LLM extraction. This
    # prevents the whole sentence (including the old prompt) becoming a device.
    normalized_reply = normalize_device_name(reply)
    inventory_match = next((name for name in sorted(available_devices or [], key=len, reverse=True)
                            if normalize_device_name(name) and normalize_device_name(name) in normalized_reply), None)
    if inventory_match:
        slots["device_ref"] = inventory_match
    elif slots.get("device_ref"):
        candidate_ref = str(slots["device_ref"]).strip(" \t\r\n.,;:'\"“”‘’")
        if (len(candidate_ref) > 80 or re.search(r"\b(?:show|readings?|historical|metric|period|last|past)\b", candidate_ref, re.IGNORECASE)):
            slots.pop("device_ref", None)
    if not slots.get("device_ref"):
        # Handle a complete sentence when the selected device is not in the
        # cached list, while excluding the surrounding metric and time text.
        named = re.search(
            r"\b(?:readings?|data)\s+for\s+(.+?)(?=\s+(?:for|from|over|during|between)\s+(?:the\s+)?(?:last|past|\d)|[,;.]|$)",
            reply,
            re.IGNORECASE,
        )
        if named:
            slots["device_ref"] = named.group(1).strip(" \t\r\n.,;:'\"“”‘’")
    time_match = re.search(
        r"\b(?:for\s+)?(?:the\s+)?(?:last|past)\s+\d{1,3}\s+(?:days?|weeks?|months?)\b",
        reply,
        re.IGNORECASE,
    )
    if time_match:
        slots["time_phrase"] = time_match.group(0).strip()
    elif not slots.get("time_phrase") and has_time_scope(reply):
        slots["time_phrase"] = reply.strip()
    if not slots.get("device_ref") and not slots.get("list_devices"):
        device_text = reply.strip()
        device_text = re.sub(r"\b(?:metric|parameter|period|time range|energy|kwh|mwh|demand|voltage|current|amps?|power factor|frequency|thd|from|between|last|past|today|yesterday)\b.*$", "", device_text, flags=re.IGNORECASE).strip(" ,.;")
        if device_text and not question_metrics(device_text) and len(device_text.split()) <= 8:
            slots["device_ref"] = device_text
    # Keep the original intent and site, but remove the meta-instruction and
    # its examples. Add only the slots the operator supplied in this reply.
    query = re.split(r"\b(?:please ask|for example)\b", original_message, maxsplit=1, flags=re.IGNORECASE)[0].strip(" \t\r\n.,;:")
    if not query:
        query = "Show historical device readings"
    merged = dict(context)
    if slots.get("metric") and not merged.get("metric"): merged["metric"] = str(slots["metric"])
    if not merged.get("metric"):
        reply_metrics = question_metrics(reply)
        if reply_metrics:
            merged["metric"] = reply_metrics[0]
            slots["metric"] = reply_metrics[0]
    if slots.get("time_phrase") and not has_time_scope(query):
        query = f"{query} {slots['time_phrase']}"
    if not slots.get("time_phrase") and has_time_scope(reply):
        query = f"{query} {reply}"
    if slots.get("device_ref"): merged["_clarification_device_ref"] = str(slots["device_ref"])
    if not slots.get("device_ref") and has_named_device_reference(reply):
        slots["device_ref"] = reply.strip()
        merged["_clarification_device_ref"] = reply.strip()
    if slots.get("building_ref") and str(slots["building_ref"]).isdigit() and not merged.get("building_id"):
        merged["building_id"] = int(slots["building_ref"])
    if slots.get("metric") and not question_metrics(query):
        query += f" Metric: {slots['metric']}."
    if slots.get("device_ref") and normalize_device_name(str(slots["device_ref"])) not in normalize_device_name(query):
        query += f" Device: {slots['device_ref']}."
    if slots.get("site_ref"):
        query += f" Site: {slots['site_ref']}."
    return query, merged, slots


def build_incomplete_request_clarification(message: str, context: dict, request_id: str, required_fields: list[str] | None = None, known_slots: dict | None = None, turn_number: int = 1, list_devices: bool = False, turn_id: str | None = None) -> dict:
    missing = []
    clarification_requested = is_historical_readings_clarification_request(message)
    if required_fields is not None:
        missing = list(required_fields)
    elif not context.get("site_id"):
        missing.append("site/site scope (select this site in DaxView so it can be authorized)")
    if required_fields is None and (clarification_requested or needs_device_choice(message) or (question_metrics(message) and not has_named_device_reference(message) and not context.get("device_id") and not wants_device_usage_ranking(message))):
        missing.append("device or meter")
    if required_fields is None and (clarification_requested or not question_metrics(message)) and not any(word in message.lower() for word in ("alarm", "device list", "devices", "meters")):
        missing.append("metric (such as energy, demand, voltage, or current)")
    if required_fields is None and (clarification_requested or not has_time_scope(message)):
        missing.append("time period (such as last 7 days or 1–7 October 2026)")
    devices = []
    if context.get("site_id") and context.get("company_id") and context.get("deployment_id"):
        try:
            arguments = build_historical_arguments("site_device_list", {**context, "limit": 100}, message)
            result = call_authorized_internal_operation(turn_id or str(uuid.uuid4()), "site_device_list", arguments, request_id, message)
            data = historical_result_data(result)
            devices = [str(first_value(item, ("device_name", "name", "meter_name", "label")))
                       for item in first_list(data, ("devices", "rows", "items", "meters"))
                       if isinstance(item, dict) and first_value(item, ("device_name", "name", "meter_name", "label"))]
        except Exception as error:
            log_event("clarification_device_list_unavailable", request_id=request_id, error=str(error))
    known = dict(known_slots or {})
    if context.get("_device_name"):
        known["device_name"] = context["_device_name"]
    elif context.get("_clarification_device_ref"):
        known["device_ref"] = context["_clarification_device_ref"]
    if context.get("metric"):
        known["metric"] = context["metric"]
    if context.get("_clarification_time_phrase"):
        known["time_phrase"] = context["_clarification_time_phrase"]
    missing_lower = [str(field).lower() for field in (missing or ["device or meter", "metric", "time period"])]
    needs_device = any("device" in field or "meter" in field for field in missing_lower)
    needs_metric = any("metric" in field or "parameter" in field for field in missing_lower)
    needs_period = any("period" in field or "time" in field for field in missing_lower)
    known_device = known.get("device_name") or known.get("device_ref")
    known_metric = known.get("metric")
    known_period = known.get("time_phrase")
    site_name = context.get("_site_name")

    if list_devices and devices:
        prompt = "Available devices for the selected site: " + "; ".join(devices[:50]) + "."
        prompt += " Reply with the device name"
        if needs_metric:
            prompt += ", metric"
        if needs_period:
            prompt += " and time period"
        prompt += "."
    elif turn_number <= 1 and devices:
        prompt = "I understand you want historical device readings"
        if site_name:
            prompt += f" for {site_name}"
        prompt += ". Available devices: " + "; ".join(devices[:50]) + "."
        prompt += " Choose a device and specify the metric and time period. Metrics can include energy (kWh), demand (kW), voltage (V), or current (A)."
        prompt += f" For example: ‘Show energy (kWh) readings for {devices[0]} for the last 7 days.’"
    else:
        known_parts = []
        if known_device:
            known_parts.append(f"device {known_device}")
        if known_metric:
            known_parts.append(f"metric {known_metric}")
        if known_period:
            known_parts.append(f"period {known_period}")
        prompt = "I understand you want historical device readings."
        if known_parts:
            prompt += " I have " + ", ".join(known_parts) + "."
        if needs_device:
            if devices:
                prompt += " Available devices: " + "; ".join(devices[:50]) + "."
            prompt += " Which device or meter should I use?"
        if needs_metric and needs_period:
            prompt += " Which metric (such as energy, demand, voltage, or current) and time period should I use?"
        elif needs_metric:
            prompt += " Which metric should I use (such as energy, demand, voltage, or current)?"
        elif needs_period:
            prompt += " What time period should I use (such as the last 7 days or 1–7 October 2026)?"
        if needs_device or needs_metric or needs_period:
            example_device = known_device or (devices[0] if devices else "<device>")
            example_metric = known_metric or "energy (kWh)"
            example_period = known_period or "the last 7 days"
            prompt += f" Example: ‘Show {example_metric} readings for {example_device} for {example_period}.’"
    if not context.get("site_id"):
        prompt += " Select the requested site in DaxView first; I need its authorized site context to list devices or fetch readings."
    return {
        "provider": "daxview-question-filter",
        "reply": prompt,
        "required_fields": missing,
        "example": "For example: ‘Show energy readings for Main Meter from 1–7 October 2026.’",
        "clarification_state": {"status": "waiting", "intent": "historical_device_readings", "original_message": message, "required_fields": missing, "known_slots": known, "turn_number": turn_number, "available_devices": devices[:50]},
        "resolved_plan": resolved_plan_for_result("clarification", [], context),
    }


def wants_highest_demand_device(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\b(?:highest|max(?:imum)?|peak|top)\s+demand\s+device\b", lowered)
        or re.search(r"\bdevice\s+(?:with|has|having)\s+(?:the\s+)?(?:highest|max(?:imum)?|peak)\s+demand\b", lowered)
        or re.search(r"\b(?:highest|max(?:imum)?|peak)\s+(?:maximum\s+)?demand\b", lowered)
        and requested_device_id(message) is None
    )


def has_time_scope(message: str) -> bool:
    lowered = message.lower()
    if any(
        phrase in lowered
        for phrase in (
            "today",
            "yesterday",
            "last ",
            "past ",
            "this week",
            "last week",
            "this month",
            "last month",
            "august",
            "september",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "october",
            "november",
            "december",
        )
    ):
        return True
    return bool(
        re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", lowered)
        or relative_historical_range(message)
    )


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
    if "demand_peak_summary" in tools and not has_time_scope(message):
        return True
    return not has_daxview_scope(message)


def build_daxview_clarification(message: str, tools: list[str]) -> str:
    if needs_device_choice(message):
        return "Which device should I use for this energy consumption request? Choose a specific device, or say all devices if you want the whole site."
    if "demand_peak_summary" in tools and not has_time_scope(message):
        return "For max demand, which time range should I check: today, yesterday, last 7 days, this month, or a specific date range?"
    if "device_energy_ranking" in tools or "telemetry_top_consumers" in tools:
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


def sandbox_ollama_status() -> dict:
    """Report whether Ollama responds and the configured chat model is installed."""
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)]
        installed = any(name == CHAT_MODEL or name.startswith(f"{CHAT_MODEL}:") for name in names)
        return {"status": "online", "model": CHAT_MODEL,
                "model_installed": installed, "available_models": names}
    except Exception as error:
        return {"status": "offline", "model": CHAT_MODEL, "model_installed": False,
                "error": str(error)[:180]}


def decide_sandbox_data_access(message: str, history: list[dict], request_id: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "needs_sample_data": {"type": "boolean"},
            "categories": {"type": "array", "items": {"type": "string", "enum": ["energy", "devices", "demand", "telemetry", "alarms", "site"]}},
            "formula_topics": {"type": "array", "items": {"type": "string", "enum": ["energy", "demand", "power_quality", "carbon", "comparison"]}},
            "reason": {"type": "string"},
        },
        "required": ["needs_sample_data", "categories", "formula_topics", "reason"],
        "additionalProperties": False,
    }
    system = (
        "You are deciding whether a normal chat answer needs values from an optional local synthetic EMS sample. "
        "Conceptual questions, general explanations, greetings, and calculations using numbers the user supplied "
        "do not need sample data. Only choose sample data when the user asks about actual readings, the sample/site, "
        "a device ranking, a time trend, alarms, or a report based on the sample. Choose the smallest category set "
        "that can answer the current question, using only the current question. Never select every category "
        "by default. If the question is ambiguous, do not load sample data. Category guidance: site energy, daily "
        "energy, or sample carbon estimates use energy; top/highest/lowest consuming devices use devices; peak or "
        "maximum demand uses demand; voltage, current, or power factor uses telemetry; alarm counts/types use alarms; "
        "site identity or a site overview uses site. Carbon calculations may include energy plus the carbon formula. "
        "User-provided readings must be calculated without reading the sample. Return only the requested JSON fields."
    )
    normalized_history = []
    for turn in history[-10:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            normalized_history.append({"role": role, "content": content[:2500]})
    debug_trace_event("sandbox_decision_started", request_id=request_id, model=CHAT_MODEL)
    started = time.perf_counter()
    try:
        response = ollama_json("/api/chat", {
            "model": CHAT_MODEL,
            "messages": [{"role": "system", "content": system}, *normalized_history,
                         {"role": "user", "content": message[:4000]}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 160},
        }, timeout=AI_SANDBOX_PLANNER_TIMEOUT)
        raw = str(((response.get("message") or {}).get("content") or "")).strip()
        decision = json.loads(raw)
        allowed_categories = {"energy", "devices", "demand", "telemetry", "alarms", "site"}
        categories = list(dict.fromkeys(
            item for item in decision.get("categories", []) if item in allowed_categories
        ))
        needs_data = bool(decision.get("needs_sample_data")) and bool(categories)
        if not needs_data:
            categories = []
        allowed_formulas = {"energy", "demand", "power_quality", "carbon", "comparison"}
        formula_topics = list(dict.fromkeys(
            item for item in decision.get("formula_topics", []) if item in allowed_formulas
        ))
        normalized = {
            "needs_sample_data": needs_data,
            "categories": categories,
            "formula_topics": formula_topics,
            "reason": str(decision.get("reason") or "")[:300],
        }
    except Exception as error:
        normalized = {"needs_sample_data": False, "categories": [], "formula_topics": [],
                      "reason": f"Data decision unavailable: {error}"[:300]}
    debug_trace_event("sandbox_decision_response_debug", request_id=request_id, model=CHAT_MODEL,
                      duration_ms=round((time.perf_counter() - started) * 1000), **normalized)
    return normalized


def fast_sandbox_data_decision(message: str) -> dict | None:
    """Skip an LLM routing round-trip for clear sample-data requests and conceptual questions."""
    text = re.sub(r"[^a-z0-9%]+", " ", message.lower()).strip()
    sample_reference = any(term in text for term in (
        "sample", "synthetic", "demo", "this site", "this device", "today", "yesterday",
        "last week", "this week", "last 7 days", "last 30 days", "for the week",
    ))
    has_carbon = any(term in text for term in ("carbon", "emission", "co2"))
    if has_carbon and sample_reference:
        return {"needs_sample_data": True, "categories": ["energy"],
                "formula_topics": ["carbon"], "reason": "Sample emissions request; energy data and carbon factor are required.",
                "routing": "fast_path"}

    has_device = any(term in text for term in ("device", "devices", "consumer", "consumers"))
    has_ranking = any(term in text for term in ("rank", "top", "highest", "lowest", "most energy"))
    if has_device and has_ranking:
        return {"needs_sample_data": True, "categories": ["devices"],
                "formula_topics": [], "reason": "Sample device ranking request.", "routing": "fast_path"}

    has_demand = any(term in text for term in ("peak demand", "maximum demand", "max demand"))
    if sample_reference and has_demand:
        return {"needs_sample_data": True, "categories": ["demand"],
                "formula_topics": ["demand"], "reason": "Sample demand request.", "routing": "fast_path"}

    conceptual = text.startswith(("what is ", "what are ", "what does ", "explain ", "define "))
    if conceptual and not sample_reference:
        return {"needs_sample_data": False, "categories": [], "formula_topics": [],
                "reason": "General conceptual question; sample readings are not needed.", "routing": "fast_path"}
    return None


def sandbox_calculation_fallback(message: str, facts: dict) -> str | None:
    """Provide an explicit deterministic answer when the answer model cannot finish."""
    carbon = facts.get("carbon_estimate")
    if carbon:
        return (
            "The LLM did not finish, so here is the calculation from the synthetic sample. "
            f"Estimated emissions: {carbon['kg_co2e']:,.2f} kg CO2e = "
            f"{facts['daily_energy_total_kwh']:,.2f} kWh x "
            f"{carbon['factor_kg_co2e_per_kwh']} kg CO2e/kWh. "
            f"Period: {carbon['period']}. Limitation: {carbon['status']}."
        )
    ranking = facts.get("device_energy_ranking")
    if ranking and any(word in message.lower() for word in ("rank", "top", "highest", "lowest")):
        rows = sorted(ranking, key=lambda row: float(row.get("energy_kwh") or 0), reverse=True)
        lines = ["The LLM did not finish; these are the calculated synthetic sample device rankings:"]
        lines.extend(f"{index}. {row['name']}: {row['energy_kwh']:,.2f} kWh"
                     for index, row in enumerate(rows[:5], 1))
        return "\n".join(lines)
    demand = facts.get("peak_demand_kw")
    if demand:
        return ("The LLM did not finish; calculated from the synthetic sample: "
                f"peak demand {demand['value']} kW at {demand['timestamp']}; "
                f"minimum demand {facts['minimum_demand_kw']['value']} kW. "
                f"Readings use {facts['demand_interval_minutes']}-minute intervals.")
    return None


def run_sandbox_chat(message: str, history: list[dict], request_id: str, progress=None) -> dict:
    """Answer as a general chat model and load sample readings only when its planner requests them."""
    def set_progress(phase: str) -> None:
        if progress:
            progress(phase)

    debug_trace_event("sandbox_input_debug", request_id=request_id, mode="normal_chat", message=message)
    set_progress("Asking the LLM whether sample readings are needed")
    decision = fast_sandbox_data_decision(message)
    if decision:
        debug_trace_event("sandbox_decision_fast_path", request_id=request_id, **decision)
    else:
        set_progress("Asking the LLM whether sample readings are needed")
        decision = decide_sandbox_data_access(message, history, request_id)
    categories = decision["categories"]
    dataset = sample_dataset() if decision["needs_sample_data"] else None
    if dataset:
        facts, charts, report = prepare_sandbox_context(message, dataset, categories)
        dataset_info = {"name": dataset["dataset_name"], "source": dataset["source"],
                        "is_live": False, "categories_loaded": categories}
        debug_trace_event("sandbox_data_loaded_debug", request_id=request_id, source=dataset["source"],
                          categories=categories, fields=list(facts.keys()))
    else:
        facts, charts, report = {}, [], None
        dataset_info = {"name": None, "source": "not_used", "is_live": False, "categories_loaded": []}
        debug_trace_event("sandbox_data_skipped_debug", request_id=request_id, reason=decision["reason"])
    formulas = formula_reference(decision["formula_topics"])
    set_progress("Preparing a normal chat answer" if not dataset else f"Sample data loaded: {', '.join(categories)}")
    normalized_history = []
    for turn in history[-10:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            normalized_history.append({"role": role, "content": content[:2500]})
    system = (
        "You are a helpful general chatbot with strong Energy Management System knowledge. Answer naturally, "
        "answer each request independently, and answer non-EMS questions too. No sample readings were "
        "loaded unless an optional data block is explicitly included below. If no sample data is included, answer "
        "conceptual questions normally and calculate from values the user provided; do not invent site readings. "
        "When synthetic sample facts are included, use only those facts, state that they are synthetic, retain units "
        "and periods, and do not imply they are live DaxView data. Keep answers focused and avoid generic standards "
        "or cautions unless they are relevant."
    )
    data_context = {}
    if dataset:
        data_context["synthetic_sample_data"] = {"categories": categories, "calculated_facts": facts}
    if formulas:
        data_context["formula_reference"] = formulas
    data_block = ""
    if data_context:
        data_block = "\n\nOptional context selected for this question (synthetic readings are not live):\n" + json.dumps(
            data_context, ensure_ascii=True
        )
    messages = [{"role": "system", "content": system}, *normalized_history,
                {"role": "user", "content": message[:4000] + data_block}]
    debug_trace_event("sandbox_llm_request", request_id=request_id, model=CHAT_MODEL,
                      data_categories=categories, prompt_preview=preview(messages[-1]["content"], 500))
    set_progress("Waiting for Ollama to generate the answer")
    started = time.perf_counter()
    answer_mode = "llm"
    try:
        data = ollama_json("/api/chat", {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 192},
        }, timeout=AI_SANDBOX_MODEL_TIMEOUT)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        debug_trace_event("sandbox_model_failed", request_id=request_id, model=CHAT_MODEL, error=str(error))
        reply = sandbox_calculation_fallback(message, facts) or ""
        if reply:
            answer_mode = "deterministic_fallback"
        else:
            raise RuntimeError(f"Sandbox model request failed: {error}") from error
    else:
        model_message = data.get("message") or {}
        reply = str(model_message.get("content") or "").strip()
    if not reply:
        debug_trace_event("sandbox_model_failed", request_id=request_id, model=CHAT_MODEL,
                          error="empty model answer", thinking_chars=len(str(model_message.get("thinking") or "")),
                          done_reason=data.get("done_reason"), eval_count=data.get("eval_count"))
        reply = sandbox_calculation_fallback(message, facts) or ""
        if not reply:
            raise RuntimeError("Ollama finished without a visible answer. Retry once; if it repeats, check the Ollama container logs and available memory.")
        answer_mode = "deterministic_fallback"
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    debug_trace_event("sandbox_answer_fallback" if answer_mode != "llm" else "sandbox_llm_response_debug",
                      request_id=request_id, model=CHAT_MODEL, duration_ms=elapsed_ms,
                      answer_mode=answer_mode, answer=reply)
    result = {
        "request_id": request_id,
        "provider": "ollama-sandbox",
        "model": CHAT_MODEL,
        "answer_mode": answer_mode,
        "answer": reply,
        "charts": charts,
        "report": report,
        "evidence": facts,
        "dataset": dataset_info,
        "formula_reference": formulas,
        "data_decision": decision,
        "data_accessed": bool(dataset),
        "mcp_called": False,
    }
    debug_trace_event("sandbox_outcome_debug", status="completed", **result)
    return result


def process_sandbox_job(job_id: str, request_id: str, message: str, history: list[dict]) -> None:
    with SANDBOX_JOBS_LOCK:
        job = SANDBOX_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["phase"] = "Starting the LLM data decision"
    try:
        def update_phase(phase: str) -> None:
            with SANDBOX_JOBS_LOCK:
                if job_id in SANDBOX_JOBS:
                    SANDBOX_JOBS[job_id]["phase"] = phase

        result = run_sandbox_chat(message, history, request_id, update_phase)
        with SANDBOX_JOBS_LOCK:
            job = SANDBOX_JOBS.get(job_id)
            if job:
                job.update({"status": "completed", "phase": "Answer ready", "result": result})
    except Exception as error:
        debug_trace_event("sandbox_job_failed", request_id=request_id, job_id=job_id, error=str(error))
        with SANDBOX_JOBS_LOCK:
            job = SANDBOX_JOBS.get(job_id)
            if job:
                job.update({"status": "failed", "phase": "Generation failed", "error": str(error)})


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
    if DAXVIEW_MCP_SESSION_ID:
        mcp_post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )


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
        debug_trace_event(
            "mcp_tool_request_payload_debug",
            request_id=request_id,
            tool=tool_name,
            arguments=arguments or {},
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
        debug_trace_event(
            "mcp_tool_response_payload_debug",
            request_id=request_id,
            tool=tool_name,
            shape=mcp_result_shape(result),
            sample=debug_json_sample(result, DAXVIEW_MCP_DEBUG_RESPONSE_LIMIT),
        )
    return result


def select_daxview_tools(message: str) -> list[str]:
    tools = []
    for intent in select_historical_operations(message):
        tool, _ = public_tool_for_internal(intent)
        if tool not in tools:
            tools.append(tool)
    return tools


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
            result = call_daxview_mcp_tool(tool_name, {}, request_id)
            issue = result_problem(result) or result_problem(mcp_structured_result(result))
            if issue:
                raise RuntimeError(issue)
            results.append({"tool": tool_name, "result": result})
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


def needs_ems_library(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in (
        "standard", "compliance", "certif", "iso", "iec", "ieee", "protocol",
        "modbus", "bacnet", "snmp", "calibration", "accuracy", "enpi",
        "how does", "how do", "explain", "what is", "troubleshoot",
    ))


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
Only cite EMS library facts when they directly answer the question. Do not add ISO or protocol background to routine site results.
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
            "/api/chat",
            {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=OLLAMA_GENERATE_TIMEOUT,
        )
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        log_event("api_to_ollama_error", request_id=request_id, error=str(error))
        raise RuntimeError(f"Ollama is not ready: {error}") from error
    reply = str(data.get("message", {}).get("content") or data.get("response", "")).strip()
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
    if reply is None:
        return ""
    reply = str(reply)
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


DEBUG_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Data Trace Dashboard</title>
    <link rel="stylesheet" href="/debug/assets/prediction_lab.css" />
    <link rel="stylesheet" href="/debug/assets/sandbox.css" />
    <script src="/debug/assets/chart.umd.js"></script>
    <script src="/debug/assets/lucide.min.js"></script>
    <style>
      :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { margin: 0; background: #f5f7fb; color: #172033; }
      header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 22px; background: #111827; color: white; }
      header h1 { font-size: 18px; margin: 0; }
      header .meta { color: #cbd5e1; font-size: 12px; }
      main { display: grid; grid-template-columns: minmax(270px, 22vw) minmax(0, 1fr); gap: 14px; padding: 14px; align-items: start; }
      .panel { min-width: 0; background: white; border: 1px solid #dbe3ef; border-radius: 8px; overflow: hidden; }
      .panel h2 { margin: 0; padding: 12px 14px; font-size: 13px; border-bottom: 1px solid #e5ebf3; background: #f8fafc; }
      .toolbar { display: flex; gap: 8px; align-items: center; padding: 10px 14px; border-bottom: 1px solid #e5ebf3; }
      .tabs { display: flex; gap: 4px; padding: 0 14px; border-bottom: 1px solid #e5ebf3; }
      .tabs button { background: transparent; border: 0; border-bottom: 2px solid transparent; color: #475569; border-radius: 0; }
      .tabs button.active { color: #0369a1; border-bottom-color: #0284c7; }
      input, button { font: inherit; }
      input { flex: 1; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 6px; }
      button { padding: 8px 10px; border: 1px solid #0ea5e9; color: white; background: #0284c7; border-radius: 6px; cursor: pointer; }
      .request-list { max-height: calc(100vh - 150px); overflow: auto; }
      .request { padding: 10px 12px; border-bottom: 1px solid #edf2f7; cursor: pointer; }
      .request:hover, .request.active { background: #e0f2fe; }
      .request .id { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; color: #475569; }
      .request .events { font-size: 12px; margin-top: 4px; color: #0f172a; }
      .chips { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 6px; }
      .chip { font-size: 11px; border-radius: 999px; padding: 2px 7px; background: #e2e8f0; color: #334155; }
      .chip.fail { background: #fee2e2; color: #991b1b; }
      .chip.ok { background: #dcfce7; color: #166534; }
      .timeline { min-width: 0; padding: 12px 14px; overflow: visible; }
      .flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 8px; padding: 12px 14px; border-bottom: 1px solid #e5ebf3; background: #fbfdff; }
      .stage { border: 1px solid #cbd5e1; border-radius: 8px; padding: 9px; min-height: 76px; background: white; }
      .stage.done { border-color: #22c55e; background: #f0fdf4; }
      .stage.fail { border-color: #ef4444; background: #fef2f2; }
      .stage.pending { color: #64748b; background: #f8fafc; }
      .stage .label { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #334155; }
      .stage .count { font-size: 22px; font-weight: 700; margin-top: 6px; color: #0f172a; }
      .stage .detail { font-size: 11px; margin-top: 3px; color: #475569; overflow-wrap: anywhere; }
      .summary-grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; padding: 12px 14px; border-bottom: 1px solid #e5ebf3; }
      .metric { border: 1px solid #e2e8f0; border-radius: 8px; padding: 9px; background: #ffffff; }
      .metric .label { color: #64748b; font-size: 11px; }
      .metric .value { margin-top: 4px; font-size: 13px; font-weight: 700; overflow-wrap: anywhere; }
      .event { border-left: 3px solid #94a3b8; padding: 0 0 14px 12px; margin-left: 6px; }
      .event.fail { border-left-color: #ef4444; }
      .event.ok { border-left-color: #22c55e; }
      .event h3 { margin: 0 0 6px; font-size: 13px; }
      .event .time { color: #64748b; font-size: 11px; margin-bottom: 6px; }
      .inspectors { display: grid; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #e5ebf3; }
      .outcome { padding: 14px; border-bottom: 1px solid #e5ebf3; }
      .outcome h3 { margin: 0 0 10px; font-size: 14px; }
      .outcome-answer { line-height: 1.6; font-size: 13px; max-width: 90ch; }
      .answer-section { padding: 0 0 12px; margin: 0 0 12px; border-bottom: 1px solid #e5ebf3; }
      .answer-section:last-child { border-bottom: 0; margin-bottom: 0; }
      .answer-heading { margin: 0 0 7px; color: #0f766e; font-size: 13px; font-weight: 750; }
      .answer-line { margin: 3px 0; overflow-wrap: anywhere; }
      .answer-line strong { display: inline; color: #172033; }
      .generation-details { margin-top: 12px; border: 1px solid #dbe3ef; background: #f8fafc; padding: 10px; }
      .generation-details summary { cursor: pointer; font-size: 12px; font-weight: 700; }
      .generation-details pre { margin-top: 8px; max-height: 480px; overflow: auto; }
      .outcome-state { margin-top: 10px; padding: 9px 10px; border-left: 3px solid #94a3b8; background: #f8fafc; font-size: 12px; }
      .outcome-state.fail { border-left-color: #ef4444; background: #fef2f2; color: #991b1b; }
      .inspectors h3 { margin: 0 0 8px; font-size: 13px; }
      .inspectors details { border-top: 1px solid #e5ebf3; padding: 8px 0; }
      .inspectors summary { cursor: pointer; font-size: 12px; font-weight: 700; }
      .answer-block { border-left: 3px solid #0ea5e9; padding: 8px 10px; margin-top: 8px; white-space: pre-wrap; font-size: 12px; line-height: 1.5; }
      .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
      .chart-item { border: 1px solid #dbe3ef; padding: 10px; min-width: 0; max-width: 100%; }
      .chart-item strong { font-size: 12px; }
      .chart-summary { margin: 8px 0 0; color: #475569; font-size: 12px; line-height: 1.5; }
      .chart-canvas-wrap { position: relative; height: 260px; max-height: 260px; width: 100%; margin-top: 8px; overflow: hidden; }
      .chart-item canvas { display: block; width: 100% !important; height: 260px !important; max-height: 260px !important; }
      .chart-item details { margin-top: 8px; font-size: 12px; }
      .chart-item summary { cursor: pointer; }
      .chart-item table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
      .chart-item th, .chart-item td { padding: 5px 6px; border-bottom: 1px solid #e5ebf3; text-align: left; }
      .table-grid { display: grid; gap: 10px; margin-top: 10px; }
      .table-item { border: 1px solid #dbe3ef; min-width: 0; max-width: 100%; background: #fff; }
      .table-item-head { padding: 9px 10px; border-bottom: 1px solid #e5ebf3; display: flex; justify-content: space-between; gap: 12px; align-items: baseline; background: #f8fafc; color: #172033; }
      .table-item-head strong { font-size: 12px; }
      .table-meta { color: #64748b; font-size: 11px; }
      .table-debug { margin: 8px 10px; }
      .table-debug summary { color: #475569; cursor: pointer; font-size: 11px; }
      .table-debug pre { margin-top: 6px; max-height: 260px; overflow: auto; }
      .table-raw-preview { margin: 8px 10px; }
      .table-raw-preview strong { font-size: 11px; color: #475569; }
      .table-raw-preview pre { margin-top: 5px; max-height: 220px; overflow: auto; }
      .table-scroll { overflow: auto; max-width: 100%; max-height: 360px; }
      .data-table { border-collapse: collapse; width: 100%; font-size: 12px; white-space: nowrap; }
      .data-table th, .data-table td { padding: 7px 9px; border-bottom: 1px solid #e5ebf3; text-align: left; color: #172033 !important; }
      .data-table th { position: sticky; top: 0; z-index: 1; background: #f8fafc; color: #475569; }
      .data-table tbody tr:hover { background: #f8fafc; }
      pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #0f172a; color: #dbeafe; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.45; margin: 0; }
      .empty { padding: 18px; color: #64748b; }
      @media (max-width: 1100px) { .flow { grid-template-columns: repeat(3, minmax(118px, 1fr)); } .summary-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); } }
      @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
      @media (max-width: 560px) { .flow, .summary-grid { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <header>
      <div>
        <h1>AI Data Trace Dashboard</h1>
        <div class="meta">DaxView -> AI Server -> Data Plan -> MCP -> AI Server -> DaxView</div>
      </div>
      <button id="refresh">Refresh</button>
    </header>
    <nav class="tabs" aria-label="Dashboard views">
      <button type="button" data-view="traces" class="active">Traces</button>
      <button type="button" data-view="predictions">Predictions</button>
      <button type="button" data-view="sandbox">Sandbox</button>
    </nav>
    <main id="trace-workspace">
      <section class="panel">
        <h2>Recent Requests</h2>
        <div class="toolbar"><input id="filter" placeholder="Search saved traces..." /></div>
        <div id="requests" class="request-list"><div class="empty">Loading...</div></div>
        <button id="load-more" type="button" hidden>Load older traces</button>
      </section>
      <section class="panel">
        <h2>Timeline</h2>
        <div id="timeline" class="timeline"><div class="empty">Select a request.</div></div>
      </section>
    </main>
    __PREDICTION_WORKSPACE__
    __SANDBOX_WORKSPACE__
    <script>
      let grouped = [];
      let selected = null;
      let activeView = "traces";
      let nextTraceCursor = null;
      let currentTraceSearch = "";

      function eventClass(name) {
        if (/failed|error|unavailable|rejected|timeout|404|403|409|429|500/i.test(name)) return "fail";
        if (/response|completed|ok/i.test(name)) return "ok";
        return "";
      }
      function summarize(events) {
        const tools = [...new Set(events.map(e => e.tool).filter(Boolean))];
        const failed = events.some(e => /failed|error|unavailable|rejected/i.test(e.event || ""));
        return {tools, failed};
      }
      const flowStages = [
        {key: "incoming", label: "DaxView In", match: e => /daxview|api_to_ui/i.test(e.event || "") || e.job_id},
        {key: "sandbox_input", label: "Sandbox Input", match: e => e.event === "sandbox_input_debug"},
        {key: "sandbox_decision", label: "LLM Data Decision", match: e => /sandbox_decision/i.test(e.event || "")},
        {key: "sandbox_data", label: "Sample Data", match: e => /sandbox_data_(loaded|skipped)/i.test(e.event || "")},
        {key: "planner", label: "LLM Planner", match: e => /planner/i.test(e.event || "")},
        {key: "plan", label: "Data Plan", match: e => /data_plan/i.test(e.event || "")},
        {key: "mcp", label: "MCP Tool", match: e => /mcp_tool/i.test(e.event || "")},
        {key: "retrieval", label: "Knowledge", match: e => /retrieve|context|embedding/i.test(e.event || "")},
        {key: "model", label: "AI Model", match: e => /ollama|agent|synthesizer|mcp_answer_refine|sandbox_llm|sandbox_decision/i.test(e.event || "")},
        {key: "response", label: "Response", match: e => e.event === "sandbox_outcome_debug" || e.event === "sandbox_answer_fallback" ||
          (/response|completed|api_to_ui/i.test(e.event || "") && !/sandbox_decision_response/i.test(e.event || ""))},
      ];
      function durationMs(events) {
        const times = events.map(e => Date.parse(e.timestamp)).filter(Number.isFinite).sort((a, b) => a - b);
        return times.length > 1 ? times[times.length - 1] - times[0] : 0;
      }
      function stageState(events, stage) {
        const matches = events.filter(stage.match);
        const failed = matches.some(e => /failed|error|unavailable|rejected|timeout/i.test(e.event || ""))
          || (stage.key === "mcp" && events.some(e => e.event === "energy_prediction_unavailable"));
        return {matches, failed, status: failed ? "fail" : matches.length ? "done" : "pending"};
      }
      function renderFlow(events) {
        return `<div class="flow">${flowStages.map(stage => {
          const state = stageState(events, stage);
          const last = state.matches[state.matches.length - 1] || {};
          const detail = last.tool || last.provider || last.model || last.event || "No trace yet";
          return `<div class="stage ${state.status}">
            <div class="label">${escapeHtml(stage.label)}</div>
            <div class="count">${state.matches.length}</div>
            <div class="detail">${escapeHtml(detail)}</div>
          </div>`;
        }).join("")}</div>`;
      }
      function renderSummary(group) {
        const events = group.events;
        const info = summarize(events);
        const models = [...new Set(events.map(e => e.model).filter(Boolean))];
        const jobs = [...new Set(events.map(e => e.job_id).filter(Boolean))];
        const errors = events.filter(e => /failed|error|unavailable|rejected|timeout/i.test(e.event || ""));
        const elapsed = durationMs(events);
        const metrics = [
          ["Request", group.id],
          ["Duration", elapsed ? `${elapsed} ms` : "n/a"],
          ["Tools", info.tools.join(", ") || "none"],
          ["Models", models.join(", ") || "none"],
          ["Job IDs", jobs.join(", ") || "none"],
          ["Errors", String(errors.length)],
          ["First Event", events[0]?.event || "n/a"],
          ["Last Event", events[events.length - 1]?.event || "n/a"],
        ];
        return `<div class="summary-grid">${metrics.map(([label, value]) => `<div class="metric">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
        </div>`).join("")}</div>`;
      }
      function renderInspectors(events) {
        const answers = events.filter(e => e.event === "mcp_answer_refine_response_debug" && e.role !== "compare");
        const payloads = events.filter(e => e.event === "mcp_tool_request_payload_debug" || e.event === "mcp_tool_response_payload_debug");
        const answerHtml = answers.map((item, index) => `<div class="answer-block"><strong>${escapeHtml(item.role || "model")} - ${escapeHtml(item.model || "unknown model")}</strong><br>${escapeHtml(item.answer || "")}</div>`).join("");
        const payloadHtml = payloads.map(item => `<details><summary>${escapeHtml(item.event.replace("_debug", ""))} · ${escapeHtml(item.tool || "")}</summary><pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre></details>`).join("");
        if (!answerHtml && !payloadHtml) return "";
        return `<section class="inspectors">
          ${answerHtml ? `<div><h3>Model answers</h3>${answerHtml}</div>` : ""}
          ${payloadHtml ? `<div><h3>Data payloads</h3>${payloadHtml}</div>` : ""}
        </section>`;
      }
      function appendInlineText(parent, text) {
        const pattern = /\\*\\*(.+?)\\*\\*/g;
        let cursor = 0;
        let match;
        while ((match = pattern.exec(text)) !== null) {
          parent.append(document.createTextNode(text.slice(cursor, match.index)));
          const strong = document.createElement("strong");
          strong.textContent = match[1];
          parent.append(strong);
          cursor = pattern.lastIndex;
        }
        parent.append(document.createTextNode(text.slice(cursor)));
      }
      function renderReadableAnswer(text) {
        let normalized = String(text || "").replace(/\\r/g, "");
        normalized = normalized.replace(/^Site energy summary\\s+Site energy summary:/im, "Site energy summary:");
        normalized = normalized.replace(/^(Site energy summary:|Active alarms Summary:|telemetry_timeseries(?:\\s+Telemetry Timeseries)?:|Quick read:)\\s*/gim, "$1\\n");
        normalized = normalized.replace(/^(Top energy-consuming devices)\\s+(?=Device ranking for this site)/gim, "$1\\n");
        normalized = normalized.replace(/\\s+(?=(?:Site energy summary:|telemetry_timeseries\\b|Active alarms Summary:|Top energy-consuming devices\\b|Quick read:))/gi, "\\n\\n");
        normalized = normalized.replace(/\\s+(?=(?:Daily values(?: \\([^)]*\\))?:|Total:|Summary:|Latest active alarms:|Device ranking for this site|Showing \\d+ measured|Energy:|Alarms:|Consumption leader:|Data check:|Coverage check:))/g, "\\n");
        const root = document.createElement("div");
        root.className = "outcome-answer";
        const headings = /^(?:Site energy summary:?|telemetry_timeseries(?:\\s+Telemetry Timeseries)?:?|Active alarms(?: Summary)?:?|Top energy-consuming devices:?|Quick read:?)$/i;
        normalized.split(/\\n\\s*\\n/).filter(section => section.trim()).forEach(sectionText => {
          const section = document.createElement("section");
          section.className = "answer-section";
          const lines = sectionText.split("\\n").filter(line => line.trim());
          lines.forEach((rawLine, index) => {
            const line = rawLine.trim();
            if (headings.test(line) || (index === 0 && line.endsWith(":") && line.length < 48)) {
              const heading = document.createElement("h4");
              heading.className = "answer-heading";
              appendInlineText(heading, line.replace(/:$/, ""));
              section.append(heading);
              return;
            }
            const row = document.createElement("div");
            row.className = "answer-line";
            const labelMatch = line.match(/^([-•]\\s*)?([^:]{1,44}:)(\\s*)(.*)$/);
            if (labelMatch) {
              if (labelMatch[1]) row.append(document.createTextNode(labelMatch[1]));
              const label = document.createElement("strong");
              appendInlineText(label, labelMatch[2]);
              row.append(label, document.createTextNode(labelMatch[3]));
              appendInlineText(row, labelMatch[4]);
            } else {
              appendInlineText(row, line);
            }
            section.append(row);
          });
          root.append(section);
        });
        return root;
      }
      function tableColumn(column) {
        if (typeof column === "string") return { key: column, label: column };
        if (Array.isArray(column)) return { key: column[0], label: column[1] || column[0] };
        return {
          key: column?.key || column?.field || column?.name || column?.accessorKey || column?.dataIndex || column?.label || column?.title || "",
          label: column?.label || column?.title || column?.name || column?.key || column?.field || "",
        };
      }
      function tableCellValue(row, column, index) {
        let source = row?.data ?? row?.values ?? row?.cells ?? row?.record ?? row;
        if (Array.isArray(source)) source = source[index];
        else if (source && typeof source === "object") {
          const normalize = value => String(value ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
          const aliases = {
            rank: ["index", "number", "#"], period: ["date", "timestamp", "bucket", "time"],
            value: ["energy", "reading", "amount", "consumption", "kwh"],
            device: ["device_name", "meter_name", "name"], device_id: ["deviceid", "meter_id", "id"],
            alarm: ["alarm_name", "alarm_type", "type", "name"], severity: ["priority"],
            started_at: ["started", "first_seen", "start_time"], latest: ["latest_occurrence", "last_seen", "last_occurrence", "latest_at"],
            frequency: ["value", "reading", "frequency_hz"], highest_at: ["peak_time", "timestamp", "time"],
            maximum_demand: ["peak_kw", "max_demand_kw", "value"],
          };
          const wanted = [column.key, column.label, ...(aliases[normalize(column.key)] || [])].map(normalize).filter(Boolean);
          const actual = Object.keys(source).find(key => wanted.includes(normalize(key)));
          source = actual !== undefined ? source[actual] : Object.values(source)[index];
        }
        if (source && typeof source === "object") source = source.value ?? source.text ?? source.label ?? source.name ?? JSON.stringify(source);
        return source == null || source === "" ? "—" : source;
      }
      function renderGenerationDetails(outcome, final) {
        const details = outcome?.generation || final?.generation;
        if (!details) return "";
        return `<details class="generation-details"><summary>Generation details · charts, tables, and report</summary>
          <p><strong>Answer:</strong> ${escapeHtml(details.answer?.builder || "not recorded")}${details.answer?.model ? ` · ${escapeHtml(details.answer.model)}` : ""}</p>
          <p><strong>Inputs:</strong> selected operation IDs and sanitized arguments shown below.</p>
          <p><strong>Charts:</strong> ${escapeHtml(details.charts?.builder || "not requested")} · ${escapeHtml(details.charts?.count ?? 0)} chart(s).</p>
          <p><strong>Tables:</strong> ${escapeHtml(details.tables?.builder || "not requested")} · ${escapeHtml(details.tables?.count ?? 0)} table(s).</p>
          <p><strong>Report/presentation:</strong> ${escapeHtml(details.report?.builder || "not generated")} · ${escapeHtml(details.report?.type || "no report type recorded")}.</p>
          <pre>${escapeHtml(JSON.stringify(details, null, 2))}</pre></details>`;
      }
      function renderOutcome(events) {
        const outcome = events.find(event => event.event === "ai_outcome_debug" || event.event === "sandbox_outcome_debug");
        const final = events.find(event => event.event === "ai_final_response_debug");
        const failed = events.find(event => event.event === "daxview_job_failed");
        const unavailable = events.find(event => event.event === "energy_prediction_unavailable");
        const fallback = events.find(event => event.event === "mcp_answer_refine_fallback");
        const answer = outcome?.answer || final?.answer || "";
        const charts = outcome?.charts || final?.charts || [];
        const tables = outcome?.tables || final?.tables || [];
        const report = outcome?.report || null;
        const generationHtml = renderGenerationDetails(outcome, final);
        const error = outcome?.error || failed?.error || "";
        const unavailableCode = unavailable?.error_code || (/DAXVIEW_UNAVAILABLE/.test(error) ? "DAXVIEW_UNAVAILABLE" : "");
        const chartExpected = events.some(event => event.event === "sandbox_outcome_debug" ||
          (event.operation_ids || []).some(id => ["energy_forecast", "site_energy_summary", "telemetry_timeseries", "alarm_frequency_summary", "demand_peak_summary"].includes(id))
        );
        const chartItems = charts.map(chart => {
          const chartSeries = Array.isArray(chart.series) ? chart.series : [];
          const valueRows = (chart.labels || []).map((label, index) =>
            `<tr><td>${escapeHtml(label)}</td>${chartSeries.map(series => `<td>${escapeHtml(series.data?.[index] ?? "")}</td>`).join("")}</tr>`
          ).join("");
          return `<div class="chart-item"><strong>${escapeHtml(chart.title || "Chart")}</strong>
            <div class="chart-canvas-wrap"><canvas data-spec="${escapeHtml(JSON.stringify(chart))}" aria-label="${escapeHtml(chart.title || "Chart preview")}"></canvas></div>
            ${chart.summary ? `<p class="chart-summary">${escapeHtml(chart.summary).replace(/\\n/g, "<br>")}</p>` : ""}
            <details><summary>View chart values</summary><table><thead><tr><th>Period</th>${chartSeries.map(series => `<th>${escapeHtml(series.name || "Value")}${series.unit ? ` (${escapeHtml(series.unit)})` : ""}</th>`).join("")}</tr></thead><tbody>${valueRows}</tbody></table></details>
          </div>`;
        }).join("");
        const tableItems = tables.map(table => {
          const columns = Array.isArray(table.columns) ? table.columns.map(tableColumn) : [];
          const rows = Array.isArray(table.rows) ? table.rows : [];
          const heading = `<tr>${columns.map(column => `<th scope="col">${escapeHtml(column.label || column.key || "")}</th>`).join("")}</tr>`;
          const body = rows.map(row => `<tr>${columns.map((column, index) => {
            return `<td>${escapeHtml(tableCellValue(row, column, index))}</td>`;
          }).join("")}</tr>`).join("") || `<tr><td colspan="${Math.max(columns.length, 1)}">No rows were returned for this table.</td></tr>`;
          const range = table.time_range || {};
          const rangeText = range.start_time || range.end_time
            ? ` · ${escapeHtml(range.start_time || "")} – ${escapeHtml(range.end_time || "")}${range.timezone ? ` (${escapeHtml(range.timezone)})` : ""}`
            : "";
          const truncated = table.truncated ? " · truncated" : "";
          return `<section class="table-item"><header class="table-item-head"><strong>${escapeHtml(table.title || "Data")}</strong>
            <span class="table-meta">${escapeHtml(table.source_tool || "MCP")} · ${escapeHtml(rows.length)} shown${table.row_count != null ? ` of ${escapeHtml(table.row_count)}` : ""}${truncated}${rangeText}</span></header>
            <div class="table-scroll"><table class="data-table"><thead>${heading || "<tr><th>Data</th></tr>"}</thead><tbody>${body}</tbody></table></div>
            <div class="table-raw-preview"><strong>Raw row data (first 5 records)</strong><pre>${escapeHtml(JSON.stringify(rows.slice(0, 5), null, 2))}</pre></div>
            <details class="table-debug"><summary>Inspect raw table payload</summary><pre>${escapeHtml(JSON.stringify({columns: table.columns, sample_rows: rows.slice(0, 3), row_count: table.row_count}, null, 2))}</pre></details></section>`;
        }).join("");
        let state = "";
        if (unavailableCode) state = `DaxView returned ${unavailableCode}. There are no historical values to plot for this request.`;
        else if (error) state = error;
        else if (!outcome && !final) state = "Waiting for the AI Server result.";
        else if (chartExpected && !charts.length && outcome?.charts_enabled === false) state = "Chart generation is disabled in the AI Server configuration.";
        else if (chartExpected && !charts.length) state = "No numeric chart data was available for this answer.";
        return `<section class="outcome"><h3>Outcome preview</h3>
          ${fallback ? `<div class="outcome-state">Model refinement unavailable; showing the calculated fallback.</div>` : ""}
          ${answer ? renderReadableAnswer(answer).outerHTML : ""}
          ${tableItems ? `<div class="table-grid">${tableItems}</div>` : ""}
          ${chartItems ? `<div class="chart-grid">${chartItems}</div>` : ""}
          ${report ? `<details><summary>Report structure</summary><pre>${escapeHtml(JSON.stringify(report, null, 2))}</pre></details>` : ""}
          ${generationHtml}
          ${state ? `<div class="outcome-state ${unavailableCode || error ? "fail" : ""}">${escapeHtml(state)}</div>` : ""}
        </section>`;
      }
      function drawCharts() {
        document.querySelectorAll("canvas[data-spec]").forEach(canvas => {
          try {
            const spec = JSON.parse(canvas.dataset.spec);
            if (window.renderDashboardChart) window.renderDashboardChart(canvas, spec);
          } catch (error) { console.error("Chart preview failed", error); }
        });
      }
      function groupTraces(traces) {
        const map = new Map();
        for (const trace of traces) {
          const id = trace.request_id || trace.job_id || "no-request-id";
          if (!map.has(id)) map.set(id, []);
          map.get(id).push(trace);
        }
        return [...map.entries()].map(([id, events]) => ({id, events: events.slice().reverse()}));
      }
      async function load() {
        const search = document.querySelector("#filter").value.trim();
        if (search !== currentTraceSearch) {
          grouped = [];
          nextTraceCursor = null;
          currentTraceSearch = search;
        }
        const query = new URLSearchParams({limit: "200"});
        if (nextTraceCursor) query.set("before_id", String(nextTraceCursor));
        if (search) query.set("q", search);
        const res = await fetch(`/debug/traces?${query}`);
        if (!res.ok) {
          document.querySelector("#requests").innerHTML = '<div class="empty">Unauthorized or debug dashboard disabled.</div>';
          return;
        }
        const data = await res.json();
        const incoming = groupTraces(data.traces || []);
        const merged = new Map(grouped.map(group => [group.id, group]));
        incoming.forEach(group => merged.set(group.id, group));
        grouped = [...merged.values()];
        nextTraceCursor = data.next_before_id || null;
        document.querySelector("#load-more").hidden = !nextTraceCursor;
        renderRequests();
        if (selected) renderTimeline(selected);
      }
      function renderRequests() {
        const filter = document.querySelector("#filter").value.toLowerCase();
        const root = document.querySelector("#requests");
        const rows = grouped.filter(group =>
          (activeView === "traces" || group.events.some(event =>
            event.event === "energy_prediction_debug" || (event.operation_ids || []).includes("energy_forecast")))
          && JSON.stringify(group).toLowerCase().includes(filter)
        );
        if (!rows.length) {
          root.innerHTML = activeView === "predictions"
            ? '<div class="empty">No prediction requests yet. Ask DaxView for a 7-day energy forecast, then refresh.</div>'
            : '<div class="empty">No traces yet. Ask an AI question, then refresh.</div>';
          return;
        }
        root.innerHTML = rows.map(group => {
          const info = summarize(group.events);
          const classes = ["request", selected === group.id ? "active" : ""].join(" ");
          const chips = [`<span class="chip ${info.failed ? "fail" : "ok"}">${info.failed ? "failed/error" : "ok/no error"}</span>`]
            .concat(info.tools.map(tool => `<span class="chip">${tool}</span>`)).join("");
          return `<div class="${classes}" data-id="${group.id}">
            <div class="id">${escapeHtml(group.id)}</div>
            <div class="events">${group.events.length} event(s)</div>
            <div class="chips">${chips}</div>
          </div>`;
        }).join("");
        root.querySelectorAll(".request").forEach(node => {
          node.addEventListener("click", () => {
            selected = node.dataset.id;
            renderRequests();
            renderTimeline(selected);
          });
        });
      }
      function renderTimeline(id) {
        const group = grouped.find(item => item.id === id);
        const root = document.querySelector("#timeline");
        if (!group) {
          root.innerHTML = '<div class="empty">Select a request.</div>';
          return;
        }
        const eventsHtml = group.events.map(event => {
          const cls = eventClass(event.event || "");
          return `<article class="event ${cls}">
            <h3>${escapeHtml(event.event || "event")}</h3>
            <div class="time">${escapeHtml(event.timestamp || "")}</div>
            <pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre>
          </article>`;
        }).join("");
        const prediction = group.events.find(event => event.event === "energy_prediction_debug")?.prediction;
        const predictionHtml = activeView === "predictions" && prediction
          ? `<div class="summary-grid">
              <div class="metric"><div class="label">History</div><div class="value">${escapeHtml(prediction.history_start)} to ${escapeHtml(prediction.history_end)}</div></div>
              <div class="metric"><div class="label">Coverage</div><div class="value">${escapeHtml(prediction.coverage_percent)}%</div></div>
              <div class="metric"><div class="label">Backtest MAE</div><div class="value">${escapeHtml(prediction.backtest_mae)} ${escapeHtml(prediction.unit)}</div></div>
              <div class="metric"><div class="label">Method</div><div class="value">${escapeHtml(prediction.method)}</div></div>
            </div>` : "";
        root.innerHTML = activeView === "predictions"
          ? predictionHtml + renderOutcome(group.events) + renderInspectors(group.events) + eventsHtml
          : renderFlow(group.events) + renderSummary(group) + renderOutcome(group.events) + renderInspectors(group.events) + eventsHtml;
        drawCharts();
      }
      function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
      }
      document.querySelector("#refresh").addEventListener("click", () => {
        if (activeView === "predictions") window.PredictionLab.show();
        else if (activeView === "sandbox") window.SandboxChat.refresh();
        else load();
      });
      document.querySelector("#load-more").addEventListener("click", load);
      document.querySelector("#filter").addEventListener("input", () => {
        grouped = [];
        nextTraceCursor = null;
        clearTimeout(window.traceSearchTimer);
        window.traceSearchTimer = setTimeout(load, 250);
      });
      document.querySelectorAll(".tabs button").forEach(button => button.addEventListener("click", () => {
        activeView = button.dataset.view;
        document.querySelectorAll(".tabs button").forEach(tab => tab.classList.toggle("active", tab === button));
        document.querySelector("#trace-workspace").hidden = activeView !== "traces";
        document.querySelector("#prediction-workspace").hidden = activeView !== "predictions";
        document.querySelector("#sandbox-workspace").hidden = activeView !== "sandbox";
        if (activeView === "predictions") {
          window.PredictionLab.show();
          return;
        }
        if (activeView === "sandbox") {
          window.SandboxChat.refresh();
          return;
        }
        selected = null;
        renderRequests();
        if (selected) renderTimeline(selected);
        else document.querySelector("#timeline").innerHTML = '<div class="empty">Select a request.</div>';
      }));
      load();
      setInterval(load, 10000);
    </script>
    <script src="/debug/assets/prediction_lab.js"></script>
  </body>
</html>"""


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
    contexts = retrieve_context(message) if needs_ems_library(message) else []
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
        results = [{"operation_id": item.get("tool", "mcp"), "arguments": {}, "result": item.get("result", {})}
                   for item in state["mcp_context"].get("results", [])]
        state["reply"] = refine_historical_answer_with_model(state["message"], results, direct_answer, state["request_id"])
        state["provider"] = "daxview-mcp-refined"
        state["model"] = CHAT_MODEL
        state["decision_model"] = "Daxview MCP direct extractor"
        state["agent_answers"] = [
            {
                "agent": "Daxview MCP Direct Extractor",
                "role": "Read structured MCP data and answer deterministic count/status questions.",
                "model": "deterministic",
                "answer": state["reply"],
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


def daxview_history_event(event: dict) -> dict:
    event_type = event["event_type"]
    data = event.get("event_data") or {}
    item = {"type": event_type, "text": data.get("text") or data.get("prompt") or ""}
    if event_type == "waiting_for_user":
        item.update({key: data[key] for key in ("fields", "input_type", "choices", "submit_template") if key in data})
    return item


class ChatHandler(BaseHTTPRequestHandler):
    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-DaxView-Assertion, X-Debug-Key")
        self.end_headers()
        self.wfile.write(body)

    def _debug_allowed(self) -> bool:
        return AI_DEBUG_DASHBOARD_ENABLED

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
                                    "assistant_events": [daxview_history_event(event) for event in events],
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
                context = {
                    **context,
                    "deployment_id": identity["deployment_id"],
                    "company_id": identity["company_id"],
                    "user_id": identity["user_id"],
                }
                if not turn_id:
                    self._send_json(400, json_error("invalid_turn", "turn_id is required.", False, request_id))
                    return True
                if not message:
                    self._send_json(400, json_error("invalid_turn", "message is required.", False, request_id))
                    return True
                clarification_turn_id = str(request.get("clarification_for_turn_id") or "").strip()
                pending = get_pending_clarification(conversation_id, identity, clarification_turn_id) if clarification_turn_id else None
                if not clarification_turn_id:
                    candidate_pending = get_latest_pending_clarification(conversation_id, identity)
                    candidate_state = (candidate_pending or {}).get("clarification_state") or {}
                    if candidate_pending and looks_like_clarification_reply(message, candidate_state):
                        pending = candidate_pending
                        clarification_turn_id = str(candidate_pending["id"])
                parent_turn_id = None
                clarification = {}
                if pending:
                    clarification = pending.get("clarification_state") or {}
                    original_message = str(clarification.get("original_message") or pending.get("user_message") or "")
                    prior_context = pending.get("context") or {}
                    if isinstance(prior_context, str):
                        prior_context = json.loads(prior_context)
                    clarification["original_message"] = original_message
                    clarification["slots"] = clarification.get("slots") or {}
                    known_slots = clarification.get("known_slots") or {}
                    context = {**prior_context, **context, **clarification.get("slots", {}), **known_slots}
                    if known_slots.get("metric"):
                        context["metric"] = known_slots["metric"]
                    if known_slots.get("time_phrase"):
                        context["_clarification_time_phrase"] = known_slots["time_phrase"]
                        context["_time_window"] = requested_historical_range(str(known_slots["time_phrase"]))
                    if known_slots.get("device_ref"):
                        context["_clarification_device_ref"] = known_slots["device_ref"]
                    context["_original_user_message"] = original_message
                    context["_clarification_reply"] = message
                    candidate_message, context, extracted = extract_clarification_details(
                        original_message,
                        message,
                        context,
                        request_id,
                        available_devices=clarification.get("available_devices") or [],
                    )
                    slot_updates = {}
                    known_slots = dict(clarification.get("known_slots") or clarification.get("slots") or {})
                    list_devices = bool(extracted.get("list_devices"))
                    if extracted.get("metric"):
                        slot_updates["metric"] = str(extracted["metric"])
                        known_slots["metric"] = str(extracted["metric"])
                    if extracted.get("time_phrase"):
                        slot_updates["_time_window"] = requested_historical_range(str(extracted["time_phrase"]))
                        slot_updates["_clarification_time_phrase"] = str(extracted["time_phrase"])
                        known_slots["time_phrase"] = str(extracted["time_phrase"])
                    if extracted.get("device_ref"):
                        slot_updates["_clarification_device_ref"] = str(extracted["device_ref"])
                        known_slots["device_ref"] = str(extracted["device_ref"])
                    if extracted.get("device_ref") or extracted.get("list_devices"):
                        context.pop("_clarification_device_unresolved", None)
                    if slot_updates:
                        clarification["slots"].update(slot_updates)
                        context.update(slot_updates)
                    if not context.get("site_id") and extracted.get("site_ref"):
                        site_ref = str(extracted["site_ref"]).strip().lower()
                        authorized_sites = context.get("authorized_sites") or context.get("sites") or []
                        site_matches = [item for item in authorized_sites if isinstance(item, dict) and
                                        site_ref in str(item.get("name") or item.get("site_name") or "").lower()]
                        if len(site_matches) == 1:
                            context["site_id"] = site_matches[0].get("site_id") or site_matches[0].get("id")
                            context["_site_name"] = site_matches[0].get("name") or site_matches[0].get("site_name")
                    if context.get("site_id") and extracted.get("device_ref") and not extracted.get("list_devices") and not context.get("device_id"):
                        resolution = resolve_named_device_reference(turn_id, str(extracted["device_ref"]), context, request_id)
                        if resolution.get("status") == "matched":
                            context["device_id"] = resolution["device_id"]
                            context["_device_name"] = resolution["device_name"]
                        else:
                            context["_clarification_device_unresolved"] = True
                    operations, _merged_context, _plan = choose_historical_operations(candidate_message, context, request_id)
                    remaining = []
                    if not context.get("site_id") and any("site/site scope" in str(field) for field in clarification.get("required_fields", [])):
                        remaining.append("site/site scope")
                    if context.get("_clarification_device_unresolved") or ("device or meter" in clarification.get("required_fields", []) and not context.get("device_id") and not wants_all_devices(candidate_message) and not wants_highest_demand_device(candidate_message)):
                        remaining.append("device or meter")
                    if any("metric" in field for field in clarification.get("required_fields", [])) and not context.get("metric"):
                        remaining.append("metric")
                    if any("period" in field or "time" in field for field in clarification.get("required_fields", [])) and not context.get("_clarification_time_phrase") and not has_time_scope(message):
                        remaining.append("time period")
                    if list_devices and "device or meter" not in remaining:
                        remaining.append("device or meter")
                    if not context.get("site_id"):
                        remaining.append("site/site scope (select the authorized site in DaxView)")
                    if remaining:
                        clarification_result = build_incomplete_request_clarification(
                            original_message, context, request_id, remaining, known_slots,
                            int(clarification.get("turn_number") or 1) + 1, list_devices, turn_id,
                        )
                        clarification_result["clarification_state"] = {
                            **clarification,
                            "status": "waiting",
                            "original_message": original_message,
                            "last_reply": message,
                            "required_fields": clarification_result.get("required_fields") or [],
                            "known_slots": known_slots,
                            "turn_number": int(clarification.get("turn_number") or 1) + 1,
                            "available_devices": clarification_result.get("clarification_state", {}).get("available_devices", []),
                        }
                        context["_clarification_result"] = clarification_result
                        context["_clarification_parent_turn_id"] = str(pending["id"])
                        context["_clarification_pending_state"] = clarification_result["clarification_state"]
                    parent_turn_id = str(pending["id"])
                    message = candidate_message
                    stored_user_message = str(request.get("message") or "").strip()
                elif clarification_turn_id:
                    self._send_json(409, json_error("clarification_not_pending", "That clarification is no longer waiting for a reply.", False, request_id))
                    return True
                else:
                    stored_user_message = message
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
                            INSERT INTO daxview_turns (id, conversation_id, deployment_id, company_id, user_id, user_message, request_id, context, parent_turn_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                turn_id,
                                conversation_id,
                                identity["deployment_id"],
                                identity["company_id"],
                                identity["user_id"],
                                stored_user_message,
                                request_id,
                                json.dumps(context),
                                parent_turn_id,
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
                    if parent_turn_id and not context.get("_clarification_result"):
                        with db() as conn:
                            with conn.cursor() as cur:
                                cur.execute("UPDATE daxview_turns SET clarification_state = %s WHERE id = %s",
                                            (json.dumps({**clarification, "status": "answered", "last_reply": request.get("message")}), parent_turn_id))
                            conn.commit()
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
        if path == "/debug/ai":
            if not AI_DEBUG_DASHBOARD_ENABLED:
                self._send_html(404, "<h1>Debug dashboard disabled</h1>")
                return
            html = DEBUG_DASHBOARD_HTML.replace("__PREDICTION_WORKSPACE__", (STATIC_DIRECTORY / "prediction_lab.html").read_text(encoding="utf-8"))
            html = html.replace("__SANDBOX_WORKSPACE__", (STATIC_DIRECTORY / "sandbox_workspace.html").read_text(encoding="utf-8"))
            html = html.replace("<script src=\"/debug/assets/prediction_lab.js\"></script>",
                                "<script src=\"/debug/assets/prediction_lab.js\"></script><script src=\"/debug/assets/sandbox.js\"></script>")
            self._send_html(200, html)
            return
        if path.startswith("/debug/assets/"):
            files = {
                "prediction_lab.js": ("prediction_lab.js", "text/javascript"),
                "prediction_lab.css": ("prediction_lab.css", "text/css"),
                "sandbox.js": ("sandbox.js", "text/javascript"),
                "sandbox.css": ("sandbox.css", "text/css"),
                "chart.umd.js": ("vendor/chart.umd.js", "text/javascript"),
                "lucide.min.js": ("vendor/lucide.min.js", "text/javascript"),
            }
            item = files.get(path.removeprefix("/debug/assets/"))
            if not self._debug_allowed() or not item:
                self._send_json(404, {"error": "Not found"})
                return
            body = (STATIC_DIRECTORY / item[0]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", item[1])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/debug/sandbox":
            if not self._debug_allowed():
                self._send_json(404, {"error": "Dashboard disabled"})
                return
            self._send_json(200, {"dataset": {"dataset_name": "Synthetic Office EMS Sample",
                                                "site": {"name": "Sample Office Campus", "site_id": 17},
                                                "available_categories": ["site", "energy", "devices", "demand", "telemetry", "alarms"],
                                                "is_live": False},
                                  "formulas": formula_reference(),
                                  "model": CHAT_MODEL, "ollama": sandbox_ollama_status()})
            return
        sandbox_job_match = re.fullmatch(r"/debug/sandbox/jobs/([0-9a-fA-F-]{36})", path)
        if sandbox_job_match:
            if not self._debug_allowed():
                self._send_json(404, {"error": "Dashboard disabled"})
                return
            with SANDBOX_JOBS_LOCK:
                job = SANDBOX_JOBS.get(sandbox_job_match.group(1))
                payload = dict(job) if job else None
            if not payload:
                self._send_json(404, {"error": "Sandbox job not found. It may have expired after a server restart."})
                return
            self._send_json(200, payload)
            return
        if path == "/debug/predictions" or path.startswith("/debug/predictions/"):
            if not self._debug_allowed():
                self._send_json(404, {"error": "Dashboard disabled"})
                return
            try:
                result = PREDICTION_SERVICE.get(path.removeprefix("/debug/predictions").strip("/"), parse_qs(parsed.query))
                self._send_json(200, result)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/debug/traces":
            if not self._debug_allowed():
                self._send_json(404, json_error("disabled", "Debug dashboard is disabled.", False))
                return
            query = parse_qs(parsed.query)
            try:
                limit = min(max(int((query.get("limit") or [str(TRACE_PAGE_SIZE)])[0]), 1), 500)
                before_id = max(int((query.get("before_id") or ["0"])[0]), 0)
                search = str((query.get("q") or [""])[0]).strip()[:200]
                if DATABASE_URL:
                    with db() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                SELECT id, payload FROM ai_debug_traces
                                WHERE (%s = 0 OR id < %s)
                                ORDER BY id DESC LIMIT %s
                                """,
                                (before_id, before_id, limit * (5 if search else 1) + 1),
                            )
                            rows = cur.fetchall()
                    scanned = rows
                    if search:
                        rows = [row for row in scanned if search.lower() in json.dumps(row["payload"], ensure_ascii=False).lower()]
                    has_more = len(scanned) > limit or (search and len(rows) < limit and bool(scanned))
                    page = rows[:limit]
                    self._send_json(200, {
                        "traces": [{**row["payload"], "trace_id": row["id"]} for row in page],
                        "next_before_id": (scanned[-1]["id"] if search else page[-1]["id"]) if has_more and scanned else None,
                        "persistent": True,
                    })
                else:
                    self._send_json(200, {"traces": list(TRACES)[:limit], "next_before_id": None, "persistent": False})
            except Exception as error:
                log_event("debug_trace_query_failed", error=str(error))
                self._send_json(200, {"traces": list(TRACES)[:TRACE_LIMIT], "next_before_id": None, "persistent": False})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/debug/"):
            if not self._debug_allowed():
                self._send_json(404, {"error": "Dashboard disabled"})
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self._send_json(403, {"error": "Dashboard requests must use the same origin."})
                return
            if int(self.headers.get("Content-Length", "0")) > 2_000_000:
                self._send_json(413, {"error": "Upload limit is 2 MB."})
                return
        try:
            request = self._read_json()
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if self.handle_daxview_post(path, request):
            return
        if path == "/debug/sandbox":
            message = str(request.get("message") or "").strip()
            if not message:
                self._send_json(400, {"error": "message is required"})
                return
            if len(message) > 4000:
                self._send_json(413, {"error": "Question must be 4,000 characters or fewer."})
                return
            history = request.get("history") if isinstance(request.get("history"), list) else []
            job_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            with SANDBOX_JOBS_LOCK:
                if len(SANDBOX_JOBS) >= 100:
                    for old_id, old_job in list(SANDBOX_JOBS.items()):
                        if old_job.get("status") in {"completed", "failed"}:
                            SANDBOX_JOBS.pop(old_id, None)
                            if len(SANDBOX_JOBS) < 80:
                                break
                if len(SANDBOX_JOBS) >= 100:
                    self._send_json(429, {"error": "Sandbox is busy. Wait for a running answer to finish, then retry."})
                    return
                SANDBOX_JOBS[job_id] = {"job_id": job_id, "request_id": request_id, "status": "queued"}
            SANDBOX_EXECUTOR.submit(process_sandbox_job, job_id, request_id, message, history)
            self._send_json(202, {"job_id": job_id, "request_id": request_id, "status": "queued"})
            return
        if path.startswith("/debug/predictions/"):
            try:
                result = PREDICTION_SERVICE.post(path.removeprefix("/debug/predictions/"), request)
                self._send_json(202 if "job_id" in result else 200, result)
            except (ValueError, TypeError) as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/debug/mcp-verify":
            text = str(request.get("ground_truth_yaml") or "")
            case_count = len(re.findall(r"^\s*-\s+name:", text, flags=re.MULTILINE))
            self._send_json(
                200,
                {
                    "status": "ready",
                    "checks": [
                        {
                            "name": "ground truth file loaded",
                            "status": "pass" if case_count else "skip",
                            "detail": f"{case_count} verification case(s) detected. Fill expected values before strict live assertions.",
                            "request_id": str(uuid.uuid4()),
                        }
                    ],
                },
            )
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
    try:
        ensure_debug_trace_schema()
    except Exception as error:
        print(json.dumps({"event": "debug_trace_schema_unavailable", "error": str(error)}), flush=True)
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"AI-Server listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAI-Server stopped")
    finally:
        server.server_close()
