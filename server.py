"""EMS-only chatbot API with Ollama, PostgreSQL, and pgvector RAG."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import base64
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
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from langchain_core.runnables import RunnableLambda
import psycopg
from psycopg.rows import dict_row
from prediction_lab import PredictionService, daily_dataset
from ems_contracts import measurement_context, metric_clarification, percentage_difference, question_metrics, result_problem
from contracts.tool_schemas import TOOL_ARGUMENT_KEYS, TOOL_SCHEMAS, validate_tool_arguments
from memory.conversation import load_conversation_history as load_conversation_history_from_db
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
OLLAMA_EMBEDDING_TIMEOUT = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
RAG_MATCH_LIMIT = int(os.getenv("RAG_MATCH_LIMIT", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
TRACE_LIMIT = int(os.getenv("CHATBOT_TRACE_LIMIT", "25"))
AI_DEBUG_DASHBOARD_ENABLED = os.getenv("AI_DEBUG_DASHBOARD_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AI_REFINE_MCP_WITH_MODEL = os.getenv("AI_REFINE_MCP_WITH_MODEL", "true").lower() in {"1", "true", "yes", "on"}
AI_MCP_FALLBACKS_ENABLED = os.getenv("AI_MCP_FALLBACKS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
AI_CHARTS_ENABLED = os.getenv("AI_CHARTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
AI_PLANNER_ENABLED = os.getenv("AI_PLANNER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AI_PLANNER_SHADOW_MODE = os.getenv("AI_PLANNER_SHADOW_MODE", "false").lower() in {"1", "true", "yes", "on"}
AI_CHAT_MEMORY_ENABLED = os.getenv("AI_CHAT_MEMORY_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
PLANNER_MODEL = os.getenv("PLANNER_MODEL", OLLAMA_MODEL)
PLANNER_TIMEOUT = int(os.getenv("PLANNER_TIMEOUT", "15"))
DEMAND_RANKING_MAX_DEVICES = int(os.getenv("DEMAND_RANKING_MAX_DEVICES", "20"))
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
DAXVIEW_API_ENABLED = os.getenv("DAXVIEW_API_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_API_BASE_URL = os.getenv("DAXVIEW_API_BASE_URL", "https://event.daxview.com/api").strip().rstrip("/")
DAXVIEW_API_AUTH_TOKEN = os.getenv("DAXVIEW_API_AUTH_TOKEN", "").strip()
DAXVIEW_API_SESSION_COOKIE = os.getenv("DAXVIEW_API_SESSION_COOKIE", "").strip()
DAXVIEW_API_USERNAME = os.getenv("DAXVIEW_API_USERNAME", "").strip()
DAXVIEW_API_PASSWORD = os.getenv("DAXVIEW_API_PASSWORD", "").strip()
DAXVIEW_API_TIMEOUT = int(os.getenv("DAXVIEW_API_TIMEOUT", "20"))
DAXVIEW_API_DEBUG_RESPONSE = os.getenv("DAXVIEW_API_DEBUG_RESPONSE", "false").lower() in {"1", "true", "yes", "on"}
DAXVIEW_API_DEBUG_RESPONSE_LIMIT = int(os.getenv("DAXVIEW_API_DEBUG_RESPONSE_LIMIT", "4000"))
DAXVIEW_API_ENERGY_RANKING_PATH = os.getenv(
    "DAXVIEW_API_ENERGY_RANKING_PATH",
    "/core/billing/sites/{site_id}/demand/daily/",
).strip()
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
    "device_energy_ranking",
    "site_energy_summary",
    "alarm_frequency_summary",
    "site_metadata_summary",
    "site_device_list",
    "telemetry_timeseries",
    "telemetry_metric_catalog",
    "latest_telemetry_snapshot",
    "active_alarm_summary",
    "meter_status_summary",
    "energy_comparison_summary",
    "data_availability_summary",
    "alarm_detail_lookup",
    "power_quality_summary",
    "demand_peak_summary",
    "tariff_cost_summary",
    "device_energy_breakdown",
    "energy_forecast",
    "anomaly_detection_summary",
    "report_summary",
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
        "which alarms occurred", "top alarms",
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
    "device_energy_ranking": "Rank devices by energy consumption for a site/window.",
    "site_energy_summary": "Summarize site energy values over a time window.",
    "alarm_frequency_summary": "Rank historical alarm types by count/severity.",
    "site_metadata_summary": "Return site/building/device/meter metadata counts.",
    "site_device_list": "List devices/meters in the scoped site/building.",
    "telemetry_timeseries": "Return a device metric trend such as voltage/current/energy.",
    "telemetry_metric_catalog": "Discover available telemetry metrics and coverage by device.",
    "latest_telemetry_snapshot": "Return latest persisted telemetry readings by device/metric.",
    "active_alarm_summary": "Return currently active/open alarms.",
    "meter_status_summary": "Return device/meter online/offline/stale state.",
    "energy_comparison_summary": "Compare two energy periods.",
    "data_availability_summary": "Check data coverage/missing readings.",
    "alarm_detail_lookup": "Look up one alarm by alarm ID.",
    "power_quality_summary": "Summarize voltage sag/swell/THD/power-quality events.",
    "demand_peak_summary": "Return max/peak demand for site/building/device.",
    "tariff_cost_summary": "Estimate tariff or cost for energy use.",
    "device_energy_breakdown": "Break down site energy by device/building.",
    "energy_forecast": "Forecast future energy from historical data.",
    "anomaly_detection_summary": "Find abnormal readings or usage patterns.",
    "report_summary": "Generate a compact EMS management report.",
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
        TRACES.appendleft(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **redact_debug_value(record),
            }
        )
    print(json.dumps(record, ensure_ascii=False), flush=True)


def debug_trace_event(event: str, **fields: object) -> None:
    if not AI_DEBUG_DASHBOARD_ENABLED:
        return
    TRACES.appendleft(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **redact_debug_value(fields),
        }
    )


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


def requested_top_limit(message: str) -> int | None:
    match = re.search(r"\btop\s+(\d{1,2}|five|ten)\b", message, re.IGNORECASE)
    if not match:
        return None
    value = {"five": 5, "ten": 10}.get(match.group(1).lower())
    return value if value is not None else min(max(int(match.group(1)), 1), 20)


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


def wants_lowest_consumer(message: str) -> bool:
    lowered = message.lower()
    return bool(re.search(r"\b(?:lowest|least|minimum|smallest)\b", lowered) and re.search(r"\b(?:device|meter|consumer|energy|consume|consumption|usage)\b", lowered))


def wants_top_lowest_sum(message: str) -> bool:
    lowered = message.lower()
    return wants_lowest_consumer(message) and bool(re.search(r"\b(?:sum|total|add|combined|their\s+2|both)\b", lowered))


def wants_all_devices(message: str) -> bool:
    return bool(re.search(r"\b(?:all|every)\s+(?:(?:of\s+)?the\s+)?devices?\b", message, re.IGNORECASE))


def wants_device_inventory(message: str) -> bool:
    lowered = message.lower()
    if wants_device_usage_ranking(message) or wants_highest_demand_device(message):
        return False
    return bool(
        re.search(r"\b(?:list|show|get|give)\b.*\b(?:devices?|meters?|device\s*ids?)\b", lowered)
        or re.search(r"\bwhich\s+(?:devices?|meters?)\b.*\b(?:have data|available|online|offline|status|best|suitable)\b", lowered)
        or re.search(r"\b(?:devices?|meters?)\b.*\b(?:in|under|for)\s+(?:this|selected|current)\s+site\b", lowered)
        or re.search(r"\b(?:best|suitable)\b.*\b(?:voltage|current|trend|testing)\b", lowered)
    )


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
    if match and match.group(1) in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        return match.group(1)
    for tool_name in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        if re.search(rf"\b(?:test|call|run|use)\s+{re.escape(tool_name)}\b", lowered):
            return tool_name
    return None


def select_historical_operations(message: str) -> list[str]:
    explicit_tool = explicit_mcp_tool_request(message)
    if explicit_tool:
        return [explicit_tool]
    lowered = message.lower()
    operations = select_historical_operations_without_device_timeseries(message)
    if wants_device_metric_timeseries(message):
        operations = [name for name in operations if name != "site_energy_summary"]
        if "telemetry_timeseries" not in operations:
            operations.append("telemetry_timeseries")
    if wants_available_parameters_follow_up(message):
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
        operations = [name for name in operations if name != "site_device_list"]
        if "device_energy_breakdown" not in operations:
            operations.append("device_energy_breakdown")
        if not re.search(r"\b(?:daily|site energy|site-wide|overall|total site|site total|site usage)\b", lowered):
            operations = [name for name in operations if name != "site_energy_summary"]
        if any(term in lowered for term in ("offline", "online", "status")) or wants_all_devices(message):
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


def plan_turn_with_llm(message: str, history: list[dict], context: dict, request_id: str):
    messages = [
        {"role": "system", "content": planner_system_prompt(DAXVIEW_TOOL_DESCRIPTIONS)},
    ]
    for turn in history[-6:]:
        if turn.get("user_message"):
            messages.append({"role": "user", "content": str(turn["user_message"])[:1000]})
        if turn.get("assistant_reply"):
            messages.append({"role": "assistant", "content": str(turn["assistant_reply"])[:900]})
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
            match = re.search(r"\d+", str(ref))
            if match:
                merged["device_id"] = int(match.group(0))
                break
    return merged


def choose_historical_operations(message: str, context: dict, turn_id: str, session_id: str, request_id: str) -> tuple[list[str], dict, dict | None]:
    regex_operations = select_historical_operations(message)
    if not (AI_PLANNER_ENABLED or AI_PLANNER_SHADOW_MODE):
        return regex_operations, context, None
    history = load_conversation_history(session_id, turn_id) if AI_CHAT_MEMORY_ENABLED else []
    try:
        plan, latency_ms = plan_turn_with_llm(message, history, context, request_id)
        planned_context = planner_slot_context(message, context, plan)
        planned_tools = [tool for tool in plan.tools if tool in DAXVIEW_ALLOWED_HISTORICAL_TOOLS]
        if AI_PLANNER_SHADOW_MODE or not AI_PLANNER_ENABLED:
            log_event(
                "planner_shadow_compare",
                request_id=request_id,
                regex_tools=regex_operations,
                planner_tools=planned_tools,
                agreement=regex_operations == planned_tools,
                latency_ms=latency_ms,
                confidence=plan.confidence,
            )
            return regex_operations, context, {
                "status": "shadow",
                "tools": [{"tool": tool, "arguments": {}} for tool in planned_tools],
                "slots": planned_context,
            }
        if plan.clarification and plan.clarification.question:
            return [], {**planned_context, "_planner_clarification": plan.clarification.question}, {
                "status": "clarification",
                "tools": [],
                "slots": planned_context,
            }
        return planned_tools or regex_operations, planned_context, {
            "status": "ok",
            "tools": [{"tool": tool, "arguments": {}} for tool in planned_tools],
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
    local_now = end.astimezone(timezone(timedelta(hours=8)))
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
        args["limit"] = int(context.get("limit") or 100)
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
    elif operation_id == "active_alarm_summary":
        args["limit"] = int(context.get("limit") or 50)
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
    arguments = validate_tool_arguments(operation_id, arguments)
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
        }
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
            plan = json.loads(response.read().decode("utf-8"))
            log_event(
                "daxview_data_plan_response",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                has_authorization=bool(plan.get("authorization_id")),
                arguments=redact_debug_value(plan.get("arguments") if isinstance(plan.get("arguments"), dict) else {}),
            )
            debug_trace_event(
                "daxview_data_plan_response_debug",
                request_id=request_id,
                turn_id=turn_id,
                operation_id=operation_id,
                has_authorization=bool(plan.get("authorization_id")),
                arguments=redact_debug_value(plan.get("arguments") if isinstance(plan.get("arguments"), dict) else {}),
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
        raise


def call_authorized_historical_tool(operation_id: str, authorization_id: str, arguments: dict, request_id: str) -> dict:
    if operation_id not in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        raise ValueError("historical operation is not allowlisted")
    arguments = validate_tool_arguments(operation_id, arguments)
    result = call_daxview_mcp_tool(operation_id, {"authorization_id": authorization_id, **arguments}, request_id)
    issue = result_problem(mcp_structured_result(result)) or result_problem(result)
    if issue:
        raise RuntimeError(f"{operation_id} could not return valid data: {issue}")
    if AI_DEBUG_DASHBOARD_ENABLED and operation_id == "site_energy_summary" and arguments.get("bucket") == "day":
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


def daxview_api_headers() -> dict:
    headers = {"Accept": "application/json"}
    if DAXVIEW_API_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {DAXVIEW_API_AUTH_TOKEN}"
    if DAXVIEW_API_SESSION_COOKIE:
        headers["Cookie"] = DAXVIEW_API_SESSION_COOKIE
    return headers


def daxview_api_path(path: str, **values) -> str:
    return "/" + path.format(**values).lstrip("/")


def daxview_api_url(path: str, query: dict | None = None) -> str:
    url = f"{DAXVIEW_API_BASE_URL}{path}"
    if not query:
        return url
    from urllib.parse import urlencode
    pairs = [(key, str(value)) for key, value in query.items() if value is not None]
    return f"{url}?{urlencode(pairs)}" if pairs else url


def call_daxview_api(method: str, path: str, request_id: str, query: dict | None = None, payload: dict | None = None) -> dict:
    if not DAXVIEW_API_ENABLED:
        raise RuntimeError("DaxView API provider is disabled")
    if not DAXVIEW_API_BASE_URL:
        raise RuntimeError("DAXVIEW_API_BASE_URL is not configured")
    headers = daxview_api_headers()
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if DAXVIEW_API_USERNAME and DAXVIEW_API_PASSWORD and "Authorization" not in headers:
        token = base64.b64encode(f"{DAXVIEW_API_USERNAME}:{DAXVIEW_API_PASSWORD}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    url = daxview_api_url(path, query)
    log_event("daxview_api_request", request_id=request_id, method=method.upper(), path=path)
    request = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=DAXVIEW_API_TIMEOUT) as response:
            body = response.read().decode("utf-8", errors="replace")
            if not body:
                return {}
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body}
            if DAXVIEW_API_DEBUG_RESPONSE:
                log_event(
                    "daxview_api_response_debug",
                    request_id=request_id,
                    path=path,
                    sample=debug_json_sample(parsed, DAXVIEW_API_DEBUG_RESPONSE_LIMIT),
                )
            return parsed if isinstance(parsed, dict) else {"items": parsed}
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
        log_event("daxview_api_error", request_id=request_id, method=method.upper(), path=path, status=error.code, response=safe_body)
        raise


def first_nested_list(value) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    direct = first_list(value, ("rows", "items", "results", "data", "devices", "meters", "parameters", "series", "points", "daily", "records"))
    if direct:
        return direct
    for nested in value.values():
        rows = first_nested_list(nested)
        if rows:
            return rows
    return []


def normalize_api_device(row: dict) -> dict:
    device_id = first_value(row, ("device_id", "id", "deviceId", "meter_id", "meterId"))
    return {
        "device_id": device_id,
        "device_name": first_value(row, ("device_name", "name", "label", "deviceName", "meter_name")) or f"Device {device_id or 'unknown'}",
        "device_type": first_value(row, ("device_type", "type", "deviceType")),
        "building_id": first_value(row, ("building_id", "buildingId")),
        "building_name": first_value(row, ("building_name", "buildingName")),
        "status": first_value(row, ("status", "connection_status", "data_status", "state")),
        "last_seen": first_value(row, ("last_seen", "lastSeen", "last_reading_at", "updated_at")),
        "manufacturer": row.get("manufacturer"),
        "meter_model": first_value(row, ("meter_model", "model", "meterModel")),
    }


def normalize_api_devices(payload: dict, arguments: dict) -> dict:
    rows = [row for row in first_nested_list(payload) if isinstance(row, dict)]
    site_id = arguments.get("site_id")
    if site_id:
        site_text = str(site_id)
        filtered = [row for row in rows if str(first_value(row, ("site_id", "site", "siteId")) or site_text) == site_text]
        rows = filtered or rows
    devices = [normalize_api_device(row) for row in rows]
    limit = int(arguments.get("limit") or 100)
    return {"site_id": site_id, "devices": devices[:limit], "device_count": len(devices), "truncated": len(devices) > limit}


def normalize_api_energy_ranking(payload: dict, arguments: dict) -> dict:
    rows = [row for row in first_nested_list(payload) if isinstance(row, dict)]
    normalized = []
    for row in rows:
        device_id = first_value(row, ("device_id", "deviceId", "meter_id", "meterId", "id"))
        value = first_value(row, (
            "value", "kwh", "energy", "energy_kwh", "total_kwh", "total_energy_kwh",
            "consumption", "consumption_kwh", "usage_kwh", "import_kwh", "daily_kwh",
        ))
        if value is None:
            continue
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        normalized.append({
            **row,
            "device_id": device_id,
            "device_name": first_value(row, ("device_name", "deviceName", "meter_name", "name", "label")) or f"Device {device_id or 'unknown'}",
            "value": amount,
            "unit": first_value(row, ("unit", "reading_unit")) or "kWh",
        })
    normalized.sort(key=lambda row: float(row.get("value") or 0), reverse=True)
    limit = int(arguments.get("limit") or 5)
    return {
        "site_id": arguments.get("site_id"),
        "metric": "energy",
        "unit": "kWh",
        "rankings": normalized[:limit],
        "device_count": len(normalized),
        "calculation": "direct_daxview_api_rows",
    }


def run_daxview_api_operation(operation_id: str, arguments: dict, request_id: str) -> dict:
    if operation_id == "site_device_list":
        query = {
            "site_id": arguments.get("site_id"),
            "site": arguments.get("site_id"),
            "building_id": arguments.get("building_id"),
            "limit": arguments.get("limit"),
        }
        payload = call_daxview_api("GET", "/core/devices/", request_id, query=query)
        data = normalize_api_devices(payload, arguments)
        return {"structuredContent": {"status": "ok", "operation_id": operation_id, "data": data}}
    if operation_id in {"device_energy_breakdown", "device_energy_ranking", "telemetry_top_consumers"}:
        path = daxview_api_path(DAXVIEW_API_ENERGY_RANKING_PATH, site_id=arguments["site_id"])
        query = {
            "site_id": arguments.get("site_id"),
            "building_id": arguments.get("building_id"),
            "from": arguments.get("start_time") or arguments.get("start"),
            "to": arguments.get("end_time") or arguments.get("end"),
            "start": arguments.get("start_time") or arguments.get("start"),
            "end": arguments.get("end_time") or arguments.get("end"),
            "timeZone": arguments.get("timezone"),
            "timezone": arguments.get("timezone"),
            "limit": arguments.get("limit"),
        }
        payload = call_daxview_api("GET", path, request_id, query=query)
        data = normalize_api_energy_ranking(payload, arguments)
        return {"structuredContent": {"status": "ok", "operation_id": "device_energy_ranking", "data": data}}
    raise ValueError(f"DaxView API provider does not support {operation_id} yet")


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
        top_name = top_row.get("device_name") or top_row.get("name") or top_row.get("label") or f"Device {top_row.get('device_id', 'unknown')}"
        low_name = low_row.get("device_name") or low_row.get("name") or low_row.get("label") or f"Device {low_row.get('device_id', 'unknown')}"
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
            or f"Device {row.get('device_id', 'unknown')}"
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
    complete_window = completed_daily_range(message) is not None and bool(arguments and arguments.get("bucket") == "day")
    if isinstance(buckets, list) and buckets:
        if has_specific_months:
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
            if complete_window and arguments:
                start_day = parse_datetime(arguments.get("start"))
                end_day = parse_datetime(arguments.get("end"))
                if start_day and end_day:
                    first_day = start_day.astimezone(timezone(timedelta(hours=8))).date().isoformat()
                    last_day = end_day.astimezone(timezone(timedelta(hours=8))).date().isoformat()
                    if not first_day <= str(label) < last_day:
                        continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                daily_values[str(label)] = float(value)
            if has_specific_months:
                continue
            if has_specific_dates and not include_range_summary and str(label) not in requested_date_set:
                continue
            if shown_rows < 31:
                row_unit = item.get("unit") or unit
                row_factor, row_display_unit, row_precision = requested_energy_unit(message, row_unit)
                lines.append(f"- {label}: {format_energy_value(value, row_factor, row_display_unit, row_precision if row_factor != 1.0 else precision)}")
                shown_rows += 1
        if not has_specific_months and len(daily_values) > 31:
            lines.append(f"Showing 31 of {len(daily_values)} daily row(s).")
    elif total_value is None:
        lines.append("No energy values were returned for this site and time range.")
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
    compliance_note = format_compliance_context(build_compliance_context(message, ["site_energy_summary"]))
    if compliance_note:
        lines.append("")
        lines.append(compliance_note)
    return "\n".join(lines)


def summarize_alarm_frequency(data: dict, arguments: dict | None = None) -> str:
    rows = data.get("rows") or data.get("alarms") or data.get("items")
    rows = rows if isinstance(rows, list) else []
    lines = ["Alarm frequency summary returned by Daxview MCP:"]
    if not rows:
        lines.append("No alarm-frequency rows were returned for this site and time range.")
    limit = min(max(int((arguments or {}).get("limit") or 10), 1), 20)
    for index, row in enumerate(rows[:limit], 1):
        if not isinstance(row, dict):
            continue
        name = row.get("alarm_name") or row.get("name") or row.get("type") or row.get("severity") or "Alarm"
        count = next((row[key] for key in ("count", "frequency", "total", "value") if row.get(key) is not None), None)
        detail = f"{index}. {name} - {format_number(count, 0)} occurrence(s)"
        if row.get("severity"):
            detail += f", severity {row['severity']}"
        latest = row.get("latest_occurrence") or row.get("last_seen") or row.get("latest_at")
        if latest:
            detail += f", latest {latest}"
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
    if site.get("name"):
        lines.append(f"Site: {site.get('name')} (ID {site.get('id', site.get('site_id', 'unknown'))})")
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


def summarize_site_devices(data: dict) -> str:
    devices = first_list(data, ("devices", "rows", "items", "meters"))
    lines = []
    total = data.get("device_count") or data.get("meter_count") or data.get("row_count") or len(devices)
    if data.get("site_name") or data.get("building_name"):
        scope = data.get("building_name") or data.get("site_name")
        lines.append(f"Scope: {scope}")
    if total:
        lines.append(f"Total devices returned: {total}")
    if not devices:
        lines.append("No devices were returned.")
        return "\n".join(lines)

    status_counts: dict[str, int] = {}
    for device in devices:
        if isinstance(device, dict):
            status = str(device.get("status") or device.get("connection_status") or device.get("state") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts:
        lines.append("Status counts:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status.title()}: {count}")

    lines.append("Sample devices:")
    for index, device in enumerate(devices[:10], 1):
        if not isinstance(device, dict):
            lines.append(f"- {index}. {device}")
            continue
        status = str(device.get("status") or device.get("connection_status") or device.get("state") or "unknown")
        name = device.get("device_name") or device.get("name") or device.get("meter_name") or "Unnamed device"
        device_id = device.get("device_id") or device.get("id") or device.get("meter_id") or "unknown"
        device_type = device.get("device_type") or device.get("type") or device.get("model")
        last_seen = device.get("last_updated") or device.get("last_seen") or device.get("last_telemetry_at")
        metrics = first_value(device, (
            "supported_metrics", "metrics", "metric_names", "telemetry_metrics",
            "parameters", "parameter_names", "available_parameters", "capabilities",
        ))
        details = [f"ID {device_id}", status.title()]
        if device_type:
            details.append(str(device_type))
        if device.get("has_data") is not None:
            details.append(f"has data {device.get('has_data')}")
        if last_seen:
            details.append(f"last seen {last_seen}")
        lines.append(f"- {index}. {name} ({', '.join(details)})")
        if metrics:
            if isinstance(metrics, list):
                metric_text = ", ".join(str(item) for item in metrics[:12])
                if len(metrics) > 12:
                    metric_text += f", +{len(metrics) - 12} more"
            else:
                metric_text = str(metrics)
            lines.append(f"  Parameters: {metric_text}")

    if len(devices) > 10:
        lines.append(f"Showing 10 of {len(devices)} device(s).")
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
        device_id = row.get("device_id") or row.get("id") or "unknown"
        status = row.get("status") or row.get("data_status") or "unknown"
        last_seen = row.get("last_seen") or row.get("last_updated") or row.get("last_telemetry_at")
        last_seen_text = f", last seen {last_seen}" if last_seen else ""
        lines.append(f"- {index}. {name} (ID {device_id}) - {status}{last_seen_text}")
    if len(rows) > len(focus_rows):
        lines.append(f"Showing {len(focus_rows)} of {len(rows)} device(s).")
    return "\n".join(lines)


def summarize_generic_tool(operation_id: str, data: dict) -> str:
    title = operation_id.replace("_", " ").title()
    lines = [f"{title} returned by DaxView MCP:"]
    if not data:
        lines.append("No structured data was returned.")
        return "\n".join(lines)

    scalar_keys = (
        "status",
        "site_id",
        "site_name",
        "building_id",
        "building_name",
        "device_id",
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
            "device_id",
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
                    if key in used_keys or value is None or isinstance(value, (dict, list)):
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


def summarize_demand_peak(data: dict, message: str, arguments: dict | None = None) -> str:
    ranked_devices = data.get("ranked_devices") if isinstance(data.get("ranked_devices"), list) else []
    if ranked_devices:
        lines = [f"Highest demand devices ({format_time_window(data, arguments)}):"]
        top_peak = ranked_devices[0].get("peak_kw") if ranked_devices else None
        for index, row in enumerate(ranked_devices[:requested_top_limit(message) or 5], 1):
            diff = ""
            if index > 1 and top_peak not in (None, 0):
                pct = percentage_difference(top_peak, row.get("peak_kw"))
                if pct is not None:
                    diff = f" ({format_number(pct, 1)}% below top)"
            status = f", {row.get('status')}" if row.get("status") else ""
            lines.append(
                f"{index}. {row.get('device_name') or 'Device'} (ID {row.get('device_id')}): "
                f"{format_number(row.get('peak_kw'), 2)} kW{diff}{status}"
            )
        if data.get("device_count"):
            lines.append(f"Compared {data.get('device_count')} device(s) with returned demand data.")
        return "\n".join(lines)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
    rows = first_list(data, ("rows", "items", "results", "devices"))
    source = summary
    peak_keys = ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw")
    if first_value(source, peak_keys) is None:
        source = next((row for row in rows if isinstance(row, dict) and first_value(row, peak_keys) is not None), summary)
    device_id = first_value(source, ("device_id", "meter_id")) or (arguments or {}).get("device_id")
    device_name = first_value(source, ("device_name", "meter_name", "name"))
    heading = f"Max demand for {device_name or f'device ID {device_id}'}" if device_id or device_name else "Max demand"
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
    if operation_id in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"} and wants_device_usage_ranking(message):
        return summarize_top_consumers(data, arguments, message)
    if operation_id == "site_energy_summary":
        return summarize_site_energy(data, message, arguments)
    if operation_id == "alarm_frequency_summary":
        return summarize_alarm_frequency(data, arguments)
    if operation_id == "energy_forecast":
        return summarize_energy_forecast(data)
    if operation_id == "active_alarm_summary":
        return summarize_active_alarms(data)
    if operation_id == "site_metadata_summary":
        return summarize_site_metadata(data)
    if operation_id == "site_device_list":
        return summarize_site_devices(data)
    if operation_id == "meter_status_summary":
        return summarize_meter_status(data)
    if operation_id == "demand_peak_summary":
        return summarize_demand_peak(data, message, arguments)
    if operation_id in DAXVIEW_ALLOWED_HISTORICAL_TOOLS:
        return summarize_generic_tool(operation_id, data)
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

    top_result = next((item for item in results if item["operation_id"] in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}), None)
    device_result = next((item for item in results if item["operation_id"] == "site_device_list"), None)
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
            name = row.get("device_name") or row.get("name") or f"Device {device_id or 'unknown'}"
            status = (matched or {}).get("status") or (matched or {}).get("connection_status") or row.get("status")
            lines.append(f"{index}. {name} (ID {device_id or 'unknown'}): {status or 'status unavailable'}")
        if not measured:
            lines.append("No measured devices could be matched to current status.")
        if wants_all_devices(message):
            unranked = [device for device in devices if isinstance(device, dict)
                        and str(device.get("device_id") or device.get("id") or device.get("meter_id")) not in ranked_ids]
            if unranked:
                lines.append("Other devices (usage not returned, so not ranked):")
                for device in unranked:
                    name = device.get("device_name") or device.get("name") or "Unnamed device"
                    device_id = device.get("device_id") or device.get("id") or device.get("meter_id") or "unknown"
                    lines.append(f"- {name} (ID {device_id})")
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
    return "\n\n".join(sections)


def refine_historical_answer_with_model(message: str, results: list[dict], deterministic_answer: str, request_id: str) -> str:
    if not AI_REFINE_MCP_WITH_MODEL:
        return deterministic_answer
    prompt = build_mcp_refine_prompt(message, results, deterministic_answer, request_id)
    primary_answer = ensure_historical_answer_coverage(
        message, results, deterministic_answer,
        run_mcp_refine_model(prompt, request_id, CHAT_MODEL, deterministic_answer, role="primary"),
        request_id,
    )
    if not AI_COMPARE_MODEL_ENABLED:
        return primary_answer
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
    return primary_answer


def ensure_historical_answer_coverage(
    message: str, results: list[dict], deterministic_answer: str, model_answer: str, request_id: str,
) -> str:
    if not model_answer:
        return deterministic_answer
    generic_failure_patterns = (
        r"\bempty input\b",
        r"\btesting the system\b",
        r"\bhow can i assist you today\b",
        r"\bplease let me know if you have any questions\b",
        r"\bneed help with something specific\b",
    )
    if results and any(re.search(pattern, model_answer, flags=re.IGNORECASE) for pattern in generic_failure_patterns):
        log_event("historical_answer_coverage_fallback", request_id=request_id, missing_items="generic_model_reply")
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
Answer each part of the user's question using the matching tool result. Never present energy consumption as an alarm count.
Do not include standards, protocols, or library background unless the user asks for them.
For predictions, label every future value as an estimate and retain the method, coverage, and backtest error.
Answer every numbered request below. If a value was not returned, state that for the affected request.
When daily values are requested, retain every date and its value from the deterministic draft.
Keep the answer concise but useful:
- Start with the direct answer.
- Use short sections and bullets.
- Mention the tool/data limitation only if a requested value is missing.
- Hide backend/internal terms such as MCP, row_count, source_count, aggregation, authorization_id, request_id, and JSON field names.
- If the user asks for max demand, explain whether the result is site-level, building-level, or device-level based on the data. If device-level detail is missing, say which follow-up question would retrieve it.
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


def chart_from_historical_result(operation_id: str, result: dict, arguments: dict | None = None) -> dict | None:
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
        for row in rows[:10]:
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
            value = chart_value(row, ("count", "frequency", "total", "value"))
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
    if operation_id == "meter_status_summary":
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
            value = chart_value(data, ("peak_kw", "max_kw", "demand_kw", "value", "peak", "maximum"))
            if value is not None:
                labels = [str(data.get("peak_time") or data.get("timestamp") or "Peak demand")]
                values = [value]
        return build_chart_spec("bar", "Peak Demand", labels, values, data.get("unit") or "kW", "Demand")
    if operation_id == "telemetry_timeseries":
        rows = first_list(data, ("rows", "series", "buckets", "data", "points"))
        pairs = [(str(row.get("timestamp") or row.get("date") or row.get("bucket") or ""),
                  chart_value(row, ("value", "average", "avg", "reading")))
                 for row in rows[:1000] if isinstance(row, dict)]
        pairs = sorted((label, value) for label, value in pairs if label and value is not None)
        metric = (arguments or {}).get("metric", "Telemetry")
        return build_chart_spec("line", f"{str(metric).title()} History", [p[0] for p in pairs],
                                [p[1] for p in pairs], str(data.get("unit") or ""), str(metric))
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


def build_charts_from_historical_results(results: list[dict], preview: bool = False) -> list[dict]:
    if not AI_CHARTS_ENABLED and not preview:
        return []
    charts = []
    for item in results:
        chart = chart_from_historical_result(
            str(item.get("operation_id")),
            item.get("result") if isinstance(item.get("result"), dict) else {},
            item.get("arguments") if isinstance(item.get("arguments"), dict) else None,
        )
        if chart:
            charts.append(chart)
    return charts


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
    "follow up",
    "follow-up",
    "reply to this",
    "reply to this answer",
    "about this answer",
    "based on this answer",
    "use previous answer",
    "use the previous answer",
    "based on previous",
    "based on the previous",
    "from this result",
    "from that result",
    "explain this result",
    "make it cleaner",
    "summarize this",
    "summarise this",
    "turn this into",
    "what should i check",
    "what should i do next",
)


def is_follow_up_message(message: str) -> bool:
    lowered = message.lower().strip()
    if re.match(r"^device\s*(?:id\s*)?[:#-]?\s*\d+\b", lowered):
        return True
    if re.fullmatch(r"(?:why|how|what about that|and that|same one)[?.! ]*", lowered):
        return True
    if re.match(r"^(?:and\s+)?(?:what about|how about|for the same|on the same|can you also|show me more about|which day|which one|which was)\b", lowered):
        return True
    return any(phrase in lowered for phrase in FOLLOW_UP_PHRASES if phrase != "why")


def current_turn_identity(turn_id: str) -> dict | None:
    if not DATABASE_URL:
        return None
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT deployment_id, company_id, user_id, conversation_id
                FROM daxview_turns
                WHERE id = %s
                """,
                (turn_id,),
            )
            return cur.fetchone()


def load_conversation_history(conversation_id: str, current_turn_id: str, limit: int = 6) -> list[dict]:
    if not DATABASE_URL:
        return []
    identity = current_turn_identity(current_turn_id)
    if not identity:
        return []
    with db() as conn:
        return load_conversation_history_from_db(conn, conversation_id, current_turn_id, identity, limit)


def previous_daxview_turn(conversation_id: str, current_turn_id: str) -> dict | None:
    identity = current_turn_identity(current_turn_id)
    if not identity:
        return None
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_message, context, assistant_reply, resolved_plan
                FROM daxview_turns
                WHERE conversation_id = %s
                  AND id <> %s
                  AND deployment_id = %s
                  AND company_id = %s
                  AND user_id = %s
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (
                    conversation_id,
                    current_turn_id,
                    identity["deployment_id"],
                    identity["company_id"],
                    identity["user_id"],
                ),
            )
            rows = cur.fetchall()
    return next(
        (row for row in rows if row.get("user_message") and not str(row["user_message"]).lower().startswith("requested context")),
        None,
    )


def first_device_id_from_text(text: str) -> int | None:
    if not text:
        return None
    patterns = (
        r"^\s*1\.\s+.+?\(ID\s+(\d+)\)",
        r"Highest energy-consuming device:.+?\(ID\s+(\d+)\)",
        r"Highest demand devices?.+?\(ID\s+(\d+)\)",
        r"\b(?:device|meter)\s+ID\s+(\d+)\b",
        r"\(ID\s+(\d+)\)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            return int(match.group(1))
    return None


def wants_available_parameters_follow_up(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\b(?:continue|based on this|other parameter|other parameters|available parameter|available parameters|what else|anything else)\b", lowered)
        and re.search(r"\b(?:device|parameter|metric|voltage|current|telemetry|available|continue)\b", lowered)
    )


def follow_up_metric_from_message(message: str) -> str | None:
    metrics = question_metrics(message)
    if metrics:
        return metrics[0]
    lowered = message.lower()
    if re.search(r"\b(?:parameter|parameters|metric|metrics|available|what else)\b", lowered):
        return "energy"
    return None


def resolve_follow_up_message(turn_id: str, message: str, context: dict, conversation_id: str, request_id: str) -> tuple[str, dict]:
    previous = previous_daxview_turn(conversation_id, turn_id)
    if not previous or not previous.get("user_message"):
        return message, context
    previous_message = str(previous["user_message"])
    previous_answer = str(previous.get("assistant_reply") or "")
    previous_context = previous.get("context") if isinstance(previous.get("context"), dict) else {}
    previous_plan = previous.get("resolved_plan") if isinstance(previous.get("resolved_plan"), dict) else {}
    reusable = previous_plan.get("status") in {"ok", "no_data"} or not previous_plan
    site_changed = context.get("site_id") and str(context["site_id"]) != str(previous_context.get("site_id"))
    merged_context = {**previous_context, **context} if not site_changed else dict(context)
    if site_changed:
        merged_context = dict(context)
    elif reusable and AI_CHAT_MEMORY_ENABLED:
        slots = previous_plan.get("slots") if isinstance(previous_plan.get("slots"), dict) else {}
        for key in ("device_id", "building_id", "metric", "_time_window"):
            if key not in merged_context and slots.get(key) is not None:
                merged_context[key] = slots[key]
    if has_time_scope(message):
        merged_context.pop("_time_window", None)
    if requested_device_id(message) is not None:
        merged_context["device_id"] = requested_device_id(message)
    if not site_changed and not merged_context.get("device_id") and requested_device_id(message) is None:
        previous_device_id = requested_device_id(previous_message) or first_device_id_from_text(previous_answer)
        if previous_device_id is not None:
            merged_context["device_id"] = previous_device_id
    if wants_available_parameters_follow_up(message):
        if not merged_context.get("device_id"):
            inferred_device_id = first_device_id_from_text(previous_answer) or requested_device_id(previous_message)
            if inferred_device_id:
                merged_context["device_id"] = inferred_device_id
        if follow_up_metric_from_message(message):
            merged_context["metric"] = follow_up_metric_from_message(message)
        if not has_time_scope(message):
            merged_context.setdefault("_time_window", previous_context.get("_time_window") or default_historical_range())
        resolved = message
        if not select_historical_operations(resolved):
            resolved = f"Check data availability and available telemetry parameters for device ID {merged_context.get('device_id', '')}".strip()
        log_event(
            "daxview_follow_up_resolved",
            request_id=request_id,
            turn_id=turn_id,
            previous_turn_id=str(previous.get("id")),
            inferred_device_id=merged_context.get("device_id"),
        )
        return resolved, merged_context
    if not is_follow_up_message(message) and (select_historical_operations(message) or not AI_CHAT_MEMORY_ENABLED):
        return message, merged_context
    if select_historical_operations(message):
        resolved = message
        if not has_time_scope(message):
            prior_window = re.search(
                r"\b(?:last|past)\s+\d{1,3}\s+(?:days?|weeks?|months?)\b|\b(?:today|yesterday|last week|this week)\b",
                previous_message,
                re.IGNORECASE,
            )
            if prior_window:
                resolved += f" for {prior_window.group(0)}"
    else:
        prior_question = previous_message
        if has_time_scope(message):
            prior_question = re.sub(
                r"\b(?:last|past)\s+\d{1,3}\s+(?:days?|weeks?|months?)\b|\b(?:today|yesterday|last week|this week)\b",
                "", prior_question, flags=re.IGNORECASE,
            )
        resolved = f"{message}. Regarding {prior_question}"
    log_event(
        "daxview_follow_up_resolved",
        request_id=request_id,
        turn_id=turn_id,
        previous_turn_id=str(previous.get("id")),
    )
    return resolved, merged_context


def historical_argument_clarification(error: ValueError, operation_id: str) -> str:
    detail = str(error).lower()
    if "device_id" in detail:
        return (
            "I need the specific device ID before I can request a telemetry time series. "
            "Try: `Show the voltage trend for device ID 380 for the last 24 hours.`"
        )
    if "alarm_id" in detail:
        return "I need the alarm ID before I can look up alarm details. Try: `Show details for alarm ID 123`."
    if "site_id" in detail:
        return "Choose a DaxView site before I access historical data."
    if operation_id == "telemetry_timeseries":
        return "Choose a device ID and time range before I request telemetry time-series data."
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
        elif "NO_DATA" in error:
            parts.append(f"{label} returned no data for the requested scope and time range.")
        elif "UNAVAILABLE" in error:
            parts.append(f"{label} is currently unavailable from DaxView.")
        else:
            parts.append(f"{label} could not be retrieved: {error}")
    return " ".join(parts)


def device_selection_prompt_from_result(mcp_result: dict) -> tuple[str, list[dict]]:
    data = historical_result_data(mcp_result)
    devices = first_list(data, ("devices", "rows", "items", "meters"))
    choices = []
    lines = [
        "Which device should I use for the max-demand check?",
        "Select a device, or reply with its ID (for example, `device ID 380 for the last 7 days`).",
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
        detail = f"{name} (ID {device_id}, {str(status).title()}"
        if device_type:
            detail += f", {device_type}"
        detail += ")"
        if index <= 12:
            lines.append(f"{index}. {detail}")
        description = f"ID {device_id} | {status}"
        if device_type:
            description += f" | {device_type}"
        choices.append({"label": f"{name} (ID {device_id})", "value": str(device_id), "description": description})
    total = data.get("device_count") or data.get("meter_count") or data.get("row_count") or len(devices)
    if total and len(devices) > 12:
        lines.append(f"Showing the first 12 of {len(devices)} returned devices here; {len(choices)} are selectable.")
    if not choices:
        lines.append("No selectable devices were returned. Please provide the device ID manually.")
    return "\n".join(lines), choices


def build_device_choice_response(turn_id: str, message: str, context: dict, request_id: str) -> dict:
    try:
        arguments = build_historical_arguments("site_device_list", context, message)
        plan = request_daxview_data_plan(turn_id, "site_device_list", arguments, request_id)
        authorization_id = plan.get("authorization_id")
        normalized_arguments = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else arguments
        if not authorization_id:
            raise RuntimeError("Daxview did not return a data authorization for site_device_list")
        result = call_authorized_historical_tool("site_device_list", str(authorization_id), normalized_arguments, request_id)
        prompt, choices = device_selection_prompt_from_result(result)
    except Exception as error:
        log_event("daxview_device_choice_failed", request_id=request_id, turn_id=turn_id, error=str(error))
        prompt = (
            "Which device should I use for the max-demand check? "
            "Reply with the device ID, for example: `device ID 380 for the last 7 days`."
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
        "submit_template": f"max demand for device ID {{value}} for {time_scope}",
    }


def run_highest_demand_device_ranking(turn_id: str, message: str, context: dict, request_id: str) -> list[dict]:
    device_args = validate_tool_arguments("site_device_list", {
        "site_id": int(context["site_id"]),
        **({"building_id": int(context["building_id"])} if context.get("building_id") else {}),
        "limit": DEMAND_RANKING_MAX_DEVICES + 1,
    })
    plan = request_daxview_data_plan(turn_id, "site_device_list", device_args, request_id)
    authorization_id = plan.get("authorization_id")
    normalized_args = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else device_args
    if not authorization_id:
        raise RuntimeError("DaxView did not authorize site_device_list for demand ranking")
    device_result = call_authorized_historical_tool("site_device_list", str(authorization_id), normalized_args, request_id)
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
    for device in selectable:
        device_id = device.get("device_id") or device.get("id") or device.get("meter_id")
        try:
            demand_args = build_historical_arguments("demand_peak_summary", {**context, "device_id": int(device_id)}, message)
            per_plan = request_daxview_data_plan(turn_id, "demand_peak_summary", demand_args, request_id)
            per_auth = per_plan.get("authorization_id")
            per_args = per_plan.get("arguments") if isinstance(per_plan.get("arguments"), dict) else demand_args
            if not per_auth:
                continue
            per_result = call_authorized_historical_tool("demand_peak_summary", str(per_auth), per_args, request_id)
            per_data = historical_result_data(per_result)
            summary = per_data.get("summary") if isinstance(per_data.get("summary"), dict) else per_data
            peak = first_value(summary, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw"))
            if peak is None:
                rows = first_list(per_data, ("rows", "items", "results", "devices"))
                for row in rows:
                    if isinstance(row, dict):
                        peak = first_value(row, ("peak_kw", "max_demand_kw", "maximum_demand_kw", "peak_demand_kw", "max_kw"))
                        if peak is not None:
                            break
            if peak is None:
                continue
            ranked.append({
                "device_id": int(device_id),
                "device_name": device.get("device_name") or device.get("name") or f"Device {device_id}",
                "status": device.get("status") or device.get("connection_status"),
                "peak_kw": float(peak),
                "result": per_result,
                "arguments": per_args,
            })
        except Exception as error:
            log_event("demand_device_rank_step_failed", request_id=request_id, device_id=device_id, error=str(error))
    ranked.sort(key=lambda row: row["peak_kw"], reverse=True)
    return [{
        "operation_id": "demand_peak_summary",
        "arguments": {key: context[key] for key in ("site_id", "building_id", "_time_window") if key in context},
        "result": {"structuredContent": {"data": {"ranked_devices": ranked, "device_count": len(selectable)}}},
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
    plan = request_daxview_data_plan(turn_id, "site_energy_summary", source_args, request_id)
    authorization_id = plan.get("authorization_id")
    normalized = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else source_args
    if not authorization_id:
        raise RuntimeError("DaxView did not authorize site_energy_summary for prediction")
    try:
        source = call_authorized_historical_tool("site_energy_summary", str(authorization_id), normalized, request_id)
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
        value = chart_value(row, value_keys)
        if value is not None:
            candidates.append((value, row))
    if not candidates:
        return None
    peak, row = max(candidates, key=lambda item: item[0])
    return {
        "status": "ok",
        "data": {
            "peak_kw": round(float(peak), 4),
            "peak_time": first_value(row, ("timestamp", "time", "bucket", "date", "start_time", "started_at")),
            "source": "manual calculation from telemetry_timeseries",
            "sample_count": len(candidates),
            "calculation": "max(demand telemetry values)",
            "start_time": arguments.get("start_time"),
            "end_time": arguments.get("end_time"),
            "device_id": arguments.get("device_id"),
            "site_id": arguments.get("site_id"),
            "building_id": arguments.get("building_id"),
        },
    }


def try_manual_demand_peak_fallback(turn_id: str, arguments: dict, request_id: str) -> dict:
    telemetry_args = dict(arguments)
    telemetry_args["metric"] = "demand"
    telemetry_args["phase"] = telemetry_args.get("phase") or "all"
    telemetry_args["bucket"] = telemetry_args.get("bucket") or "1h"
    telemetry_args["aggregation"] = telemetry_args.get("aggregation") or "auto"
    telemetry_args["limit"] = int(telemetry_args.get("limit") or 1000)
    plan = request_daxview_data_plan(turn_id, "telemetry_timeseries", telemetry_args, request_id)
    authorization_id = plan.get("authorization_id")
    normalized = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else telemetry_args
    if not authorization_id:
        raise RuntimeError("Daxview did not return a data authorization for telemetry_timeseries")
    telemetry_result = call_authorized_historical_tool("telemetry_timeseries", str(authorization_id), normalized, request_id)
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
    plan = request_daxview_data_plan(turn_id, "device_energy_breakdown", breakdown_args, request_id)
    authorization_id = plan.get("authorization_id")
    normalized = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else breakdown_args
    normalized = dict(normalized)
    if "start" in normalized:
        normalized["start_time"] = normalized.pop("start")
    if "end" in normalized:
        normalized["end_time"] = normalized.pop("end")
    if not authorization_id:
        raise RuntimeError("Daxview did not return a data authorization for device_energy_breakdown")
    result = call_authorized_historical_tool("device_energy_breakdown", str(authorization_id), normalized, request_id)
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
    message, context = resolve_follow_up_message(turn_id, message, context, session_id, request_id)
    clarification = metric_clarification(message)
    if clarification:
        return {"provider": "daxview-question-filter", "reply": clarification, "resolved_plan": resolved_plan_for_result("clarification", [], context)}
    operation_ids, context, planner_plan = choose_historical_operations(message, context, turn_id, session_id, request_id)
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
    if not operation_ids:
        result = langchain_chat_response(message, request_id, session_id)
        result["resolved_plan"] = resolved_plan_for_result("ok", [], context)
        return result
    if "demand_peak_summary" in operation_ids and needs_device_choice(message) and not context.get("device_id"):
        result = build_device_choice_response(turn_id, message, context, request_id)
        result["resolved_plan"] = resolved_plan_for_result("clarification", operation_ids, context)
        return result
    context = dict(context)
    context.setdefault("_time_window", completed_daily_range(message) or requested_historical_range(message))
    if requested_device_id(message):
        context["device_id"] = requested_device_id(message)
    save_resolved_turn_context(turn_id, context)
    if wants_highest_demand_device(message) and not context.get("device_id"):
        ranked_results = run_highest_demand_device_ranking(turn_id, message, context, request_id)
        if ranked_results and ranked_results[0].get("clarification"):
            return {
                "provider": "daxview-question-filter",
                "reply": ranked_results[0]["clarification"],
                "input_type": "select",
                "choices": ranked_results[0].get("choices") or [],
                "submit_template": "max demand for device ID {value} for the same period",
                "resolved_plan": resolved_plan_for_result("clarification", operation_ids, context, ranked_results),
            }
        deterministic_reply = summarize_historical_answers(message, ranked_results, request_id)
        refined_reply = refine_historical_answer_with_model(message, ranked_results, deterministic_reply, request_id)
        charts = build_charts_from_historical_results(ranked_results)
        return {
            "provider": "daxview-historical-mcp",
            "reply": refined_reply,
            "charts": charts,
            "debug_charts": charts,
            "resolved_plan": resolved_plan_for_result("ok", ["demand_peak_summary"], context, ranked_results),
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
            if DAXVIEW_API_ENABLED:
                result = run_daxview_api_operation(operation_id, arguments, request_id)
                normalized_arguments = arguments
            else:
                plan = request_daxview_data_plan(turn_id, operation_id, arguments, request_id)
                authorization_id = plan.get("authorization_id")
                normalized_arguments = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else arguments
                if not authorization_id:
                    raise RuntimeError(f"Daxview did not return a data authorization for {operation_id}")
                result = call_authorized_historical_tool(operation_id, str(authorization_id), normalized_arguments, request_id)
            if AI_MCP_FALLBACKS_ENABLED and operation_id == "demand_peak_summary" and demand_peak_needs_manual_fallback(result):
                try:
                    results.append(try_manual_demand_peak_fallback(turn_id, normalized_arguments, request_id))
                    continue
                except Exception as fallback_error:
                    errors.append({"operation_id": "telemetry_timeseries", "metric": "demand", "error": str(fallback_error)})
            results.append(
                {
                    "operation_id": operation_id,
                    "arguments": normalized_arguments,
                    "result": result,
                }
            )
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
            "provider": "daxview-direct-api" if DAXVIEW_API_ENABLED else "daxview-historical-mcp",
            "reply": readable_historical_errors(errors) or error_detail or (
                "No DaxView API endpoint returned data." if DAXVIEW_API_ENABLED else "No DaxView MCP tool returned data."
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
    charts = build_charts_from_historical_results(results)
    preview_charts = charts if charts else build_charts_from_historical_results(results, preview=True)
    debug_trace_event(
        "ai_final_response_debug",
        request_id=request_id,
        model=CHAT_MODEL if AI_REFINE_MCP_WITH_MODEL else "deterministic",
        answer=refined_reply,
        charts=preview_charts,
    )
    return {
        "provider": "daxview-direct-api" if DAXVIEW_API_ENABLED else "daxview-historical-mcp",
        "reply": refined_reply,
        "charts": charts,
        "debug_charts": preview_charts,
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
                "fields": result.get("fields") or ["site_id", "building_id", "time_range"],
            }
            if result.get("input_type"):
                waiting_event["input_type"] = result["input_type"]
            if result.get("choices"):
                waiting_event["choices"] = result["choices"]
            if result.get("submit_template"):
                waiting_event["submit_template"] = result["submit_template"]
            add_job_event(
                job_id,
                "waiting_for_user",
                waiting_event,
            )
            update_job_status(job_id, "completed")
            add_job_event(job_id, "completed", {"status": "completed"})
            debug_trace_event("ai_outcome_debug", request_id=request_id, status="waiting_for_user", answer=result["reply"], charts=[])
            save_turn_memory(turn_id, result["reply"], result.get("resolved_plan") or {})
            return
        message_event = {"text": result["reply"]}
        if result.get("charts"):
            message_event["charts"] = result["charts"]
        add_job_event(job_id, "message", message_event)
        update_job_status(job_id, "completed")
        add_job_event(job_id, "completed", {"status": "completed"})
        debug_trace_event(
            "ai_outcome_debug", request_id=request_id, status="completed",
            answer=result["reply"], charts=result.get("debug_charts") or result.get("charts") or [],
            charts_enabled=AI_CHARTS_ENABLED,
        )
        save_turn_memory(turn_id, result["reply"], result.get("resolved_plan") or {})
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


def wants_highest_demand_device(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\b(?:highest|max(?:imum)?|peak|top)\s+demand\s+device\b", lowered)
        or re.search(r"\bdevice\s+(?:with|has|having)\s+(?:the\s+)?(?:highest|max(?:imum)?|peak)\s+demand\b", lowered)
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
    return bool(re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", lowered))


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
    return select_historical_operations(message)[:5]


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
    <script src="/debug/assets/chart.umd.js"></script>
    <script src="/debug/assets/lucide.min.js"></script>
    <style>
      :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { margin: 0; background: #f5f7fb; color: #172033; }
      header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 22px; background: #111827; color: white; }
      header h1 { font-size: 18px; margin: 0; }
      header .meta { color: #cbd5e1; font-size: 12px; }
      main { display: grid; grid-template-columns: 380px 1fr; gap: 14px; padding: 14px; }
      .panel { background: white; border: 1px solid #dbe3ef; border-radius: 8px; overflow: hidden; }
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
      .timeline { padding: 12px 14px; max-height: calc(100vh - 150px); overflow: auto; }
      .flow { display: grid; grid-template-columns: repeat(6, minmax(118px, 1fr)); gap: 8px; padding: 12px 14px; border-bottom: 1px solid #e5ebf3; background: #fbfdff; }
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
      .outcome-answer { white-space: pre-wrap; line-height: 1.55; font-size: 13px; max-width: 85ch; }
      .outcome-state { margin-top: 10px; padding: 9px 10px; border-left: 3px solid #94a3b8; background: #f8fafc; font-size: 12px; }
      .outcome-state.fail { border-left-color: #ef4444; background: #fef2f2; color: #991b1b; }
      .inspectors h3 { margin: 0 0 8px; font-size: 13px; }
      .inspectors details { border-top: 1px solid #e5ebf3; padding: 8px 0; }
      .inspectors summary { cursor: pointer; font-size: 12px; font-weight: 700; }
      .answer-block { border-left: 3px solid #0ea5e9; padding: 8px 10px; margin-top: 8px; white-space: pre-wrap; font-size: 12px; line-height: 1.5; }
      .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
      .chart-item { border: 1px solid #dbe3ef; padding: 10px; min-width: 0; max-width: 100%; }
      .chart-item strong { font-size: 12px; }
      .chart-canvas-wrap { position: relative; height: 260px; max-height: 260px; width: 100%; margin-top: 8px; overflow: hidden; }
      .chart-item canvas { display: block; width: 100% !important; height: 260px !important; max-height: 260px !important; }
      .chart-item details { margin-top: 8px; font-size: 12px; }
      .chart-item summary { cursor: pointer; }
      .chart-item table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
      .chart-item th, .chart-item td { padding: 5px 6px; border-bottom: 1px solid #e5ebf3; text-align: left; }
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
        <div class="meta">DaxView -> AI Server -> API or MCP provider -> AI Server -> DaxView</div>
      </div>
      <button id="refresh">Refresh</button>
    </header>
    <nav class="tabs" aria-label="Dashboard views">
      <button type="button" data-view="traces" class="active">Traces</button>
      <button type="button" data-view="predictions">Predictions</button>
    </nav>
    <main id="trace-workspace">
      <section class="panel">
        <h2>Recent Requests</h2>
        <div class="toolbar"><input id="filter" placeholder="Filter request/tool/error..." /></div>
        <div id="requests" class="request-list"><div class="empty">Loading...</div></div>
      </section>
      <section class="panel">
        <h2>Timeline</h2>
        <div id="timeline" class="timeline"><div class="empty">Select a request.</div></div>
      </section>
    </main>
    __PREDICTION_WORKSPACE__
    <script>
      let grouped = [];
      let selected = null;
      let activeView = "traces";

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
        {key: "plan", label: "Data Plan", match: e => /data_plan/i.test(e.event || "")},
        {key: "api", label: "API Call", match: e => /daxview_api/i.test(e.event || "")},
        {key: "mcp", label: "MCP Tool", match: e => /mcp_tool/i.test(e.event || "")},
        {key: "retrieval", label: "Knowledge", match: e => /retrieve|context|embedding/i.test(e.event || "")},
        {key: "model", label: "AI Model", match: e => /ollama|agent|synthesizer|mcp_answer_refine/i.test(e.event || "")},
        {key: "response", label: "Response", match: e => /response|completed|api_to_ui/i.test(e.event || "")},
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
        const payloads = events.filter(e => e.event === "mcp_tool_request_payload_debug" || e.event === "mcp_tool_response_payload_debug" || e.event === "daxview_api_response_debug");
        const answerHtml = answers.map((item, index) => `<div class="answer-block"><strong>${escapeHtml(item.role || "model")} - ${escapeHtml(item.model || "unknown model")}</strong><br>${escapeHtml(item.answer || "")}</div>`).join("");
        const payloadHtml = payloads.map(item => `<details><summary>${escapeHtml(item.event.replace("_debug", ""))} · ${escapeHtml(item.tool || "")}</summary><pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre></details>`).join("");
        if (!answerHtml && !payloadHtml) return "";
        return `<section class="inspectors">
          ${answerHtml ? `<div><h3>Model answers</h3>${answerHtml}</div>` : ""}
          ${payloadHtml ? `<div><h3>Data payloads</h3>${payloadHtml}</div>` : ""}
        </section>`;
      }
      function renderOutcome(events) {
        const outcome = events.find(event => event.event === "ai_outcome_debug");
        const final = events.find(event => event.event === "ai_final_response_debug");
        const failed = events.find(event => event.event === "daxview_job_failed");
        const unavailable = events.find(event => event.event === "energy_prediction_unavailable");
        const fallback = events.find(event => event.event === "mcp_answer_refine_fallback");
        const answer = outcome?.answer || final?.answer || "";
        const charts = outcome?.charts || final?.charts || [];
        const error = outcome?.error || failed?.error || "";
        const unavailableCode = unavailable?.error_code || (/DAXVIEW_UNAVAILABLE/.test(error) ? "DAXVIEW_UNAVAILABLE" : "");
        const chartExpected = events.some(event =>
          (event.operation_ids || []).some(id => ["energy_forecast", "site_energy_summary", "telemetry_timeseries", "alarm_frequency_summary", "demand_peak_summary"].includes(id))
        );
        const chartItems = charts.map(chart => {
          const series = chart.series?.[0] || {};
          const valueRows = (chart.labels || []).map((label, index) =>
            `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(series.data?.[index] ?? "")}</td></tr>`
          ).join("");
          return `<div class="chart-item"><strong>${escapeHtml(chart.title || "Chart")}</strong>
            <div class="chart-canvas-wrap"><canvas data-spec="${escapeHtml(JSON.stringify(chart))}" aria-label="${escapeHtml(chart.title || "Chart preview")}"></canvas></div>
            <details><summary>View chart values</summary><table><thead><tr><th>Period</th><th>${escapeHtml(series.unit || "Value")}</th></tr></thead><tbody>${valueRows}</tbody></table></details>
          </div>`;
        }).join("");
        let state = "";
        if (unavailableCode) state = `DaxView returned ${unavailableCode}. There are no historical values to plot for this request.`;
        else if (error) state = error;
        else if (!outcome && !final) state = "Waiting for the AI Server result.";
        else if (chartExpected && !charts.length && outcome?.charts_enabled === false) state = "Chart generation is disabled in the AI Server configuration.";
        else if (chartExpected && !charts.length) state = "No numeric chart data was available for this answer.";
        return `<section class="outcome"><h3>Outcome preview</h3>
          ${fallback ? `<div class="outcome-state">Model refinement unavailable; showing the calculated fallback.</div>` : ""}
          ${answer ? `<div class="outcome-answer">${escapeHtml(answer)}</div>` : ""}
          ${chartItems ? `<div class="chart-grid">${chartItems}</div>` : ""}
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
        const res = await fetch("/debug/traces");
        if (!res.ok) {
          document.querySelector("#requests").innerHTML = '<div class="empty">Unauthorized or debug dashboard disabled.</div>';
          return;
        }
        const data = await res.json();
        grouped = groupTraces(data.traces || []);
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
      document.querySelector("#refresh").addEventListener("click", () => activeView === "predictions" ? window.PredictionLab.show() : load());
      document.querySelector("#filter").addEventListener("input", renderRequests);
      document.querySelectorAll(".tabs button").forEach(button => button.addEventListener("click", () => {
        activeView = button.dataset.view;
        document.querySelectorAll(".tabs button").forEach(tab => tab.classList.toggle("active", tab === button));
        document.querySelector("#trace-workspace").hidden = activeView === "predictions";
        document.querySelector("#prediction-workspace").hidden = activeView !== "predictions";
        if (activeView === "predictions") {
          window.PredictionLab.show();
          return;
        }
        selected = activeView === "predictions"
          ? grouped.find(group => group.events.some(event =>
              event.event === "energy_prediction_debug" || (event.operation_ids || []).includes("energy_forecast")))?.id || null
          : null;
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
                    "api_client": DAXVIEW_API_ENABLED,
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
        if path == "/debug/ai":
            if not AI_DEBUG_DASHBOARD_ENABLED:
                self._send_html(404, "<h1>Debug dashboard disabled</h1>")
                return
            self._send_html(200, DEBUG_DASHBOARD_HTML.replace("__PREDICTION_WORKSPACE__", (STATIC_DIRECTORY / "prediction_lab.html").read_text(encoding="utf-8")))
            return
        if path.startswith("/debug/assets/"):
            files = {
                "prediction_lab.js": ("prediction_lab.js", "text/javascript"),
                "prediction_lab.css": ("prediction_lab.css", "text/css"),
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
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
            self._send_json(200, {"traces": list(TRACES)})
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
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"AI-Server listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAI-Server stopped")
    finally:
        server.server_close()
