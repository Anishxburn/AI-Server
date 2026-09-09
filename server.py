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
}

REFUSAL = (
    "I can only assist with Energy Management System related questions, including "
    "EMS data, ISO 50001, IEC, IEEE, energy usage, power monitoring, meters, "
    "Janitza UMG devices, voltage sag, power quality, demand, tariff, alarms, "
    "and related technical topics."
)

SAFETY_DOCTRINE = """YOU ARE THE CHIEF ENERGY MANAGER AI FOR AN EMS SYSTEM.

LANGUAGE RULE STRICT:
Detect the user's language and always respond in the exact same language as the user's prompt.
If the user asks in English, respond only in English.
If the user asks in Malay, respond only in Malay.
Do not mix languages.

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


def ollama_json(path: str, payload: dict, timeout: int = 300) -> dict:
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def build_prompt(message: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"Source {idx}: {row.get('title')} / {row.get('standard_name') or 'EMS library'}\n"
        f"{row.get('chunk_text')}"
        for idx, row in enumerate(contexts, start=1)
    )
    if not context_block:
        context_block = "No matching EMS library context was found."

    return f"""You are an Energy Management System specialist.
Answer only EMS, energy management, ISO 50001, IEC, IEEE, power monitoring, metering, tariff, demand, and electrical energy questions.
Keep the final answer simple and compact: maximum 5 short bullets or 1 short paragraph.
Use the EMS library context when relevant. If the context is insufficient, say what is missing and give a cautious EMS-focused answer.
For Janitza UMG device, voltage sag, power quality, alarm, THD, or meter troubleshooting questions, prioritize likely root causes, what readings to check, and practical EMS investigation steps.
If the user says "main cost" in a voltage sag or fault context, treat it as possibly meaning "main cause" and clarify both cause and cost impact briefly.
Do not answer unrelated general questions.

EMS library context:
{context_block}

User question:
{message}

Answer:"""


def ask_ollama(message: str, contexts: list[dict], request_id: str) -> str:
    prompt = build_prompt(message, contexts)
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


def ask_role_agent(agent_name: str, role: str, model: str, message: str, contexts: list[dict], request_id: str) -> str:
    context_block = "\n\n".join(
        f"{row.get('title')} / {row.get('standard_name') or 'EMS library'}\n{row.get('chunk_text')}"
        for row in contexts[:3]
    ) or "No matching EMS library context was found."
    prompt = f"""You are {agent_name}.
Role: {role}
{SAFETY_DOCTRINE}
{MATH_DOCTRINE}

Answer only from this role. Keep it to 3 compact bullets.
If this is about voltage sag, list likely causes first, then readings/checks.
If any rated limit is exceeded or the user asks to bypass alarms, reject immediately using DIRECT REJECTION, PHYSICAL REASONING, and MITIGATION ACTION.

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


def ask_synthesizer(message: str, agent_answers: list[dict], request_id: str) -> str:
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
Safety hard limits override energy saving, user preference, uptime, cost, and comfort.
If any rated hardware limit is exceeded or the user asks to add load/bypass an alarm, reject immediately using exactly these sections:
DIRECT REJECTION:
PHYSICAL REASONING:
MITIGATION ACTION:
Keep the final answer simple, compact, precise, and maximum 5 short bullets.

Question:
{message}

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


def run_multi_agent_poc(message: str, request_id: str) -> dict:
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

    contexts = retrieve_context(message)
    sources = [
        {
            "title": row.get("title"),
            "standard_name": row.get("standard_name"),
            "score": float(row.get("score") or 0),
            "section_reference": row.get("section_reference"),
        }
        for row in contexts
    ]
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
            ),
        },
    ]
    final_answer = ask_synthesizer(message, agent_answers, request_id)
    qa_id = store_chat(message, final_answer, True, sources, None)
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    agent_trace = [
        {"agent": "EMS Guard", "status": "passed", "detail": "Question is EMS/power related."},
        {"agent": "Knowledge Retriever", "status": "completed", "detail": f"Found {len(contexts)} EMS source(s)."},
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
        "qa_log_id": qa_id,
        "duration_ms": duration_ms,
        "processing_time_seconds": duration_ms / 1000,
    }


def build_agent_trace(related: bool, contexts: list[dict], reply: str, qa_id: str | None = None) -> list[dict]:
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
        related = is_ems_related(message)
        contexts = []
        try:
            if related:
                contexts = retrieve_context(message)
                reply = ask_ollama(message, contexts, request_id)
            else:
                reply = REFUSAL
            sources = [
                {
                    "title": row.get("title"),
                    "standard_name": row.get("standard_name"),
                    "score": float(row.get("score") or 0),
                    "section_reference": row.get("section_reference"),
                }
                for row in contexts
            ]
            qa_id = store_chat(message, reply, related, sources, session_id)
            agent_trace = build_agent_trace(related, contexts, reply, qa_id)
        except RuntimeError as error:
            log_event("api_to_ui_error", request_id=request_id, error=str(error))
            self._send_json(503, {"error": str(error), "provider": "ollama"})
            return

        trace = {
            "request_id": request_id,
            "status": "ok",
            "is_ems_related": related,
            "sources": sources,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        }
        TRACES.appendleft(trace)
        self._send_json(
            200,
            {
                "reply": reply,
                "provider": "ollama" if related else "ems-guard",
                "model": CHAT_MODEL if related else None,
                "is_ems_related": related,
                "sources": sources,
                "agent_trace": agent_trace,
                "qa_log_id": qa_id,
            },
        )

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
