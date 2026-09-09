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


def ollama_json(path: str, payload: dict, timeout: int = 120) -> dict:
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
    log_event("api_to_ollama_request", request_id=request_id, model=OLLAMA_MODEL, prompt_preview=preview(prompt))
    try:
        data = ollama_json(
            "/api/generate",
            {
                "model": OLLAMA_MODEL,
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
                "model": OLLAMA_MODEL if related else None,
                "is_ems_related": related,
                "sources": sources,
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
