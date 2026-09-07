"""Minimal chatbot API backed by Ollama when available."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
import uuid
from collections import deque
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse


HOST = os.getenv("CHATBOT_HOST", "127.0.0.1")
PORT = int(os.getenv("CHATBOT_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
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


def log_event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def ask_ollama(message: str, request_id: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": message,
        "stream": False,
        "options": {
            "temperature": 0.7,
        },
    }
    request = Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    log_event(
        "api_to_ollama_request",
        request_id=request_id,
        ollama_url=f"{OLLAMA_URL}/api/generate",
        model=OLLAMA_MODEL,
        prompt_preview=preview(message),
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        log_event(
            "api_to_ollama_error",
            request_id=request_id,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
            error=str(error),
        )
        raise RuntimeError(f"Ollama is not ready: {error}") from error

    reply = str(data.get("response", "")).strip()
    if not reply:
        raise RuntimeError("Ollama returned an empty response")

    log_event(
        "api_from_ollama_response",
        request_id=request_id,
        duration_ms=round((time.perf_counter() - started_at) * 1000),
        model=OLLAMA_MODEL,
        reply_preview=preview(reply),
    )
    return reply


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
            self._send_json(200, {"status": "ok", "service": "AI-Server"})
            return
        if path == "/debug/traces":
            self._send_json(200, {"traces": list(TRACES)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/chat":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length))
            message = str(request.get("message", "")).strip()
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if not message:
            self._send_json(400, {"error": "message is required"})
            return

        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        trace = {
            "request_id": request_id,
            "client": self.client_address[0],
            "request_path": "/chat",
            "ui_to_api": {
                "method": "POST",
                "body": {"message": message},
            },
            "api_to_ollama": {
                "url": f"{OLLAMA_URL}/api/generate",
                "model": OLLAMA_MODEL,
                "stream": False,
            },
        }
        log_event(
            "ui_to_api_request",
            request_id=request_id,
            client=self.client_address[0],
            message_preview=preview(message),
        )

        try:
            reply = ask_ollama(message, request_id)
        except RuntimeError as error:
            trace["status"] = "error"
            trace["error"] = str(error)
            trace["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
            TRACES.appendleft(trace)
            log_event("api_to_ui_error", request_id=request_id, error=str(error))
            self._send_json(503, {"error": str(error), "provider": "ollama"})
            return

        trace["status"] = "ok"
        trace["api_to_ui"] = {
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "reply_preview": preview(reply),
        }
        trace["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        TRACES.appendleft(trace)
        log_event(
            "api_to_ui_response",
            request_id=request_id,
            duration_ms=trace["duration_ms"],
            provider="ollama",
            model=OLLAMA_MODEL,
            reply_preview=preview(reply),
        )
        self._send_json(200, {"reply": reply, "provider": "ollama", "model": OLLAMA_MODEL})

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
