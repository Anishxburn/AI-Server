"""Minimal chatbot API backed by Ollama when available."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse


HOST = os.getenv("CHATBOT_HOST", "127.0.0.1")
PORT = int(os.getenv("CHATBOT_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv(
        "CHATBOT_ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
}


def ask_ollama(message: str) -> str:
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

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama is not ready: {error}") from error

    reply = str(data.get("response", "")).strip()
    if not reply:
        raise RuntimeError("Ollama returned an empty response")
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
        if urlparse(self.path).path == "/health":
            self._send_json(200, {"status": "ok", "service": "AI-Server"})
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

        try:
            reply = ask_ollama(message)
        except RuntimeError as error:
            self._send_json(503, {"error": str(error), "provider": "ollama"})
            return

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
