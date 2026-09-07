# AI-Server

All AI Process

Minimal HTTP API for the chatbot stack.

## Run

```powershell
python server.py
```

The API listens on `http://127.0.0.1:8000`.

- `GET /health` checks availability.
- `POST /chat` accepts `{ "message": "..." }` and returns a placeholder reply.

The placeholder provider is intentionally small so an LLM or data-backed provider can be added later without changing the UI contract.
