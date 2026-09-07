# AI-Server

All AI Process

Minimal HTTP API for the chatbot stack. It calls Ollama for local LLM replies.

## Run

```powershell
python server.py
```

The API listens on `http://127.0.0.1:8000`.

- `GET /health` checks availability.
- `POST /chat` accepts `{ "message": "..." }` and returns an Ollama-generated reply.

Environment variables:

- `OLLAMA_URL`, default `http://127.0.0.1:11434`
- `OLLAMA_MODEL`, default `llama3.2:1b`

## Docker

Build and run this service by itself:

```powershell
docker build -t chatbot-api:local .
docker run --rm -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 chatbot-api:local
```

For the full UI plus API stack, use `docker compose up --build` from the `Chatbot-UI` repo.
