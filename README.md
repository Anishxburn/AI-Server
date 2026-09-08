# AI-Server

Minimal HTTP API for the chatbot stack. Routes messages to Ollama for local LLM inference.

## Quick Start

### With Docker (Recommended)

Use the full stack from the `Chatbot-UI` directory:

```powershell
cd ..\Chatbot-UI
docker compose up --build
```

This starts Ollama, this API, and the web UI together. Open http://127.0.0.1:8085.

### Local Development

```powershell
python server.py
```

The API listens on `http://127.0.0.1:8000`.

Then in another terminal, start Ollama:

```powershell
# If you have ollama installed locally
ollama serve
```

Or start it in Docker:

```powershell
docker run -p 11434:11434 ollama/ollama
```

## API Endpoints

### Health Check

```powershell
GET /health
# {"status": "ok", "service": "AI-Server"}
```

### Send a Message

```powershell
POST /chat
# Request: {"message": "What is 2+2?"}
# Response: {"reply": "...", "provider": "ollama", "model": "llama3.2:1b"}
```

Example:

```powershell
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### View Request Traces

```powershell
GET /debug/traces
# Returns last 25 requests with timing and error info
```

Example:

```powershell
curl http://127.0.0.1:8000/debug/traces | jq '.traces[0]'
```

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Model to use for inference |
| `CHATBOT_HOST` | `127.0.0.1` | API host |
| `CHATBOT_PORT` | `8000` | API port |
| `CHATBOT_TRACE_LIMIT` | `25` | Max traces to keep |
| `CHATBOT_ALLOWED_ORIGINS` | `http://localhost:5500,...` | CORS allowed origins |

### Change Model

Set before running:

```powershell
$env:OLLAMA_MODEL = "mistral:7b"
python server.py
```

Or in Docker:

```powershell
$env:OLLAMA_MODEL = "mistral:7b"
docker build -t chatbot-api:local .
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 chatbot-api:local
```

## How It Works

1. Receives a message via `POST /chat`
2. Validates the JSON request
3. Logs the request with a unique ID
4. Calls Ollama's `/api/generate` endpoint
5. Returns the generated reply as JSON
6. Logs the response with timing info

All communication is synchronous (non-streaming) for simplicity.

## Debugging

### Watch Logs

In Docker:

```powershell
docker compose logs -f chatbot-api
```

Locally:

```powershell
python server.py
# Logs appear in stdout
```

### View Traces

```powershell
curl http://127.0.0.1:8000/debug/traces | jq '.'
```

Each trace includes:

- `request_id` — Unique identifier
- `duration_ms` — Total time
- `ui_to_api` — What we received
- `api_to_ollama` — What we sent to Ollama
- `api_to_ui` — What we returned
- `status` — "ok" or "error"

### Test Ollama Directly

```powershell
# Health check
curl http://127.0.0.1:11434/api/tags

# Direct inference
curl -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "What is 2+2?",
    "stream": false
  }'
```

If this works but the chatbot doesn't, the issue is in this API's configuration.

## Docker

### Build Locally

```powershell
docker build -t chatbot-api:local .
```

### Run Standalone

```powershell
# With Ollama on host machine
docker run -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  chatbot-api:local

# With Ollama in Docker
docker run -p 8000:8000 \
  -e OLLAMA_URL=http://ollama:11434 \
  --network=chatbot-ui_default \
  chatbot-api:local
```

### Run with Full Stack

From `Chatbot-UI`:

```powershell
docker compose up --build
```

This automatically:
- Builds and runs this API
- Starts Ollama with model pulling
- Serves the web UI
- Connects everything via Docker network

## Performance

Response time depends on the model:

| Model | Speed | Quality | Hardware |
|-------|-------|---------|----------|
| llama3.2:1b | ~1s | Basic | 2GB RAM |
| llama3.2:3b | ~5s | Good | 4GB RAM |
| llama3.2:7b | ~15s | Excellent | 8GB RAM |
| mistral:7b | ~20s | Excellent | 8GB RAM |

See the Chatbot-UI documentation for full model list and tuning options.

## Extending

### Add a System Prompt

Edit `server.py`, around line 42:

```python
def ask_ollama(message: str, request_id: str) -> str:
    system_prompt = "You are a helpful assistant."
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\nUser: {message}",
        # ...
    }
```

### Enable Streaming

Modify `ask_ollama()` to use `"stream": True` and parse chunks.

### Add Multi-Model Support

Add a model parameter to `/chat`:

```python
def do_POST(self) -> None:
    # ...
    request = json.loads(self.rfile.read(content_length))
    message = str(request.get("message", "")).strip()
    model = str(request.get("model", OLLAMA_MODEL)).strip()
```

### Persistent Chat History

Store messages in a database (SQLite, PostgreSQL, etc.) and return context to Ollama.

## More Information

- **Full Stack Guide:** See [../Chatbot-UI/README.md](../Chatbot-UI/README.md)
- **Model Selection:** See [../Chatbot-UI/OLLAMA_MODELS.md](../Chatbot-UI/OLLAMA_MODELS.md)
- **Debugging:** See [../Chatbot-UI/DEBUGGING.md](../Chatbot-UI/DEBUGGING.md)
- **Ollama Docs:** https://ollama.ai/docs
- **Python HTTP:** https://docs.python.org/3/library/http.server.html

## Next Steps

1. ✅ Start the full stack from `Chatbot-UI`
2. 🔧 Customize system prompt in `server.py`
3. 📊 Add chat history/persistent storage
4. 🚀 Deploy to cloud with GPU acceleration
