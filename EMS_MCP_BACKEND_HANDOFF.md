# EMS Multi-Agent POC Backend Handoff

## Running URLs

- Normal chatbot UI: `http://127.0.0.1:8085/`
- Multi-agent POC webpage: `http://127.0.0.1:8091/`
- Multi-agent POC API: `POST http://127.0.0.1:8091/multi-agent-chat`
- Health check: `GET http://127.0.0.1:8091/health`

## Docker Services

Run from `C:\chatbot-stack\Chatbot-UI`:

```powershell
docker compose up --build -d
```

Services:

- `ollama`: local model runtime.
- `ollama-init`: pulls required models.
- `postgres`: PostgreSQL + pgvector.
- `chatbot-api`: backend API for `8085`.
- `agent-poc-api`: backend/webpage/API for `8091`.
- `chatbot-ui`: nginx frontend for `8085`.

## Models To Pull

These are pulled by `ollama-init`:

```powershell
ollama pull qwen2.5:3b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

Current model usage:

- `OLLAMA_MODEL=qwen2.5:3b`
- `DEEPSEEK_MODEL=deepseek-r1:7b`
- `CHAT_MODEL=deepseek-r1:7b`
- `EMBEDDING_MODEL=nomic-embed-text`

In the `8091` POC:

- Agent 1: `qwen2.5:3b`
- Agent 2: `qwen2.5:3b`
- Final decision maker: `deepseek-r1:7b`
- Embeddings: `nomic-embed-text`

## Multi-Agent API

Endpoint:

```http
POST http://127.0.0.1:8091/multi-agent-chat
Content-Type: application/json
```

Request:

```json
{
  "message": "My Janitza UMG 509 shows voltage sag and one feeder is near rated current. What are the likely causes and safe EMS action?"
}
```

Response shape:

```json
{
  "reply": "Final compact answer from DeepSeek",
  "provider": "multi-agent-poc",
  "model": "deepseek-r1:7b",
  "decision_model": "deepseek-r1:7b",
  "is_ems_related": true,
  "sources": [
    {
      "title": "Janitza UMG Power Quality Knowledge",
      "standard_name": "Janitza UMG / EMS Power Quality",
      "score": 0.53,
      "section_reference": null
    }
  ],
  "agent_discussion": [
    {
      "agent": "Agent 1 - Ollama EMS Triage",
      "role": "Quickly classify the issue and identify immediate EMS checks.",
      "model": "qwen2.5:3b",
      "answer": "Agent 1 answer"
    },
    {
      "agent": "Agent 2 - Qwen Power Quality",
      "role": "Find likely electrical root causes and UMG meter readings to inspect.",
      "model": "qwen2.5:3b",
      "answer": "Agent 2 answer"
    }
  ],
  "qa_log_id": "uuid"
}
```

## Current System Flow

```text
8091 webpage
  -> POST /multi-agent-chat
  -> EMS keyword guard
  -> pgvector document retrieval
  -> Agent 1 with qwen2.5:3b
  -> Agent 2 with qwen2.5:3b
  -> DeepSeek final decision maker
  -> Save Q&A into qa_logs/chat_messages
  -> Save question embedding into qa_embeddings
  -> Return final answer, agent discussion, and sources
```

## EMS Safety Prompt Rules

The `8091` POC uses a strict Chief Energy Manager safety doctrine:

- Match the user's language exactly.
- Reject adding load or overriding alarms when hardware limits are exceeded.
- Never suggest tolerant language when hardware is above 100% rated capacity.
- For critical limit breach, final answer must use:
  - `DIRECT REJECTION:`
  - `PHYSICAL REASONING:`
  - `MITIGATION ACTION:`

## Database

pgAdmin connection:

```text
Host: 127.0.0.1
Port: 15432
Database: chatbot
Username: chatbot
Password: chatbot
```

Tables:

- `documents`: uploaded/seeded knowledge documents.
- `document_chunks`: vector chunks for RAG library search.
- `chat_sessions`: chat sessions.
- `chat_messages`: user/assistant message history.
- `qa_logs`: final Q&A logs with source metadata.
- `qa_embeddings`: vector embeddings of EMS-related user questions.
- `feedback`: future rating/comment table.

Important: `document_chunks` is the vector DB for EMS knowledge. `qa_embeddings` is the vector DB for previous EMS questions/answers.

## Knowledge Ingestion API

Endpoint:

```http
POST http://127.0.0.1:8091/knowledge
Content-Type: application/json
```

Request:

```json
{
  "title": "UMG 509 Voltage Sag Notes",
  "standard_name": "Janitza UMG / Power Quality",
  "source_type": "manual",
  "text": "Voltage sag may be caused by motor starting, overloaded feeder, loose terminal, phase imbalance, transformer energization, utility disturbance, short circuit, or high inrush current."
}
```

What happens:

```text
text -> chunk_text()
chunk -> nomic-embed-text
documents row inserted
document_chunks rows inserted with vector(768)
```

## MCP Integration Point

The MCP server should be called after EMS guard and before role-agent prompting.

Recommended flow:

```text
User question
  -> EMS guard
  -> Retrieve document context from pgvector
  -> Call MCP server for live EMS/device data
  -> Agent 1 receives document context + MCP data
  -> Agent 2 receives document context + MCP data
  -> DeepSeek final decision maker receives summaries + MCP data
  -> Save final Q&A and embeddings
```

Recommended MCP data contract:

```json
{
  "device_id": "UMG-509-MSB-01",
  "device_model": "Janitza UMG 509",
  "timestamp": "2026-09-09T14:00:00+08:00",
  "measurements": {
    "voltage_l1": 230.1,
    "voltage_l2": 184.2,
    "voltage_l3": 229.8,
    "current_l1": 180.0,
    "current_l2": 210.0,
    "current_l3": 175.0,
    "rated_current": 200.0,
    "frequency": 50.0,
    "thd_v": 4.2,
    "power_factor": 0.86
  },
  "alarms": [
    {
      "type": "voltage_sag",
      "phase": "L2",
      "severity": "critical",
      "duration_ms": 420
    }
  ]
}
```

Critical safety logic should compare live MCP values against rated limits before allowing any energy optimization recommendation.

## Useful SQL

Recent final Q&A:

```sql
SELECT question, answer, is_ems_related, created_at
FROM qa_logs
ORDER BY created_at DESC
LIMIT 10;
```

Recent vectorized questions:

```sql
SELECT question, left(answer, 160) AS answer_preview, created_at
FROM qa_embeddings
ORDER BY created_at DESC
LIMIT 10;
```

Knowledge sources:

```sql
SELECT d.title, d.standard_name, left(dc.chunk_text, 160) AS chunk_preview
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id
ORDER BY dc.created_at DESC
LIMIT 10;
```
