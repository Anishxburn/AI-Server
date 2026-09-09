from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "EMS_MCP_Backend_Handoff.pdf"


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 10 * mm, "EMS Multi-Agent POC Backend Handoff")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def code_block(text, styles):
    return Preformatted(text.strip(), styles["CodeBlock"])


def bullets(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["Body"]), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
    )


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleMain",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#17202a"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=9.2,
            backColor=colors.HexColor("#f1f5f9"),
            borderColor=colors.HexColor("#d6deea"),
            borderWidth=0.5,
            borderPadding=5,
            leftIndent=0,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=8,
        )
    )

    story = [
        Paragraph("EMS Multi-Agent POC Backend Handoff", styles["TitleMain"]),
        Paragraph(
            "A concise backend handoff for the EMS chatbot, 8091 multi-agent proof of concept, pgvector RAG storage, and MCP integration point.",
            styles["Body"],
        ),
        Paragraph("Running URLs", styles["Section"]),
        bullets(
            [
                "Normal chatbot UI: http://127.0.0.1:8085/",
                "Multi-agent POC webpage: http://127.0.0.1:8091/",
                "Multi-agent POC API: POST http://127.0.0.1:8091/multi-agent-chat",
                "Health check: GET http://127.0.0.1:8091/health",
            ],
            styles,
        ),
        Paragraph("Docker Services", styles["Section"]),
        code_block("cd C:\\chatbot-stack\\Chatbot-UI\ndocker compose up --build -d", styles),
        bullets(
            [
                "ollama: local model runtime.",
                "ollama-init: pulls required models.",
                "postgres: PostgreSQL + pgvector.",
                "chatbot-api: backend API for 8085.",
                "agent-poc-api: backend/webpage/API for 8091.",
                "chatbot-ui: nginx frontend for 8085.",
            ],
            styles,
        ),
        Paragraph("Models To Pull", styles["Section"]),
        code_block(
            "ollama pull qwen2.5:3b\nollama pull deepseek-r1:7b\nollama pull nomic-embed-text",
            styles,
        ),
        bullets(
            [
                "OLLAMA_MODEL=qwen2.5:3b",
                "DEEPSEEK_MODEL=deepseek-r1:7b",
                "CHAT_MODEL=deepseek-r1:7b",
                "EMBEDDING_MODEL=nomic-embed-text",
                "8091 Agent 1: qwen2.5:3b",
                "8091 Agent 2: qwen2.5:3b",
                "8091 final decision maker: deepseek-r1:7b",
            ],
            styles,
        ),
        Paragraph("Multi-Agent API", styles["Section"]),
        code_block(
            """POST http://127.0.0.1:8091/multi-agent-chat
Content-Type: application/json

{
  "message": "My Janitza UMG 509 shows voltage sag and one feeder is near rated current. What are the likely causes and safe EMS action?"
}""",
            styles,
        ),
        Paragraph("Response Shape", styles["Section"]),
        code_block(
            """{
  "reply": "Final compact answer from DeepSeek",
  "provider": "multi-agent-poc",
  "model": "deepseek-r1:7b",
  "decision_model": "deepseek-r1:7b",
  "is_ems_related": true,
  "sources": [
    {
      "title": "Janitza UMG Power Quality Knowledge",
      "standard_name": "Janitza UMG / EMS Power Quality",
      "score": 0.53
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
}""",
            styles,
        ),
        PageBreak(),
        Paragraph("Current System Flow", styles["Section"]),
        code_block(
            """8091 webpage
  -> POST /multi-agent-chat
  -> EMS keyword guard
  -> pgvector document retrieval
  -> Agent 1 with qwen2.5:3b
  -> Agent 2 with qwen2.5:3b
  -> DeepSeek final decision maker
  -> Save Q&A into qa_logs/chat_messages
  -> Save question embedding into qa_embeddings
  -> Return final answer, agent discussion, and sources""",
            styles,
        ),
        Paragraph("EMS Safety Prompt Rules", styles["Section"]),
        bullets(
            [
                "Match the user's language exactly. English in, English out. Malay in, Malay out.",
                "Reject adding load or overriding alarms when hardware limits are exceeded.",
                "Never suggest tolerant language when hardware is above 100% rated capacity.",
                "For critical limit breach, use DIRECT REJECTION, PHYSICAL REASONING, and MITIGATION ACTION.",
            ],
            styles,
        ),
        Paragraph("Database / pgAdmin", styles["Section"]),
        code_block(
            """Host: 127.0.0.1
Port: 15432
Database: chatbot
Username: chatbot
Password: chatbot""",
            styles,
        ),
        bullets(
            [
                "documents: uploaded/seeded knowledge documents.",
                "document_chunks: vector chunks for RAG library search.",
                "chat_sessions: chat sessions.",
                "chat_messages: user/assistant message history.",
                "qa_logs: final Q&A logs with source metadata.",
                "qa_embeddings: vector embeddings of EMS-related user questions.",
                "feedback: future rating/comment table.",
            ],
            styles,
        ),
        Paragraph("Knowledge Ingestion API", styles["Section"]),
        code_block(
            """POST http://127.0.0.1:8091/knowledge
Content-Type: application/json

{
  "title": "UMG 509 Voltage Sag Notes",
  "standard_name": "Janitza UMG / Power Quality",
  "source_type": "manual",
  "text": "Voltage sag may be caused by motor starting, overloaded feeder, loose terminal, phase imbalance, transformer energization, utility disturbance, short circuit, or high inrush current."
}""",
            styles,
        ),
        Paragraph("MCP Integration Point", styles["Section"]),
        Paragraph(
            "Call the MCP server after EMS guard and pgvector retrieval, before the role-agent prompts. The MCP data should be passed into Agent 1, Agent 2, and the DeepSeek final decision maker.",
            styles["Body"],
        ),
        code_block(
            """User question
  -> EMS guard
  -> Retrieve document context from pgvector
  -> Call MCP server for live EMS/device data
  -> Agent 1 receives document context + MCP data
  -> Agent 2 receives document context + MCP data
  -> DeepSeek final decision maker receives summaries + MCP data
  -> Save final Q&A and embeddings""",
            styles,
        ),
        Paragraph("Recommended MCP Data Contract", styles["Section"]),
        code_block(
            """{
  "device_id": "UMG-509-MSB-01",
  "device_model": "Janitza UMG 509",
  "timestamp": "2026-09-09T14:00:00+08:00",
  "measurements": {
    "voltage_l1": 230.1,
    "voltage_l2": 184.2,
    "current_l2": 210.0,
    "rated_current": 200.0,
    "frequency": 50.0,
    "thd_v": 4.2,
    "power_factor": 0.86
  },
  "alarms": [
    {"type": "voltage_sag", "phase": "L2", "severity": "critical"}
  ]
}""",
            styles,
        ),
        Paragraph("Useful SQL", styles["Section"]),
        code_block(
            """SELECT question, answer, is_ems_related, created_at
FROM qa_logs
ORDER BY created_at DESC
LIMIT 10;

SELECT question, left(answer, 160) AS answer_preview, created_at
FROM qa_embeddings
ORDER BY created_at DESC
LIMIT 10;

SELECT d.title, d.standard_name, left(dc.chunk_text, 160) AS chunk_preview
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id
ORDER BY dc.created_at DESC
LIMIT 10;""",
            styles,
        ),
    ]

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
