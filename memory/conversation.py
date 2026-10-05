"""Bounded DaxView conversation memory."""

from __future__ import annotations

import json


def truncate_reply(value: object, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_conversation_history(conn, conversation_id: str, current_turn_id: str, identity: dict, limit: int = 6) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_message, assistant_reply, resolved_plan, context, created_at
            FROM daxview_turns
            WHERE conversation_id = %s
              AND id <> %s
              AND deployment_id = %s
              AND company_id = %s
              AND user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (
                conversation_id,
                current_turn_id,
                identity.get("deployment_id"),
                identity.get("company_id"),
                identity.get("user_id"),
                limit,
            ),
        )
        rows = cur.fetchall()
    history = []
    for row in reversed(rows):
        resolved = row.get("resolved_plan") or {}
        if isinstance(resolved, str):
            try:
                resolved = json.loads(resolved)
            except json.JSONDecodeError:
                resolved = {}
        history.append(
            {
                "turn_id": str(row.get("id")),
                "user_message": row.get("user_message") or "",
                "assistant_reply": truncate_reply(row.get("assistant_reply")),
                "resolved_plan": resolved if isinstance(resolved, dict) else {},
                "context": row.get("context") if isinstance(row.get("context"), dict) else {},
            }
        )
    return history

