"""Verify AI-server MCP data calls against DaxView ground truth.

Runs only when RUN_MCP_VERIFICATION=1. This script does not know DaxView SQL;
the backend team fills tests/mcp_verification/ground_truth.yaml with expected
values and the read-only SQL used to obtain them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "tests" / "mcp_verification" / "ground_truth.yaml"
OUTPUT_DIR = ROOT / "output"


def load_cases() -> str:
    return GROUND_TRUTH.read_text(encoding="utf-8")


def call_ai_debug(payload: dict) -> dict:
    base = os.environ["AI_SERVER_URL"].rstrip("/")
    request = Request(
        f"{base}/debug/mcp-verify",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    if os.getenv("RUN_MCP_VERIFICATION") != "1":
        print("SKIPPED: set RUN_MCP_VERIFICATION=1 to run live DaxView MCP verification.")
        return 0
    if not os.getenv("AI_SERVER_URL"):
        raise SystemExit("AI_SERVER_URL is required, for example https://ai.daxview.com")
    payload = {"ground_truth_yaml": load_cases()}
    result = call_ai_debug(payload)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = OUTPUT_DIR / f"mcp_verification_{stamp}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    rows = result.get("checks") or []
    print("MCP verification")
    print("name | status | detail | request_id")
    print("--- | --- | --- | ---")
    for row in rows:
        print(f"{row.get('name')} | {row.get('status')} | {row.get('detail', '')} | {row.get('request_id', '')}")
    print(f"\nWrote {output_path}")
    return 0 if all(row.get("status") in {"pass", "skip"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

