"""Small Ollama helpers shared by planner and answer composer."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


def ollama_json(base_url: str, path: str, payload: dict, timeout: int) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

