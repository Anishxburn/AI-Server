"""Compatibility wrapper for the existing deterministic regex router."""

from __future__ import annotations


def select_operations(message: str, keywords: dict[str, set[str]]) -> list[str]:
    lowered = message.lower()
    return [tool for tool, phrases in keywords.items() if any(phrase in lowered for phrase in phrases)]

