"""Pure argument helpers used by the DaxView router."""

from __future__ import annotations

from contracts.tool_schemas import strip_tool_arguments, validate_tool_arguments


def whitelisted_tool_arguments(tool: str, arguments: dict) -> dict:
    return strip_tool_arguments(tool, arguments)


def validate_and_strip_tool_arguments(tool: str, arguments: dict) -> dict:
    return validate_tool_arguments(tool, arguments)

