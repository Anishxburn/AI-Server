"""Measurement definitions shared by routing, calculations, and answer generation."""

import math
import re

METRICS = {
    "energy": {"unit": "kWh", "meaning": "interval energy consumption", "aggregation": "sum"},
    "demand": {"unit": "kW", "meaning": "demand over a defined measurement interval", "aggregation": "max"},
    "current": {"unit": "A", "meaning": "electrical current with phase/channel", "aggregation": "average"},
    "voltage": {"unit": "V", "meaning": "voltage with phase/channel", "aggregation": "average"},
    "power_factor": {"unit": "ratio", "meaning": "power factor", "aggregation": "average"},
    "frequency": {"unit": "Hz", "meaning": "electrical frequency", "aggregation": "average"},
    "thd": {"unit": "%", "meaning": "total harmonic distortion; identify voltage or current", "aggregation": "average"},
}


def question_metrics(message):
    text = message.lower()
    matches = []
    patterns = {
        "demand": r"\b(?:demand|kw)\b", "energy": r"\b(?:energy|kwh|mwh|consumption|usage)\b",
        "voltage": r"\b(?:voltage|volts?|sag|swell)\b", "current": r"\b(?:amperes?|amps?|electrical current|phase current|current trend|current reading)\b",
        "power_factor": r"\bpower factor\b", "frequency": r"\bfrequency\b", "thd": r"\b(?:thd|harmonics?)\b",
    }
    for metric, pattern in patterns.items():
        if re.search(pattern, text):
            matches.append(metric)
    return matches


def metric_clarification(message):
    text = message.lower()
    if re.search(r"\bcurrent\b", text) and not question_metrics(message):
        if not re.search(r"\bcurrent\s+(?:site|device|alarm|alarms|status|meter)\b", text):
            return "Do you mean electrical current in amperes, or the latest available reading? For electrical current, specify the device and phase when available."
    return None


def percentage_difference(value, reference):
    if isinstance(value, bool) or isinstance(reference, bool):
        return None
    try:
        value, reference = float(value), float(reference)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(value) or not math.isfinite(reference) or reference == 0:
        return None
    return (value - reference) / reference * 100


def result_problem(result):
    if not isinstance(result, dict):
        return "The MCP response was not structured data."
    if result.get("isError") or result.get("status") == "error":
        return str(result.get("error_code") or "MCP_TOOL_ERROR")
    for key in ("structuredContent", "data", "backend_response", "result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            issue = result_problem(nested)
            if issue:
                return issue
    return None


def measurement_context(message):
    return [{"metric": metric, **METRICS[metric]} for metric in question_metrics(message)]
