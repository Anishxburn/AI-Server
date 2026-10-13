"""Public DaxView MCP manifest and strict argument normalization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MAX_RANGE_DAYS = int(os.getenv("AI_HISTORICAL_MAX_RANGE_DAYS", "366"))
MAX_RESULT_ROWS = int(os.getenv("AI_HISTORICAL_MAX_RESULT_ROWS", "1000"))

TOOL_OPERATIONS = {
    "device_data": {
        "list": ({"site_id", "site_name", "building_id", "device_id", "device_type", "meter_model", "manufacturer", "status", "has_data", "search", "limit"}, 100, 500),
        "metrics": ({"site_id", "site_name", "building_id", "device_id", "start_time", "end_time", "timezone", "limit"}, 100, 500),
        "values": ({"site_id", "site_name", "building_id", "device_id", "metric", "metrics", "start_time", "end_time", "timezone", "bucket", "aggregation", "value_mode", "limit"}, 1000, 2000),
        "energy_rows": ({"site_id", "site_name", "building_id", "start_time", "end_time", "timezone", "limit"}, 100, 500),
    },
    "alarm_data": {
        "active": ({"site_id", "site_name", "building_id", "device_id", "severity", "status", "limit"}, 50, 200),
        "frequency": ({"site_id", "site_name", "building_id", "device_id", "severity", "start_time", "end_time", "timezone", "limit"}, 50, 500),
        "detail": ({"site_id", "site_name", "building_id", "device_id", "alarm_event_id", "alarm_id"}, 1, 1),
    },
    "site_summary": {
        "overview": ({"site_id", "site_name", "building_id", "limit"}, 100, 250),
        "energy": ({"site_id", "site_name", "building_id", "start_time", "end_time", "timezone", "bucket", "limit"}, 1000, 1000),
    },
}

PUBLIC_MCP_TOOLS = frozenset(TOOL_OPERATIONS)


def public_tool_for_internal(operation_id: str) -> tuple[str, str]:
    """Translate legacy in-process intent labels to public tool + operation."""
    mapping = {
        "device_energy_ranking": ("device_data", "energy_rows"),
        "device_energy_breakdown": ("device_data", "energy_rows"),
        "telemetry_top_consumers": ("device_data", "energy_rows"),
        "site_energy_summary": ("site_summary", "energy"),
        "energy_forecast": ("site_summary", "energy"),
        "tariff_cost_summary": ("site_summary", "energy"),
        "energy_comparison_summary": ("site_summary", "energy"),
        "site_metadata_summary": ("site_summary", "overview"),
        "report_summary": ("site_summary", "overview"),
        "site_device_list": ("device_data", "list"),
        "meter_status_summary": ("device_data", "list"),
        "telemetry_metric_catalog": ("device_data", "metrics"),
        "data_availability_summary": ("device_data", "metrics"),
        "telemetry_timeseries": ("device_data", "values"),
        "latest_telemetry_snapshot": ("device_data", "values"),
        "power_quality_summary": ("device_data", "values"),
        "demand_peak_summary": ("device_data", "values"),
        "anomaly_detection_summary": ("device_data", "values"),
        "active_alarm_summary": ("alarm_data", "active"),
        "alarm_frequency_summary": ("alarm_data", "frequency"),
        "alarm_detail_lookup": ("alarm_data", "detail"),
    }
    if operation_id not in mapping:
        raise ValueError(f"Unsupported internal EMS intent: {operation_id}")
    return mapping[operation_id]


def _site_scoped(args: dict) -> bool:
    return any(args.get(key) not in (None, "") for key in ("site_id", "site_name", "building_id", "device_id"))


def _validate_range(args: dict, required: bool) -> None:
    start, end = args.get("start_time"), args.get("end_time")
    if bool(start) != bool(end):
        raise ValueError("start_time and end_time must be supplied together")
    if required and not start:
        raise ValueError("start_time and end_time are required for this operation")
    if not start:
        return
    parsed = []
    for label, value in (("start_time", start), ("end_time", end)):
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{label} must be an ISO timestamp with an offset") from error
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(f"{label} must include a UTC offset")
        parsed.append(dt.astimezone(timezone.utc))
    begin, finish = parsed
    now = datetime.now(timezone.utc)
    if finish <= begin:
        raise ValueError("end_time must be after start_time")
    if finish > now + timedelta(minutes=1):
        raise ValueError("end_time must be in the past or present")
    if finish - begin > timedelta(days=MAX_RANGE_DAYS):
        raise ValueError(f"time range cannot exceed {MAX_RANGE_DAYS} days")
    if args.get("timezone"):
        try:
            ZoneInfo(str(args["timezone"]))
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
    elif required:
        raise ValueError("timezone is required for a time range")


def normalize_mcp_arguments(tool_name: str, arguments: dict) -> dict:
    if tool_name not in TOOL_OPERATIONS:
        raise ValueError(f"Unsupported DaxView MCP tool: {tool_name}")
    if not isinstance(arguments, dict):
        raise ValueError("MCP tool arguments must be an object")
    operation = arguments.get("operation")
    if operation not in TOOL_OPERATIONS[tool_name]:
        raise ValueError(f"Unsupported {tool_name} operation: {operation}")
    allowed, default_limit, max_limit = TOOL_OPERATIONS[tool_name][operation]
    unknown = set(arguments) - allowed - {"operation"}
    if unknown:
        raise ValueError(f"Unsupported fields for {tool_name}.{operation}: {', '.join(sorted(unknown))}")
    clean = {key: value for key, value in arguments.items() if key in allowed and value is not None}
    clean["operation"] = operation

    if operation != "detail" and operation not in {"active"} and not _site_scoped(clean):
        raise ValueError("site context is required for this operation")
    if operation == "active" and not _site_scoped(clean):
        raise ValueError("site context is required for active alarms")
    if tool_name == "device_data" and operation == "values":
        if not clean.get("device_id"):
            raise ValueError("device_id is required for device_data.values")
        if bool(clean.get("metric")) == bool(clean.get("metrics")):
            raise ValueError("provide exactly one of metric or metrics")
        metrics = clean.get("metrics")
        if metrics is not None and (not isinstance(metrics, list) or not 1 <= len(metrics) <= 20):
            raise ValueError("metrics must contain between 1 and 20 metric names")
        if clean.get("bucket", "1h") == "raw":
            _validate_range(clean, required=True)
            if datetime.fromisoformat(str(clean["end_time"]).replace("Z", "+00:00")) - datetime.fromisoformat(str(clean["start_time"]).replace("Z", "+00:00")) > timedelta(days=7):
                raise ValueError("raw values requests cannot exceed 7 days")
        else:
            _validate_range(clean, required=True)
        if clean.get("aggregation") == "raw" and clean.get("bucket", "1h") != "raw":
            raise ValueError("raw aggregation requires bucket=raw")
        if clean.get("value_mode") not in (None, "auto", "measured", "counter", "consumption_delta", "maximum_demand"):
            raise ValueError("unsupported value_mode")
    elif tool_name == "device_data" and operation == "metrics":
        _validate_range(clean, required=False)
        if not clean.get("start_time"):
            end = datetime.now(timezone.utc)
            clean["start_time"] = (end - timedelta(hours=24)).isoformat()
            clean["end_time"] = end.isoformat()
            clean.setdefault("timezone", "Asia/Kuala_Lumpur")
    elif operation in {"energy_rows", "frequency", "energy"}:
        _validate_range(clean, required=True)
    elif operation == "detail":
        if bool(clean.get("alarm_event_id")) == bool(clean.get("alarm_id")):
            raise ValueError("provide exactly one of alarm_event_id or alarm_id")
        if not _site_scoped(clean) and not (clean.get("alarm_event_id") or clean.get("alarm_id")):
            raise ValueError("alarm scope or alarm identifier is required")
    elif operation not in {"list", "active", "overview"}:
        _validate_range(clean, required=False)

    if operation == "frequency":
        _validate_range(clean, required=True)
    if operation == "energy":
        clean.setdefault("bucket", "day")
        if clean["bucket"] not in {"hour", "day", "week"}:
            raise ValueError("site_summary.energy bucket must be hour, day, or week")
    if operation == "values":
        clean.setdefault("bucket", "1h")
        clean.setdefault("aggregation", "auto")
        clean.setdefault("value_mode", "auto")
        if clean["bucket"] not in {"raw", "1m", "5m", "15m", "30m", "1h", "1d", "hour", "day", "week"}:
            raise ValueError("unsupported values bucket")
        if clean["aggregation"] not in {"auto", "raw", "mean", "min", "max", "sum", "last"}:
            raise ValueError("unsupported values aggregation")
        if clean["value_mode"] == "maximum_demand" and clean["aggregation"] not in {"max", "auto"}:
            raise ValueError("maximum_demand values require max aggregation")
    limit = clean.get("limit", default_limit)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    clean["limit"] = min(limit, max_limit, MAX_RESULT_ROWS)
    if "timezone" in clean:
        try:
            ZoneInfo(str(clean["timezone"]))
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
    return clean


def build_public_calls(internal_operation: str, arguments: dict, message: str = "") -> list[tuple[str, dict]]:
    """Build one or more public MCP calls from trusted in-process intent data."""
    tool, operation = public_tool_for_internal(internal_operation)
    base = {key: arguments[key] for key in ("site_id", "site_name", "building_id", "device_id", "timezone") if arguments.get(key) is not None}
    start = arguments.get("start_time") or arguments.get("start")
    end = arguments.get("end_time") or arguments.get("end")
    window = {**base, **({"start_time": start, "end_time": end} if start and end else {})}
    if window.get("start_time"):
        window.setdefault("timezone", "Asia/Kuala_Lumpur")
    limit = arguments.get("limit")

    if internal_operation in {"device_energy_ranking", "device_energy_breakdown", "telemetry_top_consumers"}:
        return [(tool, {"operation": operation, **window, "limit": limit or 5})]
    if internal_operation in {"site_energy_summary", "tariff_cost_summary"}:
        result = {"operation": operation, **window, "bucket": arguments.get("bucket") or "day"}
        if limit is not None:
            result["limit"] = limit
        return [(tool, result)]
    if internal_operation == "energy_forecast":
        forecast_start = parse_manifest_datetime(arguments.get("forecast_start")) or datetime.now(timezone.utc)
        training_days = min(max(int(arguments.get("training_days") or 35), 21), 366)
        result = {"operation": "energy", **base,
                  "start_time": (forecast_start - timedelta(days=training_days)).isoformat(),
                  "end_time": forecast_start.isoformat(), "timezone": base.get("timezone", "Asia/Kuala_Lumpur"),
                  "bucket": "day", "limit": min(training_days, 1000)}
        return [("site_summary", result)]
    if internal_operation == "energy_comparison_summary":
        calls = []
        for period in (("period_a_start", "period_a_end"), ("period_b_start", "period_b_end")):
            begin, finish = arguments.get(period[0]), arguments.get(period[1])
            if begin and finish:
                calls.append(("site_summary", {"operation": "energy", **base, "start_time": begin, "end_time": finish,
                                               "timezone": base.get("timezone", "Asia/Kuala_Lumpur"), "bucket": "day"}))
        if len(calls) != 2:
            raise ValueError("two complete comparison periods are required")
        return calls
    if internal_operation in {"site_metadata_summary", "report_summary"}:
        if internal_operation == "report_summary":
            calls = [("site_summary", {"operation": "overview", **base, "limit": 100})]
            calls.extend([
                ("device_data", {"operation": "list", **base, "limit": 100}),
                ("alarm_data", {"operation": "active", **base, "status": "open", "limit": 50}),
            ])
            if start and end:
                calls.extend([
                    ("site_summary", {"operation": "energy", **window, "bucket": "day", "limit": 100}),
                    ("device_data", {"operation": "energy_rows", **window, "limit": 100}),
                    ("alarm_data", {"operation": "frequency", **window, "limit": 10}),
                ])
            return calls
        return [(tool, {"operation": operation, **base, "limit": limit or 100})]
    if internal_operation in {"site_device_list", "meter_status_summary"}:
        result = {"operation": "list", **base, "limit": limit or 100}
        lowered = message.lower()
        if internal_operation == "meter_status_summary":
            for status in ("online", "stale", "offline", "unknown"):
                if status in lowered:
                    result["status"] = status
                    break
            if "have data" in lowered or "with data" in lowered:
                result["has_data"] = True
        return [("device_data", result)]
    if internal_operation in {"telemetry_metric_catalog", "data_availability_summary"}:
        result = {"operation": "metrics", **base, "limit": limit or 100}
        if start and end:
            result.update(start_time=start, end_time=end)
            result.setdefault("timezone", "Asia/Kuala_Lumpur")
        return [("device_data", result)]
    if internal_operation == "latest_telemetry_snapshot":
        end_dt = datetime.now(timezone.utc)
        result = {"operation": "values", **base,
                  "start_time": (end_dt - timedelta(hours=1)).isoformat(), "end_time": end_dt.isoformat(),
                  "timezone": base.get("timezone", "Asia/Kuala_Lumpur"), "bucket": "raw", "aggregation": "last", "limit": limit or 100}
        metrics = arguments.get("metrics")
        if metrics:
            result["metrics"] = metrics[:20]
        else:
            result["metric"] = arguments.get("metric") or "energy"
        return [("device_data", result)]
    if internal_operation in {"active_alarm_summary", "alarm_frequency_summary"}:
        result = {"operation": "active" if internal_operation == "active_alarm_summary" else "frequency", **base,
                  "limit": limit or (50 if internal_operation == "active_alarm_summary" else 10)}
        if operation == "frequency":
            if not start or not end:
                raise ValueError("start_time and end_time are required for alarm frequency")
            result.update(start_time=start, end_time=end)
            result.setdefault("timezone", "Asia/Kuala_Lumpur")
        if internal_operation == "active_alarm_summary":
            result["status"] = "open"
        severity = next((value for value in ("critical", "warning", "info") if value in message.lower()), None)
        if severity:
            result["severity"] = severity
        return [("alarm_data", result)]
    if internal_operation == "alarm_detail_lookup":
        result = {"operation": "detail", **base}
        event_match = re.search(r"\b(?:alarm\s*)?event\s*(?:id|#|:)?\s*(\d+)\b", message, re.IGNORECASE)
        if event_match:
            result["alarm_event_id"] = int(event_match.group(1))
        elif arguments.get("alarm_id"):
            result["alarm_id"] = arguments["alarm_id"]
        return [("alarm_data", result)]
    if internal_operation in {"telemetry_timeseries", "power_quality_summary", "demand_peak_summary", "anomaly_detection_summary"}:
        metrics = arguments.get("metrics")
        metric = arguments.get("metric")
        if internal_operation == "demand_peak_summary":
            metric = "demand"
        elif internal_operation == "power_quality_summary":
            metrics = [item for item in ("voltage", "thd", "power_factor") if item in message.lower()]
            metric = None
        if not metrics and not metric:
            metric = "energy"
        default_bucket = "15m" if internal_operation == "demand_peak_summary" else "1h"
        result = {"operation": "values", **window, "bucket": arguments.get("bucket") or default_bucket,
                  "aggregation": "max" if internal_operation == "demand_peak_summary" else arguments.get("aggregation") or "auto",
                  "value_mode": "maximum_demand" if internal_operation == "demand_peak_summary" else arguments.get("value_mode") or "auto",
                  "limit": min(int(limit or 500), 2000)}
        if metrics:
            result["metrics"] = metrics[:20]
        else:
            result["metric"] = metric
        return [("device_data", result)]
    raise ValueError(f"No public MCP mapping for {internal_operation}")


def parse_manifest_datetime(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
