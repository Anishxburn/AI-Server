"""Optional LLM planner with deterministic fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal

from contracts.tool_schemas import TOOL_SCHEMAS, validate_tool_arguments

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    ConfigDict = None  # type: ignore
    Field = lambda default=None, **_: default  # type: ignore
    ValidationError = ValueError  # type: ignore


class PlannerSlots(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    device_refs: list[str] = Field(default_factory=list)
    building_ref: str | None = None
    time_phrase: str | None = None
    comparison_time_phrase: str | None = None
    metric: str | None = None
    limit: int | None = None
    demand_limit_kw: float | None = None
    alarm_id: int | None = None


class PlannerClarification(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    field: str | None = None
    question: str | None = None


class PlannerOutput(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    intent: Literal["data_query", "knowledge_question", "follow_up", "out_of_scope"]
    is_follow_up: bool = False
    tools: list[str] = Field(default_factory=list)
    slots: PlannerSlots = Field(default_factory=PlannerSlots)
    inherit_from_previous: list[Literal["device", "building", "time", "metric", "tools"]] = Field(default_factory=list)
    clarification: PlannerClarification | None = None
    confidence: float = 0.0


def planner_json_schema() -> dict[str, Any]:
    if hasattr(PlannerOutput, "model_json_schema"):
        return PlannerOutput.model_json_schema()
    return PlannerOutput.schema()


def planner_system_prompt(tool_descriptions: dict[str, str]) -> str:
    tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in sorted(tool_descriptions.items()))
    return f"""You are a strict EMS DaxView planning assistant.
Return only JSON matching the schema. You may choose only these tools:
{tool_lines}

Rules:
- Never output site_id, company_id, deployment_id, authorization_id, start/end timestamps, or tenant identifiers.
- Use natural references from the user in slots, such as device_refs or building_ref.
- Use tools only from the allowlist.
- There are exactly three public tools. Choose a tool whenever the user requests live/site-specific DaxView facts; do not treat short or informal wording as a knowledge question.
- device_data: list/find devices, discover configured metrics, read historical telemetry (voltage/current/power/demand), and rank device energy. Examples: "list 5 devices", "show voltage on device 519", "max demand for this meter", "top energy-consuming devices".
- alarm_data: active alarms, alarm counts/frequency over a period, latest/recent alarm occurrences, or one alarm/event's details. Latest/frequency intent: "list 5 latest alarms" or "which alarms occurred most often". Active intent: "current/open alarms". Detail intent: an alarm/event ID or a question about one identified alarm.
- site_summary: site/building overview and site-wide energy totals/buckets. Examples: "site energy this week" and "give me this site's overview". Do not use it for per-device readings.
- For multi-intent questions include every relevant public tool, e.g. device_data and alarm_data for device rankings plus active alarms.
- Do not ask for clarification merely because a request is brief. Ask only when a required target cannot be resolved from authorized context or a specific device/alarm is genuinely ambiguous.
- For ordinary EMS definitions or calculations using values already supplied by the user, choose no DaxView tool.

Examples:
User: max demand yesterday for device 12
JSON: {{"intent":"data_query","is_follow_up":false,"tools":["device_data"],"slots":{{"device_refs":["12"],"time_phrase":"yesterday","metric":"demand"}},"inherit_from_previous":[],"clarification":null,"confidence":0.93}}
User: what about today?
JSON: {{"intent":"follow_up","is_follow_up":true,"tools":[],"slots":{{"time_phrase":"today"}},"inherit_from_previous":["device","tools","metric"],"clarification":null,"confidence":0.84}}
User: top consumers last week and active alarms
JSON: {{"intent":"data_query","is_follow_up":false,"tools":["device_data","alarm_data"],"slots":{{"time_phrase":"last week","metric":"energy","limit":5}},"inherit_from_previous":[],"clarification":null,"confidence":0.9}}
User: List out 5 latest alarm
JSON: {{"intent":"data_query","is_follow_up":false,"tools":["alarm_data"],"slots":{{"limit":5}},"inherit_from_previous":[],"clarification":null,"confidence":0.96}}
User: show current alarms and list devices that are offline
JSON: {{"intent":"data_query","is_follow_up":false,"tools":["alarm_data","device_data"],"slots":{{}},"inherit_from_previous":[],"clarification":null,"confidence":0.94}}
User: what is voltage
JSON: {{"intent":"knowledge_question","is_follow_up":false,"tools":[],"slots":{{}},"inherit_from_previous":[],"clarification":null,"confidence":0.9}}
User: berapa max demand hari ini
JSON: {{"intent":"data_query","is_follow_up":false,"tools":["device_data"],"slots":{{"time_phrase":"today","metric":"demand"}},"inherit_from_previous":[],"clarification":null,"confidence":0.82}}
User: show building 3
JSON: {{"intent":"follow_up","is_follow_up":true,"tools":[],"slots":{{"building_ref":"3"}},"inherit_from_previous":["tools","time","metric"],"clarification":null,"confidence":0.76}}
User: tell me a joke
JSON: {{"intent":"out_of_scope","is_follow_up":false,"tools":[],"slots":{{}},"inherit_from_previous":[],"clarification":{{"field":"topic","question":"I can help with EMS and DaxView questions. What site, device, or energy topic should I check?"}},"confidence":0.98}}
"""


def parse_planner_response(data: dict, allowlist: set[str]) -> PlannerOutput:
    content = data.get("message", {}).get("content")
    if content is None:
        content = data.get("response", "")
    payload = json.loads(str(content))
    plan = PlannerOutput(**payload)
    invalid = [tool for tool in plan.tools if tool not in allowlist]
    if invalid:
        raise ValueError(f"planner returned unsupported tools: {', '.join(invalid)}")
    return plan


def validate_planned_arguments(arguments_by_tool: dict[str, dict]) -> dict[str, dict]:
    return {tool: validate_tool_arguments(tool, args) for tool, args in arguments_by_tool.items()}

