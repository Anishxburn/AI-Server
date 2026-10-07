"""Strict DaxView MCP tool argument contracts.

TODO: sync this file with docs/MCP/EMS_TOOL_REFERENCE.md in the DaxView repo.
The current schema mirrors the AI server integration plan and the argument
builder in server.py. Unknown fields are rejected before a plan request is made.
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except Exception:  # pragma: no cover - tests install pydantic transitively
    BaseModel = object  # type: ignore
    ConfigDict = None  # type: ignore
    Field = None  # type: ignore
    ValidationError = ValueError  # type: ignore


class StrictModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")

    if ConfigDict is None:  # pydantic v1 fallback
        class Config:
            extra = "forbid"


class SiteScope(StrictModel):
    site_id: int
    building_id: int | None = None


class WindowScope(SiteScope):
    start: str
    end: str
    timezone: str = "Asia/Kuala_Lumpur"


class TimeScope(SiteScope):
    start_time: str
    end_time: str
    timezone: str = "Asia/Kuala_Lumpur"


class TelemetryTopConsumers(WindowScope):
    limit: int = 5


class SiteEnergySummary(WindowScope):
    bucket: Literal["hour", "day", "week", "month"] | str = "day"


class AlarmFrequencySummary(WindowScope):
    limit: int = 10


class SiteMetadataSummary(SiteScope):
    pass


class SiteDeviceList(SiteScope):
    limit: int = 100


class TelemetryTimeseries(TimeScope):
    device_id: int
    metric: Literal["energy", "demand", "current", "voltage", "power_factor", "frequency", "thd"] | str
    bucket: str = "1d"
    aggregation: str = "auto"
    value_mode: str = "auto"
    limit: int = 500


class TelemetryMetricCatalog(SiteScope):
    device_id: int | None = None
    start_time: str | None = None
    end_time: str | None = None
    timezone: str = "Asia/Kuala_Lumpur"
    limit: int = 100


class LatestTelemetrySnapshot(SiteScope):
    device_id: int | None = None
    metrics: list[str] | None = None
    limit: int = 100


class ActiveAlarmSummary(SiteScope):
    limit: int = 50


class MeterStatusSummary(SiteScope):
    limit: int = 100


class EnergyComparisonSummary(SiteScope):
    timezone: str = "Asia/Kuala_Lumpur"
    period_a_start: str
    period_a_end: str
    period_b_start: str
    period_b_end: str


class DataAvailabilitySummary(TimeScope):
    metric: str | None = None
    device_id: int | None = None


class AlarmDetailLookup(SiteScope):
    alarm_id: int


class PowerQualitySummary(TimeScope):
    device_id: int | None = None
    limit: int = 100


class DemandPeakSummary(TimeScope):
    device_id: int | None = None


class TariffCostSummary(TimeScope):
    device_id: int | None = None


class DeviceEnergyBreakdown(TimeScope):
    group_by: str = "device"
    limit: int = 20


class EnergyForecast(SiteScope):
    forecast_start: str
    forecast_end: str
    training_days: int
    timezone: str = "Asia/Kuala_Lumpur"


class AnomalyDetectionSummary(TimeScope):
    device_id: int | None = None
    limit: int = 20


class ReportSummary(TimeScope):
    pass


TOOL_SCHEMAS: dict[str, type[StrictModel]] = {
    "telemetry_top_consumers": TelemetryTopConsumers,
    "site_energy_summary": SiteEnergySummary,
    "alarm_frequency_summary": AlarmFrequencySummary,
    "site_metadata_summary": SiteMetadataSummary,
    "site_device_list": SiteDeviceList,
    "telemetry_timeseries": TelemetryTimeseries,
    "telemetry_metric_catalog": TelemetryMetricCatalog,
    "latest_telemetry_snapshot": LatestTelemetrySnapshot,
    "active_alarm_summary": ActiveAlarmSummary,
    "meter_status_summary": MeterStatusSummary,
    "energy_comparison_summary": EnergyComparisonSummary,
    "data_availability_summary": DataAvailabilitySummary,
    "alarm_detail_lookup": AlarmDetailLookup,
    "power_quality_summary": PowerQualitySummary,
    "demand_peak_summary": DemandPeakSummary,
    "tariff_cost_summary": TariffCostSummary,
    "device_energy_breakdown": DeviceEnergyBreakdown,
    "energy_forecast": EnergyForecast,
    "anomaly_detection_summary": AnomalyDetectionSummary,
    "report_summary": ReportSummary,
}


TOOL_ARGUMENT_KEYS = {
    name: set(model.model_fields.keys() if hasattr(model, "model_fields") else model.__fields__.keys())
    for name, model in TOOL_SCHEMAS.items()
}


def strip_tool_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key in TOOL_ARGUMENT_KEYS.get(tool, set())}


def validate_tool_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool not in TOOL_SCHEMAS:
        raise ValueError(f"Unsupported DaxView tool: {tool}")
    clean = strip_tool_arguments(tool, arguments)
    model = TOOL_SCHEMAS[tool](**clean)
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)

