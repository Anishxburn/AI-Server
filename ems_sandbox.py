"""Local synthetic EMS data and deterministic calculations for dashboard testing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def sample_dataset() -> dict:
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
    daily_energy = [
        {"date": (today - timedelta(days=6 - index)).isoformat(), "value": value}
        for index, value in enumerate([12840, 13620, 12110, 14980, 15420, 14310, 13190])
    ]
    devices = [
        {"device_id": 519, "name": "Main Incomer", "energy_kwh": 68420.0, "demand_kw": 428.6, "status": "online"},
        {"device_id": 604, "name": "Chiller Plant A", "energy_kwh": 38210.0, "demand_kw": 246.8, "status": "online"},
        {"device_id": 509, "name": "Air Handling Unit 3", "energy_kwh": 24790.0, "demand_kw": 161.4, "status": "online"},
        {"device_id": 512, "name": "Cold Room North", "energy_kwh": 19360.0, "demand_kw": 119.2, "status": "offline"},
        {"device_id": 513, "name": "Lighting Panel B", "energy_kwh": 12480.0, "demand_kw": 78.5, "status": "online"},
    ]
    demand = [
        {"timestamp": f"{today.isoformat()}T{hour:02d}:00:00+08:00", "value": value}
        for hour, value in enumerate([302, 288, 276, 269, 281, 334, 387, 421, 408, 396, 412, 429,
                                      415, 402, 391, 408, 422, 417, 398, 376, 351, 329, 315, 298])
    ]
    telemetry = [
        {"timestamp": f"{today.isoformat()}T{hour:02d}:00:00+08:00", "voltage_v": voltage,
         "current_a": current, "power_factor": pf}
        for hour, voltage, current, pf in zip(
            range(8), [238.4, 239.1, 237.8, 241.2, 242.0, 239.6, 238.9, 240.3],
            [186.0, 193.2, 201.5, 214.8, 226.1, 219.7, 207.3, 198.4],
            [0.91, 0.92, 0.90, 0.89, 0.88, 0.90, 0.92, 0.93],
        )
    ]
    alarms = [
        {"type": "Low power factor", "severity": "warning", "device_id": 519, "count_7d": 8},
        {"type": "Device offline", "severity": "critical", "device_id": 512, "count_7d": 3},
        {"type": "Voltage deviation", "severity": "warning", "device_id": 604, "count_7d": 2},
    ]
    return {
        "dataset_name": "Synthetic Office EMS Sample",
        "source": "synthetic_demo_data",
        "is_live": False,
        "site": {"site_id": 17, "name": "Sample Office Campus", "timezone": "Asia/Kuala_Lumpur"},
        "device_energy_period": "last 30 days",
        "demand_interval_minutes": 60,
        "daily_energy": daily_energy,
        "devices": devices,
        "demand_hourly": demand,
        "telemetry_hourly": telemetry,
        "alarms": alarms,
        "assumptions": {"carbon_factor_kg_per_kwh": 0.4, "carbon_factor_source": "illustrative demo assumption"},
    }


def _chart(chart_type: str, title: str, labels: list[str], values: list[float], unit: str, name: str) -> dict:
    return {"type": chart_type, "title": title, "labels": labels,
            "series": [{"name": name, "unit": unit, "data": values}]}


def prepare_sandbox_context(message: str, dataset: dict) -> tuple[dict, list[dict], dict | None]:
    """Compute trusted facts and optional visualization/report from synthetic readings."""
    lowered = message.lower()
    devices = sorted(dataset["devices"], key=lambda row: row["energy_kwh"], reverse=True)
    demand = dataset["demand_hourly"]
    peak = max(demand, key=lambda row: row["value"])
    low = min(demand, key=lambda row: row["value"])
    daily = dataset["daily_energy"]
    telemetry = dataset["telemetry_hourly"]
    facts = {
        "site": dataset["site"],
        "daily_energy_total_kwh": round(sum(row["value"] for row in daily), 2),
        "daily_energy_high": max(daily, key=lambda row: row["value"]),
        "daily_energy_low": min(daily, key=lambda row: row["value"]),
        "device_energy_ranking": devices,
        "device_energy_period": dataset["device_energy_period"],
        "device_energy_total_kwh": round(sum(row["energy_kwh"] for row in devices), 2),
        "peak_demand_kw": peak,
        "minimum_demand_kw": low,
        "average_demand_kw": round(sum(row["value"] for row in demand) / len(demand), 2),
        "telemetry_hourly": telemetry,
        "alarms": dataset["alarms"],
        "carbon_estimate": {
            "kg_co2e": round(sum(row["value"] for row in daily) * dataset["assumptions"]["carbon_factor_kg_per_kwh"], 2),
            "factor_kg_co2e_per_kwh": dataset["assumptions"]["carbon_factor_kg_per_kwh"],
            "period": "last 7 days",
            "status": "illustrative only; replace the factor with the applicable reporting factor",
        },
    }

    charts: list[dict] = []
    wants_chart = any(word in lowered for word in ("chart", "graph", "plot", "visual", "trend"))
    if wants_chart or any(word in lowered for word in ("top", "rank", "highest energy", "most energy")):
        if any(word in lowered for word in ("device", "consumer", "rank", "top")):
            rows = devices[:5]
            charts.append(_chart("bar", "Top Device Energy Consumers",
                                 [f"{row['name']} (ID {row['device_id']})" for row in rows],
                                 [row["energy_kwh"] for row in rows], "kWh", "Energy"))
        elif any(word in lowered for word in ("demand", "peak", "load")):
            charts.append(_chart("line", "Hourly Sample Demand", [row["timestamp"][11:16] for row in demand],
                                 [row["value"] for row in demand], "kW", "Demand"))
        elif any(word in lowered for word in ("voltage", "current", "power factor")):
            metric = "current_a" if "current" in lowered else "power_factor" if "power factor" in lowered else "voltage_v"
            label, unit = {"current_a": ("Current", "A"), "power_factor": ("Power Factor", "ratio"),
                           "voltage_v": ("Voltage", "V")}[metric]
            charts.append(_chart("line", f"Hourly Sample {label}", [row["timestamp"][11:16] for row in telemetry],
                                 [row[metric] for row in telemetry], unit, label))
        else:
            charts.append(_chart("bar", "Daily Site Energy", [row["date"] for row in daily],
                                 [row["value"] for row in daily], "kWh", "Energy"))

    report = None
    if any(word in lowered for word in ("report", "summary", "overview")):
        report = {
            "title": f"EMS Summary - {dataset['site']['name']}",
            "period": {"start": daily[0]["date"], "end": daily[-1]["date"]},
            "sections": [
                {"heading": "Energy", "total_kwh": facts["daily_energy_total_kwh"], "daily_values": daily},
                {"heading": "Leading consumers", "devices": devices[:5]},
                {"heading": "Demand", "peak": peak, "average_kw": facts["average_demand_kw"]},
                {"heading": "Alarms", "items": dataset["alarms"]},
            ],
            "data_source": "synthetic_demo_data",
        }
    return facts, charts, report


def formula_reference() -> list[dict]:
    return [
        {"name": "Interval energy", "formula": "E = sum(interval energy)", "unit": "kWh",
         "needs": "Non-overlapping interval consumption values; cumulative registers must be differenced first."},
        {"name": "Maximum demand", "formula": "max(interval-average active power)", "unit": "kW",
         "needs": "Configured demand interval and valid interval-average kW readings; this synthetic sample uses 60-minute intervals."},
        {"name": "Load factor", "formula": "average demand / peak demand x 100%", "unit": "%",
         "needs": "Average and peak demand from the same period and interval definition."},
        {"name": "Power factor", "formula": "active power / apparent power", "unit": "ratio",
         "needs": "Synchronized kW and kVA readings; do not assume a meter's PF convention."},
        {"name": "Estimated emissions", "formula": "energy (kWh) x emissions factor (kgCO2e/kWh)", "unit": "kgCO2e",
         "needs": "A reporting-boundary-appropriate factor and period; this demo uses an illustrative factor only."},
        {"name": "Percentage difference", "formula": "(A - B) / B x 100%", "unit": "%",
         "needs": "A non-zero baseline B; state the comparison direction."},
    ]
