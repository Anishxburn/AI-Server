"""Independent daily-energy datasets and reproducible forecasting experiments."""

import csv
import hashlib
import io
import json
import math
import sqlite3
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse

LOCAL_ZONE = timezone(timedelta(hours=8))
MAX_ROWS = 3660
METHODS = {"weekly": "Same-weekday average", "smoothing": "Seasonal exponential smoothing"}


def daily_dataset(payload: dict, today: date | None = None) -> dict:
    today = today or datetime.now(LOCAL_ZONE).date()
    if payload.get("reading_kind", "interval_energy") != "interval_energy":
        raise ValueError("Use daily interval energy, not cumulative meter-register readings.")
    if payload.get("metric", "energy") != "energy":
        raise ValueError("This prediction pilot accepts daily energy only, not kW demand or current.")
    unit = str(payload.get("unit") or "kWh").strip()
    factors = {"kWh": 1.0, "MWh": 1000.0, "Wh": 0.001}
    if unit not in factors:
        raise ValueError("Energy units must be Wh, kWh, or MWh.")
    rows = payload.get("rows")
    if "csv" in payload:
        rows = list(csv.DictReader(io.StringIO(str(payload["csv"]).lstrip("\ufeff"))))
    if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
        raise ValueError(f"Provide 1 to {MAX_ROWS} daily rows with date and value columns.")
    readings = {}
    excluded = 0
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must contain a date and value.")
        stamp = str(row.get("date") or row.get("timestamp") or row.get("bucket") or "")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            day = parsed.astimezone(LOCAL_ZONE).date() if parsed.tzinfo else parsed.date()
            raw_value = row.get("value") if row.get("value") is not None else row.get("kwh")
            if isinstance(raw_value, bool):
                raise ValueError()
            value = float(raw_value)
        except (ValueError, TypeError):
            raise ValueError(f"Row {index} requires a valid ISO date and numeric energy value.") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Row {index} has an invalid or negative energy value.")
        row_unit = str(row.get("unit") or unit).strip()
        if row_unit not in factors:
            raise ValueError(f"Row {index} has incompatible energy units.")
        for scope_key in ("site_id", "device_id"):
            row_scope = row.get(scope_key)
            if row_scope is not None and str(row_scope) != str(payload.get(scope_key)):
                raise ValueError(f"Row {index} has a different {scope_key}; import one series at a time.")
        if day >= today:
            excluded += 1
            continue
        if day in readings:
            raise ValueError(f"Duplicate day {day}: provide one daily total per selected scope.")
        readings[day] = round(value * factors[row_unit], 6)
    days = sorted(readings)
    if not days:
        raise ValueError("No complete historical days were returned.")
    span = (days[-1] - days[0]).days + 1
    if span > MAX_ROWS:
        raise ValueError("The history window is too large.")
    missing = [(days[0] + timedelta(days=n)).isoformat() for n in range(span)
               if days[0] + timedelta(days=n) not in readings]
    return {
        "name": str(payload.get("name") or "Energy history")[:120],
        "site_id": payload.get("site_id"), "device_id": payload.get("device_id"),
        "metric": "energy", "unit": "kWh", "reading_kind": "interval_energy",
        "source": str(payload.get("source") or "upload")[:80],
        "rows": [{"date": day.isoformat(), "value": readings[day]} for day in days],
        "quality": {"observed_days": len(days), "missing_days": missing,
                    "coverage_percent": round(len(days) / span * 100, 2),
                    "excluded_incomplete_days": excluded,
                    "latest_age_days": (today - days[-1]).days},
    }


def forecast_values(values: list[float], horizon: int, method: str) -> list[float]:
    if method == "weekly":
        return [statistics.mean(values[index] for index in range(len(values))
                                if index % 7 == (len(values) + step) % 7)
                for step in range(horizon)]
    if method != "smoothing":
        raise ValueError("Unknown prediction method.")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(values, trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=7,
                                 initialization_method="estimated").fit(optimized=True)
    output = [max(0.0, float(value)) for value in model.forecast(horizon)]
    if not all(math.isfinite(value) for value in output):
        raise ValueError("Forecasting returned non-finite values.")
    return output


def forecast_dataset(dataset: dict, horizon: int = 7, method: str = "auto") -> dict:
    if horizon < 1 or horizon > 14 or method not in {"auto", *METHODS}:
        raise ValueError("Choose a horizon of 1-14 days and a supported method.")
    rows = dataset["rows"]
    if dataset["quality"]["missing_days"]:
        raise ValueError("History contains missing days. Import a continuous series before forecasting; gaps are not filled with zero.")
    if len(rows) < max(28, 14 + horizon):
        raise ValueError("At least 28 consecutive complete days are needed for this experiment.")
    values = [row["value"] for row in rows]
    origins = [len(values) - horizon * fold for fold in (3, 2, 1)
               if len(values) - horizon * fold >= 14]
    candidates = {}
    predictions = {}
    failures = {}
    for candidate in METHODS:
        try:
            checks = []
            for origin in origins:
                estimates = forecast_values(values[:origin], horizon, candidate)
                checks.extend({"date": rows[origin + step]["date"], "actual": values[origin + step],
                               "predicted": estimate, "lead": step + 1}
                              for step, estimate in enumerate(estimates))
            errors = [abs(check["actual"] - check["predicted"]) for check in checks]
            denominator = sum(abs(check["actual"]) for check in checks)
            candidates[candidate] = {"name": METHODS[candidate], "mae": round(statistics.mean(errors), 4),
                                     "wape_percent": round(sum(errors) / denominator * 100, 2) if denominator else None,
                                     "folds": len(origins), "evaluated_points": len(checks)}
            predictions[candidate] = checks
        except Exception as error:
            failures[candidate] = type(error).__name__
    if method != "auto" and method not in candidates:
        raise ValueError(f"{METHODS[method]} could not be fitted to this dataset.")
    if not candidates:
        raise ValueError("No forecasting method could be fitted.")
    selected = min(candidates, key=lambda item: candidates[item]["mae"]) if method == "auto" else method
    estimates = forecast_values(values, horizon, selected)
    checks = predictions[selected]
    errors = sorted(abs(check["actual"] - check["predicted"]) for check in checks)
    error_band = errors[min(len(errors) - 1, math.ceil(0.9 * len(errors)) - 1)]
    last_day = date.fromisoformat(rows[-1]["date"])
    forecast = [{"date": (last_day + timedelta(days=index + 1)).isoformat(),
                 "value": round(value, 4), "lower": round(max(0, value - error_band), 4),
                 "upper": round(value + error_band, 4)} for index, value in enumerate(estimates)]
    reference = sum(values[-horizon:])
    total = sum(estimates)
    return {"dataset_id": dataset.get("id"), "dataset_name": dataset["name"],
            "source": dataset["source"], "site_id": dataset.get("site_id"),
            "device_id": dataset.get("device_id"), "unit": "kWh", "method": selected,
            "method_name": METHODS[selected], "horizon": horizon, "history": rows,
            "forecast": forecast, "backtest": checks, "models": candidates, "model_failures": failures,
            "quality": dataset["quality"], "forecast_total": round(total, 4),
            "reference_total": round(reference, 4),
            "change_percent": round((total - reference) / reference * 100, 2) if reference else None,
            "band_label": "Historical absolute-error band (90th percentile); not a calibrated confidence interval",
            "created_at": datetime.now(timezone.utc).isoformat()}


class PredictionStore:
    def __init__(self, directory):
        self.path = Path(directory) / "prediction_lab.sqlite3"

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save_dataset(self, dataset):
        identity = {key: dataset.get(key) for key in ("site_id", "device_id", "unit", "rows", "source")}
        dataset = {**dataset, "id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]}
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO datasets VALUES (?, ?, ?)",
                         (dataset["id"], json.dumps(dataset, allow_nan=False), datetime.now(timezone.utc).isoformat()))
        return dataset

    def get(self, table, identifier):
        if table not in {"datasets", "runs"}:
            raise ValueError("Unknown data collection.")
        with self.connect() as conn:
            row = conn.execute(f"SELECT payload FROM {table} WHERE id = ?", (identifier,)).fetchone()
        if not row:
            raise ValueError("Saved dataset or run was not found.")
        return json.loads(row["payload"])

    def list(self, table):
        if table not in {"datasets", "runs"}:
            raise ValueError("Unknown data collection.")
        with self.connect() as conn:
            rows = conn.execute(f"SELECT id, payload, created_at FROM {table} ORDER BY created_at DESC LIMIT 100").fetchall()
        output = []
        for row in rows:
            item = json.loads(row["payload"])
            output.append({key: value for key, value in {**item, "id": row["id"], "created_at": row["created_at"]}.items()
                           if key not in {"rows", "history", "forecast", "backtest"}})
        return output

    def save_run(self, result):
        result = {**result, "id": str(uuid.uuid4())}
        with self.connect() as conn:
            conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                         (result["id"], result["dataset_id"], json.dumps(result, allow_nan=False), result["created_at"]))
        return result


def extract_history(source_url, source_token, allowed_sites, site_id, history_days=90, device_id=None):
    if not source_url:
        raise ValueError("Bulk export is not configured. Import a daily-energy CSV or configure AI_PREDICTION_SOURCE_URL.")
    if int(site_id) not in {int(site["id"]) for site in allowed_sites}:
        raise ValueError("This site is not enabled for dashboard data extraction.")
    if not 28 <= int(history_days) <= 365:
        raise ValueError("Choose 28-365 days of history.")
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The configured export URL is invalid.")
    end = datetime.now(LOCAL_ZONE).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(history_days))
    rows = []
    unit = None
    while start < end:
        chunk_end = min(start + timedelta(days=31), end)
        payload = {"site_id": int(site_id), "device_id": device_id, "metric": "energy",
                   "start": start.isoformat(), "end": chunk_end.isoformat(),
                   "timezone": "Asia/Kuala_Lumpur", "bucket": "day"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if source_token:
            headers["Authorization"] = f"Bearer {source_token}"
        request = Request(source_url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urlopen(request, timeout=30) as response:
            body = response.read(4_000_001)
        if len(body) > 4_000_000:
            raise ValueError("Export response exceeded the allowed size.")
        data = json.loads(body)
        if data.get("status") == "error" or str(data.get("site_id")) != str(site_id):
            raise ValueError("Export failed or returned a different site.")
        if str(data.get("device_id")) != str(device_id):
            raise ValueError("Export returned a different device scope.")
        if data.get("reading_kind") != "interval_energy" or data.get("metric", "energy") != "energy":
            raise ValueError("Export must return daily interval energy.")
        if unit and unit != data.get("unit"):
            raise ValueError("Export units changed between batches.")
        unit = data.get("unit")
        if unit not in {"Wh", "kWh", "MWh"}:
            raise ValueError("Export must specify an energy unit: Wh, kWh or MWh.")
        if not isinstance(data.get("rows"), list):
            raise ValueError("Export must contain a rows array.")
        # Each response is checked before combining half-open date ranges.
        chunk = daily_dataset({**data, "rows": data["rows"]})
        if any(not start.date().isoformat() <= row["date"] < chunk_end.date().isoformat() for row in chunk["rows"]):
            raise ValueError("Export returned dates outside its requested range.")
        rows.extend(chunk["rows"])
        start = chunk_end
    return daily_dataset({"name": f"Site {site_id} energy history", "site_id": int(site_id),
                          "device_id": device_id, "source": "configured_export", "unit": "kWh", "rows": rows})


def demo_dataset():
    today = datetime.now(LOCAL_ZONE).date()
    rows = [{"date": (today - timedelta(days=90 - index)).isoformat(),
             "value": round(820 + 1.5 * index + (index % 7 < 5) * 210 + 35 * math.sin(index * 1.7), 2)}
            for index in range(90)]
    return daily_dataset({"name": "Sample site - synthetic energy", "source": "demo_synthetic", "rows": rows})


class PredictionService:
    def __init__(self, directory, source_url="", source_token="", sites=None):
        self.store = PredictionStore(directory)
        self.source_url = source_url
        self.source_token = source_token
        self.sites = sites or []
        self.jobs = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prediction-lab")

    def submit(self, task):
        with self.lock:
            if sum(job["status"] in {"queued", "running"} for job in self.jobs.values()) >= 4:
                raise ValueError("Prediction workers are busy. Wait for an active job to finish.")
            if len(self.jobs) >= 100:
                oldest = next((key for key, job in self.jobs.items() if job["status"] in {"completed", "failed"}), None)
                if oldest:
                    del self.jobs[oldest]
            identifier = str(uuid.uuid4())
            self.jobs[identifier] = {"id": identifier, "status": "queued"}

        def work():
            with self.lock:
                self.jobs[identifier]["status"] = "running"
            try:
                result = task()
                with self.lock:
                    self.jobs[identifier].update(status="completed", result=result)
            except Exception as error:
                # Upstream HTTP errors can contain private URLs and tokens.
                detail = str(error) if isinstance(error, ValueError) else "The data source or forecasting service failed. Check its availability and configuration."
                with self.lock:
                    self.jobs[identifier].update(status="failed", error=detail)
        self.executor.submit(work)
        return {"job_id": identifier, "status": "queued"}

    def get(self, route, query):
        identifier = (query.get("id") or [""])[0]
        if route == "":
            return {"sites": self.sites, "source_configured": bool(self.source_url),
                    "methods": METHODS, "metric": "daily interval energy", "unit": "kWh"}
        if route in {"datasets", "runs"}:
            return {"items": self.store.list(route)}
        if route in {"dataset", "run"}:
            return self.store.get("datasets" if route == "dataset" else "runs", identifier)
        if route == "jobs":
            with self.lock:
                if identifier not in self.jobs:
                    raise ValueError("Job not found; it may have expired after a restart. Check saved runs.")
                return dict(self.jobs[identifier])
        raise ValueError("Unknown prediction operation.")

    def post(self, route, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        if route == "import":
            return self.store.save_dataset(daily_dataset({**payload, "source": "upload"}))
        if route == "demo":
            return self.store.save_dataset(demo_dataset())
        if route == "extract":
            site_id = int(payload.get("site_id") or 0)
            history_days = int(payload.get("history_days") or 90)
            device_id = int(payload["device_id"]) if payload.get("device_id") else None
            return self.submit(lambda: self.store.save_dataset(extract_history(
                self.source_url, self.source_token, self.sites, site_id, history_days, device_id)))
        if route == "forecast":
            dataset = self.store.get("datasets", str(payload.get("dataset_id") or ""))
            horizon = int(payload.get("horizon") or 7)
            method = str(payload.get("method") or "auto")
            return self.submit(lambda: self.store.save_run(forecast_dataset(dataset, horizon, method)))
        raise ValueError("Unknown prediction operation.")
