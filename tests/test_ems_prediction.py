"""Focused tests for routing and daily energy prediction without service dependencies."""

import ast
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import re
import statistics
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {
    node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)
}
NAMES = (
    "requested_top_limit", "requested_forecast_days", "select_historical_operations",
    "parse_datetime", "local_bucket_date", "first_list", "predict_daily_energy",
    "is_follow_up_message", "resolve_follow_up_message", "needs_ems_library",
    "build_compliance_context", "has_time_scope", "first_regex_int", "needs_device_choice",
    "run_authorized_energy_prediction",
)


def load_functions():
    module = ast.Module(body=[FUNCTIONS[name] for name in NAMES], type_ignores=[])
    env = {
        "__builtins__": __builtins__, "date": date, "datetime": datetime,
        "timedelta": timedelta, "timezone": timezone, "math": math,
        "re": re, "statistics": statistics,
        "DAXVIEW_TOOL_KEYWORDS": next(
            ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DAXVIEW_TOOL_KEYWORDS" for target in node.targets)
        ),
        "FOLLOW_UP_PHRASES": next(
            ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FOLLOW_UP_PHRASES" for target in node.targets)
        ),
        "ENERGY_COMPLIANCE_CONTEXT": next(
            ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ENERGY_COMPLIANCE_CONTEXT" for target in node.targets)
        ),
    }
    exec(compile(module, str(SOURCE), "exec"), env)
    return env


class EmsPredictionTests(unittest.TestCase):
    def setUp(self):
        self.env = load_functions()

    def test_alarm_ranking_routes_only_to_alarm_frequency(self):
        question = "What alarm types occurred most often at this site in the last 7 days? Rank the top five."
        self.assertEqual(self.env["select_historical_operations"](question), ["alarm_frequency_summary"])
        self.assertEqual(self.env["requested_top_limit"](question), 5)

    def test_forecast_does_not_request_energy_summary_as_separate_answer(self):
        question = "Forecast energy usage for the next 7 days."
        self.assertEqual(self.env["select_historical_operations"](question), ["energy_forecast"])
        self.assertEqual(self.env["requested_forecast_days"](question), 7)

    def test_prediction_uses_complete_days_and_reports_backtest(self):
        today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        rows = [{"date": (today - timedelta(days=day)).isoformat(), "value": 100 + day % 7}
                for day in range(1, 36)]
        rows.append({"date": today.isoformat(), "value": 10000})
        result = self.env["predict_daily_energy"]({"buckets": rows, "unit": "kWh"}, 7)
        self.assertEqual(len(result["forecast"]), 7)
        self.assertEqual(result["history_end"], (today - timedelta(days=1)).isoformat())
        self.assertEqual(result["observed_days"], 35)
        self.assertLess(result["forecast"][0]["value"], 200)
        self.assertGreaterEqual(result["backtest_mae"], 0)

    def test_sparse_history_is_rejected(self):
        today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        rows = [{"date": (today - timedelta(days=day)).isoformat(), "value": 100}
                for day in range(1, 10)]
        with self.assertRaisesRegex(ValueError, "at least 21 complete daily readings"):
            self.env["predict_daily_energy"]({"buckets": rows}, 7)

    def test_delayed_reading_does_not_forecast_past_date(self):
        today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        rows = [{"date": (today - timedelta(days=day)).isoformat(), "value": 100}
                for day in range(2, 37)]
        result = self.env["predict_daily_energy"]({"buckets": rows}, 1)
        self.assertEqual(result["forecast"][0]["date"], today.isoformat())

    def test_prediction_uses_authorized_site_summary(self):
        today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        rows = [{"date": (today - timedelta(days=day)).isoformat(), "value": 100}
                for day in range(1, 36)]
        source = {"structuredContent": {"data": {"buckets": rows, "unit": "kWh"}}}
        self.env["request_daxview_data_plan"] = Mock(return_value={"authorization_id": "authorized"})
        self.env["call_authorized_historical_tool"] = Mock(return_value=source)
        self.env["mcp_structured_result"] = lambda result: result["structuredContent"]
        self.env["historical_result_data"] = lambda result, keys: result["structuredContent"]["data"]
        self.env["log_event"] = Mock()
        self.env["debug_trace_event"] = Mock()
        result = self.env["run_authorized_energy_prediction"](
            "turn", {"site_id": 17, "start": "a", "end": "b", "timezone": "Asia/Kuala_Lumpur", "forecast_days": 7}, "request"
        )
        plan_args = self.env["request_daxview_data_plan"].call_args.args
        self.assertEqual(plan_args[1], "site_energy_summary")
        self.assertEqual(plan_args[2]["bucket"], "day")
        self.env["call_authorized_historical_tool"].assert_called_once()
        self.assertEqual(self.env["call_authorized_historical_tool"].call_args.args[1], "authorized")
        self.assertEqual(len(result["structuredContent"]["data"]["forecast"]), 7)

    def test_follow_up_keeps_new_tool_intent(self):
        self.env["previous_daxview_turn"] = Mock(return_value={
            "id": "previous", "user_message": "Show active alarms for the last 7 days",
            "context": {"site_id": 17},
        })
        self.env["log_event"] = Mock()
        message, context = self.env["resolve_follow_up_message"](
            "current", "Follow up: can you forecast energy for the next 7 days?", {"site_id": 17}, "conversation", "request"
        )
        self.assertEqual(self.env["select_historical_operations"](message), ["energy_forecast"])
        self.assertEqual(context["site_id"], 17)

    def test_device_choice_continues_max_demand_question(self):
        self.env["previous_daxview_turn"] = Mock(return_value={
            "id": "previous", "user_message": "Can you get max demand for the device?",
            "context": {"site_id": 17},
        })
        self.env["log_event"] = Mock()
        message, context = self.env["resolve_follow_up_message"](
            "current", "device ID 380 for the last 7 days", {"site_id": 17}, "conversation", "request"
        )
        self.assertIn("demand_peak_summary", self.env["select_historical_operations"](message))
        self.assertFalse(self.env["needs_device_choice"](message))
        self.assertEqual(context["site_id"], 17)

    def test_standards_are_not_added_to_routine_site_answer(self):
        self.assertEqual(self.env["build_compliance_context"]("Show site energy for the last 7 days", ["site_energy_summary"]), [])
        self.assertTrue(self.env["build_compliance_context"]("How does ISO 50001 use this energy data?", ["site_energy_summary"]))
        self.assertFalse(self.env["needs_ems_library"]("Rank the top alarm types this week"))


if __name__ == "__main__":
    unittest.main()
