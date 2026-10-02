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
    "requested_top_limit", "requested_forecast_days", "requested_energy_extrema",
    "requested_question_parts", "completed_daily_range", "wants_device_usage_ranking", "wants_all_devices",
    "select_historical_operations", "build_historical_arguments", "requested_device_id",
    "parse_datetime", "local_bucket_date", "first_list", "predict_daily_energy",
    "is_follow_up_message", "resolve_follow_up_message", "needs_ems_library",
    "build_compliance_context", "has_time_scope", "first_regex_int", "needs_device_choice",
    "run_authorized_energy_prediction", "run_daxview_integration_turn",
    "build_charts_from_historical_results", "summarize_site_energy",
    "ensure_historical_answer_coverage", "ranked_consumer_rows", "summarize_top_consumers",
    "summarize_historical_answers", "summarize_demand_peak", "device_selection_prompt_from_result",
    "build_device_choice_response",
    "daxview_history_event",
)


def load_functions():
    exception = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "PredictionSourceError")
    module = ast.Module(body=[exception] + [FUNCTIONS[name] for name in NAMES], type_ignores=[])
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

    def test_combined_daily_energy_question_uses_site_summary(self):
        question = "Show this site's daily energy use for the last 7 days and what was the highest and lowest reading?"
        self.assertEqual(self.env["select_historical_operations"](question), ["site_energy_summary"])
        self.assertEqual(self.env["requested_energy_extrema"](question), (True, True))
        self.assertEqual(len(self.env["requested_question_parts"](question)), 2)

    def test_device_ranking_uses_consumers_and_status_not_site_daily_totals(self):
        question = "Which devices contributed most to energy use during that period? Show the top five and tell me which are offline."
        self.assertEqual(
            self.env["select_historical_operations"](question),
            ["telemetry_top_consumers", "site_device_list"],
        )
        all_question = "List out all the devices and rank them based on usage"
        self.assertEqual(
            self.env["select_historical_operations"](all_question),
            ["telemetry_top_consumers", "site_device_list"],
        )
        self.env["requested_historical_range"] = Mock(return_value={"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"})
        args = self.env["build_historical_arguments"]("telemetry_top_consumers", {"site_id": 17}, all_question)
        self.assertEqual(args["limit"], 100)

    def test_ranking_sorts_numeric_usage_and_joins_current_status(self):
        self.env["historical_result_data"] = lambda result: result["data"]
        self.env["first_value"] = lambda row, keys: next((row[key] for key in keys if row.get(key) is not None), None)
        self.env["format_time_window"] = lambda data, arguments: "last 7 days"
        self.env["format_number"] = lambda value, precision=2: f"{value:.2f}"
        self.env["reading_detail"] = lambda *args: None
        results = [
            {"operation_id": "telemetry_top_consumers", "arguments": {"limit": 5}, "result": {"data": {
                "unit": "kWh", "rows": [
                    {"device_id": 2, "device_name": "Low meter", "value": 10},
                    {"device_id": 1, "device_name": "High meter", "value": 20},
                    {"device_id": 3, "device_name": "Missing meter", "value": None},
                ],
            }}},
            {"operation_id": "site_device_list", "result": {"data": {"devices": [
                {"device_id": 1, "device_name": "High meter", "status": "offline"},
                {"device_id": 2, "device_name": "Low meter", "status": "online"},
                {"device_id": 3, "device_name": "Missing meter", "status": "unknown"},
            ]}}},
        ]
        answer = self.env["summarize_historical_answers"](
            "List all devices and rank them by usage; show which are offline", results, "request"
        )
        self.assertLess(answer.index("1. High meter"), answer.index("2. Low meter"))
        self.assertIn("High meter (ID 1): offline", answer)
        self.assertIn("Missing meter (ID 3)", answer)
        self.assertIn("usage not returned, so not ranked", answer)

    def test_model_cannot_drop_ranked_device(self):
        self.env["historical_result_data"] = lambda result: result["data"]
        self.env["first_value"] = lambda row, keys: next((row[key] for key in keys if row.get(key) is not None), None)
        self.env["log_event"] = Mock()
        results = [{"operation_id": "telemetry_top_consumers", "arguments": {"limit": 2}, "result": {"data": {
            "rows": [{"device_name": "High meter", "value": 20}, {"device_name": "Low meter", "value": 10}]
        }}}]
        draft = "1. High meter: 20 kWh\n2. Low meter: 10 kWh"
        answer = self.env["ensure_historical_answer_coverage"](
            "Rank devices by energy usage", results, draft, "High meter used 20 kWh.", "request"
        )
        self.assertEqual(answer, draft)
        self.env["log_event"].assert_called_once()

    def test_daily_energy_extrema_use_seven_complete_days(self):
        question = "Show this site's daily energy use for the last 7 days and what was the highest and lowest reading?"
        arguments = self.env["completed_daily_range"](question)
        arguments["bucket"] = "day"
        first = self.env["parse_datetime"](arguments["start"]).astimezone(timezone(timedelta(hours=8))).date()
        today = first + timedelta(days=7)
        rows = [{"date": (first + timedelta(days=index)).isoformat(), "value": (index + 1) * 10}
                for index in range(7)]
        rows.append({"date": today.isoformat(), "value": 1})
        self.env["requested_energy_unit"] = lambda message, unit: (1.0, unit, 2)
        self.env["format_energy_value"] = lambda value, factor, unit, precision: f"{value / factor:.2f} {unit}"
        self.env["requested_comparison_dates"] = lambda message, year: []
        self.env["requested_comparison_months"] = lambda message, year: []
        self.env["wants_relative_summary"] = lambda message: True
        self.env["format_coverage_note"] = lambda data: None
        self.env["format_compliance_context"] = lambda context: ""
        answer = self.env["summarize_site_energy"]({"buckets": rows, "unit": "kWh"}, question, arguments)
        self.assertIn(f"Highest daily energy: {(today - timedelta(days=1)).isoformat()} at 70.00 kWh", answer)
        self.assertIn(f"Lowest daily energy: {first.isoformat()} at 10.00 kWh", answer)
        self.assertEqual(sum(line.startswith("- ") for line in answer.splitlines()), 7)
        self.assertNotIn(f"- {today.isoformat()}: 1.00 kWh", answer)

    def test_refinement_cannot_drop_requested_extrema(self):
        question = "Show daily energy for the last 7 days and what was the highest and lowest reading?"
        draft = "Daily values:\n- 2026-10-01: 10.00 kWh\n- 2026-10-02: 20.00 kWh\nHighest daily energy: 2026-10-02 at 20.00 kWh.\nLowest daily energy: 2026-10-01 at 10.00 kWh."
        self.env["log_event"] = Mock()
        answer = self.env["ensure_historical_answer_coverage"](
            question, [{"operation_id": "site_energy_summary"}], draft, "Energy use was 10 and 20 kWh.", "request"
        )
        self.assertIn("Highest daily energy: 2026-10-02 at 20.00 kWh", answer)
        self.assertIn("Lowest daily energy: 2026-10-01 at 10.00 kWh", answer)
        self.assertIn("- 2026-10-01: 10.00 kWh", answer)
        self.env["log_event"].assert_called_once()

    def test_refinement_keeps_both_tool_topics(self):
        question = "Show daily energy for the last 7 days and list the active alarms."
        self.assertEqual(
            self.env["select_historical_operations"](question),
            ["site_energy_summary", "active_alarm_summary"],
        )
        self.env["log_event"] = Mock()
        self.env["summarize_historical_answer"] = Mock(return_value="Active alarms: 2 critical.")
        answer = self.env["ensure_historical_answer_coverage"](
            question,
            [{"operation_id": "site_energy_summary", "result": {}},
             {"operation_id": "active_alarm_summary", "result": {}, "arguments": {}}],
            "Daily values:\n- 2026-10-01: 10.00 kWh", "Energy was 10.00 kWh on 2026-10-01.", "request",
        )
        self.assertIn("Active alarms: 2 critical.", answer)

    def test_highest_day_follow_up_inherits_previous_question(self):
        self.env["previous_daxview_turn"] = Mock(return_value={
            "id": "previous", "user_message": "Show daily site energy for the last 7 days",
            "context": {"site_id": 17},
        })
        self.env["log_event"] = Mock()
        message, context = self.env["resolve_follow_up_message"](
            "current", "Which day was highest?", {"site_id": 17}, "conversation", "request"
        )
        self.assertIn("last 7 days", message)
        self.assertEqual(self.env["select_historical_operations"](message), ["site_energy_summary"])
        self.assertEqual(context["site_id"], 17)

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

    def test_upstream_unavailable_is_not_treated_as_a_forecast(self):
        self.env["request_daxview_data_plan"] = Mock(return_value={"authorization_id": "authorized"})
        self.env["call_authorized_historical_tool"] = Mock(return_value={
            "isError": False,
            "structuredContent": {"status": "error", "error_code": "DAXVIEW_UNAVAILABLE"},
        })
        self.env["mcp_structured_result"] = lambda result: result["structuredContent"]
        with self.assertRaises(self.env["PredictionSourceError"]) as raised:
            self.env["run_authorized_energy_prediction"](
                "turn", {"site_id": 17, "forecast_days": 7}, "request"
            )
        self.assertEqual(raised.exception.error_code, "DAXVIEW_UNAVAILABLE")

    def test_upstream_unavailable_returns_readable_answer(self):
        self.env["resolve_follow_up_message"] = Mock(side_effect=lambda turn, message, context, conversation, request: (message, context))
        self.env["build_historical_arguments"] = Mock(return_value={"site_id": 17, "forecast_days": 7})
        self.env["run_authorized_energy_prediction"] = Mock(side_effect=self.env["PredictionSourceError"]("DAXVIEW_UNAVAILABLE"))
        self.env["debug_trace_event"] = Mock()
        self.env["log_event"] = Mock()
        result = self.env["run_daxview_integration_turn"](
            "turn", "Forecast energy usage for the next 7 days", {"site_id": 17}, "request", "conversation"
        )
        self.assertIn("couldn't calculate a forecast", result["reply"])
        self.assertEqual(result["charts"], [])
        self.env["log_event"].assert_called_once()

    def test_dashboard_can_preview_charts_without_daxview_chart_payload(self):
        chart = {"type": "line", "labels": ["2026-10-03"], "series": [{"data": [42]}]}
        self.env["AI_CHARTS_ENABLED"] = False
        self.env["chart_from_historical_result"] = Mock(return_value=chart)
        results = [{"operation_id": "energy_forecast", "result": {}, "arguments": {}}]
        self.assertEqual(self.env["build_charts_from_historical_results"](results), [])
        self.assertEqual(self.env["build_charts_from_historical_results"](results, preview=True), [chart])

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

    def test_displayed_device_id_does_not_trigger_choice_again(self):
        question = "(ID 519, Online, Virtual) I want max demand for this device for the last 7 days"
        self.assertEqual(self.env["requested_device_id"](question), 519)
        self.assertFalse(self.env["needs_device_choice"](question))
        self.assertEqual(self.env["select_historical_operations"](question), ["demand_peak_summary"])
        self.env["requested_historical_range"] = Mock(return_value={"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"})
        arguments = self.env["build_historical_arguments"]("demand_peak_summary", {"site_id": 17}, question)
        self.assertEqual(arguments["device_id"], 519)

    def test_site_wide_highest_demand_device_does_not_ask_for_device(self):
        question = "What is the highest demand device today?"
        self.assertEqual(self.env["select_historical_operations"](question), ["demand_peak_summary"])
        self.assertFalse(self.env["needs_device_choice"](question))

    def test_selected_device_context_does_not_trigger_choice_again(self):
        self.env["resolve_follow_up_message"] = Mock(side_effect=lambda turn, message, context, conversation, request: (message, context))
        self.env["build_device_choice_response"] = Mock()
        self.env["build_historical_arguments"] = Mock(return_value={"site_id": 17, "device_id": 519})
        self.env["request_daxview_data_plan"] = Mock(return_value={"authorization_id": "authorized"})
        self.env["call_authorized_historical_tool"] = Mock(return_value={"structuredContent": {"data": {"peak_kw": 42}}})
        self.env["summarize_historical_answers"] = Mock(return_value="Peak demand: 42 kW")
        self.env["refine_historical_answer_with_model"] = Mock(return_value="Peak demand: 42 kW")
        self.env["build_charts_from_historical_results"] = Mock(return_value=[])
        self.env["debug_trace_event"] = Mock()
        self.env["AI_REFINE_MCP_WITH_MODEL"] = False
        result = self.env["run_daxview_integration_turn"](
            "turn", "Max demand for this device", {"site_id": 17, "device_id": 519}, "request", "conversation"
        )
        self.assertEqual(result["reply"], "Peak demand: 42 kW")
        self.env["build_device_choice_response"].assert_not_called()

    def test_device_choices_include_full_returned_list_and_keep_time_scope(self):
        devices = [{"device_id": index, "device_name": f"Meter {index}", "status": "online"}
                   for index in range(1, 16)]
        self.env["historical_result_data"] = lambda result: result["data"]
        prompt, choices = self.env["device_selection_prompt_from_result"]({"data": {"devices": devices}})
        self.assertEqual(len(choices), 15)
        self.assertIn("first 12", prompt)
        self.env["build_historical_arguments"] = Mock(return_value={"site_id": 17})
        self.env["request_daxview_data_plan"] = Mock(return_value={"authorization_id": "authorized"})
        self.env["call_authorized_historical_tool"] = Mock(return_value={"data": {"devices": devices}})
        result = self.env["build_device_choice_response"](
            "turn", "Max demand for this device today", {"site_id": 17}, "request"
        )
        self.assertEqual(result["input_type"], "select")
        self.assertEqual(len(result["choices"]), 15)
        self.assertEqual(result["submit_template"], "max demand for device ID {value} for today")

    def test_history_keeps_select_metadata_after_reload(self):
        event = {"event_type": "waiting_for_user", "event_data": {
            "prompt": "Which device?", "fields": ["device_id"], "input_type": "select",
            "choices": [{"label": "AC kWh (ID 519)", "value": "519"}],
            "submit_template": "max demand for device ID {value} for today",
        }}
        history_item = self.env["daxview_history_event"](event)
        self.assertEqual(history_item["text"], "Which device?")
        self.assertEqual(history_item["choices"][0]["value"], "519")
        self.assertEqual(history_item["input_type"], "select")

    def test_demand_summary_distinguishes_historical_and_live(self):
        self.env["format_time_window"] = lambda data, arguments: "last 7 days"
        self.env["first_value"] = lambda row, keys: next((row[key] for key in keys if row.get(key) is not None), None)
        self.env["format_number"] = lambda value, precision=2: f"{value:.2f}"
        answer = self.env["summarize_demand_peak"](
            {"peak_kw": 42, "peak_time": "2026-10-01T10:00:00+08:00", "latest_kw": 15},
            "Show the live and 7d max demand", {"device_id": 519},
        )
        self.assertIn("device ID 519 (historical, last 7 days)", answer)
        self.assertIn("Peak demand: 42.00 kW", answer)
        self.assertIn("Latest returned demand: 15.00 kW", answer)
        self.assertIn("Live value: not provided", answer)

    def test_standards_are_not_added_to_routine_site_answer(self):
        self.assertEqual(self.env["build_compliance_context"]("Show site energy for the last 7 days", ["site_energy_summary"]), [])
        self.assertTrue(self.env["build_compliance_context"]("How does ISO 50001 use this energy data?", ["site_energy_summary"]))
        self.assertFalse(self.env["needs_ems_library"]("Rank the top alarm types this week"))


if __name__ == "__main__":
    unittest.main()
