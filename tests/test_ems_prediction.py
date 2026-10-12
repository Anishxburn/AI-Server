"""Focused tests for routing and daily energy prediction without service dependencies."""

import ast
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import re
import statistics
import unittest
from unittest.mock import Mock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from ems_contracts import question_metrics, metric_clarification, percentage_difference, result_problem
from contracts.mcp_manifest import build_public_calls, public_tool_for_internal
from contracts.tool_schemas import validate_tool_arguments


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {
    node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)
}
NAMES = (
    "requested_top_limit", "connected_load_calculation_answer", "requested_forecast_days", "requested_energy_extrema", "format_number", "format_time_window", "format_display_timestamp",
    "reading_detail",
    "requested_question_parts", "completed_daily_range", "wants_device_usage_ranking", "wants_all_devices",
    "supplied_demand_readings_answer", "requested_historical_range", "default_historical_range",
    "relative_historical_range", "requested_explicit_date_range", "requested_comparison_dates", "requested_month_ranges",
    "wants_lowest_consumer", "wants_top_lowest_sum", "wants_device_inventory", "wants_device_capability_discovery",
    "wants_available_parameters_follow_up", "select_historical_operations_without_device_timeseries",
    "wants_device_metric_timeseries", "explicit_mcp_tool_request", "planner_default_intents", "select_public_tools", "planner_slot_context", "choose_historical_operations", "first_device_id_from_text", "follow_up_metric_from_message",
    "select_historical_operations", "internal_for_public_operation", "build_historical_arguments", "requested_device_id", "requested_metric",
    "parse_datetime", "format_display_timestamp", "format_answer_timestamps", "local_bucket_date", "first_dict_with_list", "first_list", "first_value", "predict_daily_energy",
    "format_coverage_note", "is_follow_up_message", "resolve_follow_up_message", "needs_ems_library",
    "build_compliance_context", "has_time_scope", "first_regex_int", "needs_device_choice",
    "run_authorized_energy_prediction", "run_daxview_integration_turn",
    "build_charts_from_historical_results", "build_tables_from_historical_results", "summarize_site_energy", "summarize_alarm_frequency",
    "ensure_historical_answer_coverage", "ranked_consumer_rows", "summarize_top_consumers",
    "summarize_generic_tool", "summarize_historical_answer", "summarize_historical_answers", "summarize_demand_peak", "summarize_site_devices", "device_selection_prompt_from_result",
    "build_device_choice_response",
    "daxview_history_event",
    "wants_highest_demand_device", "run_highest_demand_device_ranking", "choose_historical_operations",
    "resolved_plan_for_result", "readable_historical_errors", "refine_historical_answer_with_model",
    "chart_number", "chart_value", "mcp_structured_result", "historical_result_data",
    "demand_peak_needs_manual_fallback", "calculate_demand_peak_from_telemetry", "calculate_demand_peak_from_energy",
    "try_top_consumers_breakdown_fallback", "redact_debug_value",
)


def load_functions():
    exception = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "PredictionSourceError")
    module = ast.Module(body=[exception] + [FUNCTIONS[name] for name in NAMES], type_ignores=[])
    env = {
        "__builtins__": __builtins__, "date": date, "datetime": datetime,
        "timedelta": timedelta, "timezone": timezone, "math": math,
        "ZoneInfo": ZoneInfo, "ZoneInfoNotFoundError": ZoneInfoNotFoundError,
        "AI_DISPLAY_TIMEZONE": "Asia/Kuala_Lumpur",
        "re": re, "statistics": statistics,
        "question_metrics": question_metrics, "metric_clarification": metric_clarification,
        "percentage_difference": percentage_difference,
        "result_problem": result_problem,
        "validate_tool_arguments": validate_tool_arguments,
        "public_tool_for_internal": public_tool_for_internal,
        "build_public_calls": build_public_calls,
        "DEMAND_RANKING_MAX_DEVICES": 20,
        "AI_CHAT_MEMORY_ENABLED": False,
        "MONTH_NAMES": next(
            ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MONTH_NAMES" for target in node.targets)
        ),
        "load_conversation_history": lambda conversation_id, turn_id: [],
        "AI_PLANNER_ENABLED": False,
        "AI_PLANNER_SHADOW_MODE": False,
        "AI_REFINE_MCP_WITH_MODEL": True,
        "AI_MCP_FALLBACKS_ENABLED": True,
        "AI_COMPARE_MODEL_ENABLED": False,
        "AI_COMPARE_MODEL_SHOW_TO_USER": False,
        "AI_COMPARE_MODELS": ["deepseek-r1:1.5b"],
        "CHAT_MODEL": "qwen3:8b",
        "DAXVIEW_ALLOWED_HISTORICAL_TOOLS": next(
            ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DAXVIEW_ALLOWED_HISTORICAL_TOOLS" for target in node.targets)
        ),
        "requested_historical_range": Mock(return_value={"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"}),
        "save_resolved_turn_context": Mock(),
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
    env["requested_historical_range_impl"] = env["requested_historical_range"]
    env["requested_historical_range"] = Mock(return_value={"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"})
    env.setdefault("log_event", Mock())
    env.setdefault("debug_trace_event", Mock())

    def authorized_call(turn_id, operation, arguments, request_id, message=""):
        plan = env["request_daxview_data_plan"](turn_id, operation, arguments, request_id)
        return env["call_authorized_historical_tool"](
            operation, plan.get("authorization_id", "authorized"), arguments, request_id
        )

    def authorized_execute(turn_id, operation, arguments, request_id, message=""):
        return [{"operation_id": operation, "arguments": arguments,
                 "result": authorized_call(turn_id, operation, arguments, request_id, message)}]

    env["call_authorized_internal_operation"] = authorized_call
    env["execute_authorized_internal_operation"] = authorized_execute
    return env


class EmsPredictionTests(unittest.TestCase):
    def setUp(self):
        self.env = load_functions()

    def test_alarm_ranking_routes_only_to_alarm_frequency(self):
        question = "What alarm types occurred most often at this site in the last 7 days? Rank the top five."
        self.assertEqual(self.env["select_historical_operations"](question), ["alarm_frequency_summary"])
        self.assertEqual(self.env["requested_top_limit"](question), 5)

    def test_connected_load_question_calculates_locally_without_historical_data(self):
        question = (
            "Calculate the estimated maximum demand for a standard domestic single-phase installation with the following connected loads:\n"
            "Lighting: 12 lighting points (60W each)\n"
            "Socket outlets (10A): 15 single/double points\n"
            "Electric cooker: 6.9 kW (at 230V)\n"
            "Instantaneous water heater: 4.0 kW (at 230V)\n"
            "Air conditioner: 2.5 kW (at 230V)"
        )
        result = self.env["connected_load_calculation_answer"](question)
        self.assertEqual([row["name"] for row in result["loads"]], [
            "Lighting", "Electric cooker", "Instantaneous water heater", "Air conditioner", "Socket outlets",
        ])
        self.assertAlmostEqual(result["total_kw"], 48.62)
        self.assertIn("211.39 A", result["reply"])
        self.assertIn("not a code-diversified maximum-demand estimate", result["reply"])
        self.assertIn("no DaxView/MCP data used", result["reply"])

    def test_supplied_interval_demand_readings_and_cap_are_calculated_locally(self):
        question = (
            "A device recorded these 15-minute average power readings today: "
            "312 kW, 348 kW, 421 kW, 389 kW, and 405 kW. What was the maximum demand, "
            "when did it occur if these readings are chronological, and how much must demand "
            "be reduced to meet a 400 kW cap?"
        )
        result = self.env["supplied_demand_readings_answer"](question)
        self.assertEqual(result["peak_kw"], 421)
        self.assertEqual(result["peak_reading_index"], 3)
        self.assertIn("21.00 kW (4.99% of the peak)", result["reply"])

    def test_explicit_inclusive_date_range_maps_to_local_midnight_bounds(self):
        window = self.env["requested_historical_range_impl"](
            "Show site energy from 1–8 October 2026 in Asia/Kuala_Lumpur"
        )
        self.assertEqual(window["start"], "2026-09-30T16:00:00+00:00")
        self.assertEqual(window["end"], "2026-10-08T16:00:00+00:00")
        self.assertEqual(
            self.env["requested_explicit_date_range"]("from 1 to 8 October 2026", 2026),
            (date(2026, 10, 1), date(2026, 10, 8)),
        )

    def test_alarm_lists_never_route_to_device_inventory(self):
        self.assertEqual(
            self.env["select_historical_operations"](
                "List up to 10 active or acknowledged alarms at site 33, showing severity and device"
            ),
            ["active_alarm_summary"],
        )
        self.assertEqual(
            self.env["select_historical_operations"](
                "List the latest alarm occurrences at site 33"
            ),
            ["alarm_frequency_summary"],
        )

    def test_top_voltage_without_device_requests_metric_discovery_not_inventory(self):
        self.assertEqual(
            self.env["select_historical_operations"]("List the top 5 devices by highest voltage for 7 days"),
            ["telemetry_metric_catalog"],
        )

    def test_follow_up_reuses_previous_device_even_when_memory_flag_is_false(self):
        self.env["previous_daxview_turn"] = Mock(return_value={
            "id": "previous",
            "user_message": "Show demand for device ID 614 for the last 7 days",
            "assistant_reply": "Peak demand was 42 kW",
            "context": {"site_id": 33, "device_id": 614},
            "resolved_plan": {"status": "ok", "slots": {"site_id": 33, "device_id": 614}},
        })
        self.env["log_event"] = Mock()
        resolved, context = self.env["resolve_follow_up_message"](
            "turn", "What was its max demand for 3 days?", {"site_id": 33}, "same-conversation", "request"
        )
        self.assertEqual(resolved, "What was its max demand for 3 days?")
        self.assertEqual(context["device_id"], 614)

    def test_demand_questions_without_explicit_connected_loads_are_not_calculated(self):
        self.assertIsNone(self.env["connected_load_calculation_answer"]("What was the site's maximum demand last week?"))

    def test_original_turn_short_circuits_planner_and_mcp_for_supplied_load_math(self):
        question = (
            "Estimate maximum demand for connected loads: Lighting 10 points (50W each); "
            "Water heater 3 kW at 230V"
        )
        self.env["resolve_follow_up_message"] = lambda turn_id, message, context, session_id, request_id: (message, context)
        self.env["choose_historical_operations"] = Mock(side_effect=AssertionError("planner should not run"))
        result = self.env["run_daxview_integration_turn"]("turn", question, {}, "request", "session")
        self.assertEqual(result["provider"], "local-ems-calculation")
        self.assertIn("3.50 kW", result["reply"])
        self.assertFalse(any("mcp" in call.args[0] for call in self.env["debug_trace_event"].call_args_list))

    def test_planner_public_tool_intents_resolve_to_correct_internal_operation(self):
        intents = self.env["planner_default_intents"]
        self.assertEqual(intents("List out 5 latest alarm", ["alarm_data"]), ["alarm_frequency_summary"])
        self.assertEqual(intents("How many active alarms are there?", ["alarm_data"]), ["active_alarm_summary"])
        self.assertEqual(intents("Show details for alarm event ID 901", ["alarm_data"]), ["alarm_detail_lookup"])
        self.assertEqual(intents("List available voltage metrics", ["device_data"]), ["telemetry_metric_catalog"])
        self.assertEqual(intents("Show voltage for device 519", ["device_data"]), ["telemetry_timeseries"])
        self.assertEqual(intents("Show site energy for this week", ["site_summary"]), ["site_energy_summary"])

    def test_clear_regex_intent_is_not_blocked_by_planner_clarification(self):
        from types import SimpleNamespace
        self.env["AI_PLANNER_ENABLED"] = True
        self.env["AI_PLANNER_SHADOW_MODE"] = False
        self.env["plan_turn_with_llm"] = Mock(return_value=(SimpleNamespace(
            tools=["alarm_data"],
            slots=SimpleNamespace(limit=None, metric=None, building_ref=None, alarm_id=None, device_refs=[]),
            clarification=SimpleNamespace(question="Which site do you mean?"),
            confidence=0.9,
        ), 10))
        operations, _, _ = self.env["choose_historical_operations"](
            "List out 5 latest alarm", {"site_id": 17}, "turn", "session", "request"
        )
        self.assertEqual(operations, ["alarm_frequency_summary"])

    def test_iso_timestamps_are_rendered_in_readable_site_local_time(self):
        raw = "Alarm latest: 2026-09-30T19:53:14.552817+00:00."
        self.assertEqual(
            self.env["format_answer_timestamps"](raw, "Asia/Kuala_Lumpur"),
            "Alarm latest: 01 Oct 2026, 3:53 AM MYT.",
        )
        self.assertEqual(
            self.env["format_display_timestamp"]("2026-10-02"),
            "2026-10-02",
        )
        self.env["AI_REFINE_MCP_WITH_MODEL"] = False
        self.assertEqual(
            self.env["refine_historical_answer_with_model"]("q", [], raw, "req"),
            "Alarm latest: 01 Oct 2026, 3:53 AM MYT.",
        )

    def test_latest_alarm_request_routes_and_honors_requested_count(self):
        question = "List out 5 latest alarm"
        self.assertEqual(self.env["select_historical_operations"](question), ["alarm_frequency_summary"])
        self.assertEqual(self.env["requested_top_limit"](question), 5)
        self.env["requested_historical_range"] = Mock(return_value={
            "start": "2026-10-05T00:00:00+00:00",
            "end": "2026-10-12T00:00:00+00:00",
            "timezone": "Asia/Kuala_Lumpur",
        })
        args = self.env["build_historical_arguments"]("alarm_frequency_summary", {"site_id": 17}, question)
        self.assertEqual(args["limit"], 5)
        self.assertEqual(args["start"], "2026-10-05T00:00:00+00:00")
        self.assertEqual(args["end"], "2026-10-12T00:00:00+00:00")

    def test_latest_alarm_rows_sort_by_timestamp_not_frequency(self):
        answer = self.env["summarize_alarm_frequency"]({"rows": [
            {"alarm_name": "Frequent but older", "event_count": 9, "last_seen": "2026-10-08T00:00:00Z"},
            {"alarm_name": "Recent", "event_count": 1, "last_seen": "2026-10-11T00:00:00Z"},
        ]}, {"limit": 2}, "List out 5 latest alarm")
        self.assertLess(answer.index("Recent"), answer.index("Frequent but older"))
        self.assertIn("Latest alarm occurrences", answer)

    def test_combined_daily_energy_question_uses_site_summary(self):
        question = "Show this site's daily energy use for the last 7 days and what was the highest and lowest reading?"
        self.assertEqual(self.env["select_historical_operations"](question), ["site_energy_summary"])
        self.assertEqual(self.env["requested_energy_extrema"](question), (True, True))
        self.assertEqual(len(self.env["requested_question_parts"](question)), 2)

    def test_device_energy_question_uses_timeseries_not_site_summary(self):
        question = "What is the energy consumption for device ID 519 for the last 7 days? Show value and unit."
        self.assertEqual(self.env["select_historical_operations"](question), ["telemetry_timeseries"])
        self.env["requested_historical_range"] = Mock(return_value={
            "start": "2026-10-01T00:00:00+00:00",
            "end": "2026-10-08T00:00:00+00:00",
            "timezone": "Asia/Kuala_Lumpur",
        })
        arguments = self.env["build_historical_arguments"]("telemetry_timeseries", {"site_id": 17}, question)
        self.assertEqual(arguments["device_id"], 519)
        self.assertEqual(arguments["metric"], "energy")
        self.assertEqual(arguments["phase"], "all")
        self.assertNotIn("value_mode", arguments)

    def test_device_ranking_uses_consumers_and_status_not_site_daily_totals(self):
        question = "Which devices contributed most to energy use during that period? Show the top five and tell me which are offline."
        self.assertEqual(
            self.env["select_historical_operations"](question),
            ["device_energy_breakdown", "site_device_list"],
        )
        all_question = "List out all the devices and rank them based on usage"
        self.assertEqual(
            self.env["select_historical_operations"](all_question),
            ["device_energy_breakdown", "site_device_list"],
        )
        self.env["requested_historical_range"] = Mock(return_value={"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"})
        args = self.env["build_historical_arguments"]("device_energy_breakdown", {"site_id": 17}, all_question)
        self.assertEqual(args["limit"], 100)
        self.assertEqual(args["start_time"], "start")
        self.assertEqual(args["end_time"], "end")

    def test_top_lowest_consumer_question_includes_sum(self):
        question = "What is the top energy consume device and the lowest and what are the total sum of their 2 energy"
        self.assertTrue(self.env["wants_lowest_consumer"](question))
        self.assertTrue(self.env["wants_top_lowest_sum"](question))
        answer = self.env["summarize_top_consumers"](
            {
                "rows": [
                    {"device_name": "Top Meter", "total_kwh": 100},
                    {"device_name": "Low Meter", "total_kwh": 25},
                    {"device_name": "Middle Meter", "total_kwh": 50},
                ],
                "unit": "kWh",
            },
            {"start": "2026-10-01T00:00:00+00:00", "end": "2026-10-08T00:00:00+00:00"},
            question,
        )
        self.assertIn("Highest energy-consuming device: Top Meter at 100.00 kWh", answer)
        self.assertIn("Lowest energy-consuming device: Low Meter at 25.00 kWh", answer)
        self.assertIn("Combined total of those two devices: 125.00 kWh", answer)

    def test_invalid_metric_error_is_readable(self):
        answer = self.env["readable_historical_errors"]([
            {"operation_id": "telemetry_timeseries", "metric": "current", "error": "telemetry_timeseries could not return valid data: INVALID_METRIC"}
        ])
        self.assertIn("current", answer)
        self.assertIn("metric name is not supported", answer)

    def test_device_inventory_routes_to_site_device_list(self):
        question = "List me out devices that is in this site"
        self.assertTrue(self.env["wants_device_inventory"](question))
        self.assertEqual(self.env["select_historical_operations"](question), ["site_device_list"])

    def test_voltage_current_testing_discovers_devices_first(self):
        question = "List devices at this site that have data, then tell me which one is best to use for voltage/current trend testing."
        self.assertTrue(self.env["wants_device_capability_discovery"](question))
        self.assertEqual(self.env["select_historical_operations"](question), ["site_device_list"])

    def test_device_summary_surfaces_returned_parameters(self):
        answer = self.env["summarize_site_devices"]({
            "devices": [{
                "device_id": 519,
                "device_name": "AC kWh",
                "status": "online",
                "device_type": "Virtual",
                "has_data": True,
                "supported_metrics": ["energy", "voltage", "current"],
            }]
        })
        self.assertIn("AC kWh", answer)
        self.assertIn("Parameters: energy, voltage, current", answer)

    def test_follow_up_infers_device_from_previous_ranked_answer(self):
        self.env["AI_CHAT_MEMORY_ENABLED"] = True
        self.env["previous_daxview_turn"] = Mock(return_value={
            "id": "previous",
            "user_message": "What is the top 5 highest consumption device for 3 days",
            "assistant_reply": "1. UMG 604-PRO (ID 604): 8973282.19 kWh\n2. UMG 96S (ID 96): 878747.59 kWh",
            "context": {"site_id": 17, "_time_window": {"start": "start", "end": "end", "timezone": "Asia/Kuala_Lumpur"}},
            "resolved_plan": {"status": "ok", "slots": {"site_id": 17}},
        })
        self.env["log_event"] = Mock()
        message, context = self.env["resolve_follow_up_message"](
            "current",
            "Can u continue to get me other parameter that is available for this device",
            {"site_id": 17},
            "conversation",
            "request",
        )
        self.assertEqual(context["device_id"], 604)
        self.assertEqual(context["metric"], "energy")
        self.assertIn("data_availability_summary", self.env["select_historical_operations"](message))

    def test_comparison_answer_never_injected_into_user_answer(self):
        self.env["AI_COMPARE_MODEL_ENABLED"] = True
        self.env["AI_COMPARE_MODEL_SHOW_TO_USER"] = True
        self.env["build_mcp_refine_prompt"] = Mock(return_value="prompt")
        self.env["run_mcp_refine_model"] = Mock(side_effect=["Primary only", "Alternative text"])
        self.env["ensure_historical_answer_coverage"] = Mock(side_effect=lambda message, results, deterministic, answer, request: answer)
        self.env["log_event"] = Mock()
        self.env["debug_trace_event"] = Mock()
        answer = self.env["refine_historical_answer_with_model"]("q", [], "fallback", "request")
        self.assertEqual(answer, "Primary only")
        self.assertNotIn("Alternative", answer)

    def test_generic_model_reply_falls_back_to_mcp_draft(self):
        self.env["log_event"] = Mock()
        deterministic = "Site devices returned by DaxView MCP:\n1. AC kWh (ID 519): online"
        model_answer = "It seems like you might be testing the system or looking for a response to an empty input."
        answer = self.env["ensure_historical_answer_coverage"](
            "List all devices",
            [{"operation_id": "site_device_list", "result": {"structuredContent": {"data": {"devices": []}}}}],
            deterministic,
            model_answer,
            "request",
        )
        self.assertEqual(answer, deterministic)

    def test_clarification_style_llm_reply_cannot_replace_alarm_data_answer(self):
        deterministic = (
            "Alarm frequency summary returned by Daxview MCP:\n"
            "1. Device Offline - 3 occurrence(s); device UMG 512 (ID 379); 3 resolved"
        )
        model_answer = (
            "It seems you've sent a command or prompt, but there's no specific question provided. "
            "Could you clarify what you'd like to ask? I'm here to help!"
        )
        self.env["log_event"] = Mock()
        answer = self.env["ensure_historical_answer_coverage"](
            "Which alarm occurred most often?",
            [{"operation_id": "alarm_frequency_summary", "result": {"structuredContent": {"data": {"rows": []}}}}],
            deterministic,
            model_answer,
            "request",
        )
        self.assertEqual(answer, deterministic)
        self.env["log_event"].assert_called_once()

    def test_alarm_frequency_summary_includes_device_and_lifecycle_details(self):
        answer = self.env["summarize_alarm_frequency"]({"rows": [
            {"alarm_name": "Device Offline", "device_id": 379, "device_name": "UMG 512",
             "event_count": 3, "first_seen": "2026-10-01T00:00:00Z", "last_seen": "2026-10-02T00:00:00Z",
             "active_count": 1, "resolved_count": 2},
        ], "row_count": 1}, {"limit": 5})
        self.assertIn("3 occurrence(s)", answer)
        self.assertIn("UMG 512 (ID 379)", answer)
        self.assertIn("first 2026-10-01T00:00:00Z", answer)
        self.assertIn("1 active, 2 resolved", answer)

    def test_debug_table_uses_verified_alarm_rows_and_latest_order(self):
        tables = self.env["build_tables_from_historical_results"]([{
            "operation_id": "alarm_frequency_summary",
            "public_tool": "alarm_data",
            "operation": "frequency",
            "arguments": {"start_time": "start", "end_time": "end", "timezone": "UTC"},
            "result": {"structuredContent": {"data": {"rows": [
                {"alarm_name": "Older", "device_name": "Meter A", "device_id": 10, "event_count": 8, "last_seen": "2026-10-08T00:00:00Z"},
                {"alarm_name": "Newer", "device_name": "Meter B", "device_id": 11, "event_count": 1, "last_seen": "2026-10-11T00:00:00Z"},
            ], "row_count": 2, "truncated": False}}},
        }], "List out latest alarms")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["title"], "Latest Alarm Occurrences")
        self.assertEqual(tables[0]["source_tool"], "alarm_data.frequency")
        self.assertEqual(tables[0]["rows"][0]["alarm"], "Newer")
        self.assertEqual(tables[0]["rows"][0]["device"], "Meter B (ID 11)")
        self.assertEqual(tables[0]["time_range"]["timezone"], "UTC")

    def test_debug_alarm_table_falls_back_to_populated_aliases(self):
        tables = self.env["build_tables_from_historical_results"]([{
            "operation_id": "alarm_frequency_summary",
            "arguments": {"timezone": "Asia/Kuala_Lumpur"},
            "result": {"data": {"rows": [{
                "alarm_name": None,
                "alarm_type": "High Voltage",
                "device_name": None,
                "meter_name": "Main Meter",
                "deviceId": 519,
                "event_count": None,
                "occurrence_count": 2,
                "last_seen_at": "2026-10-11T03:30:00Z",
                "resolved_count": None,
                "resolved": 2,
            }]}}
        }], "Show latest alarm")
        row = tables[0]["rows"][0]
        self.assertEqual(row["alarm"], "High Voltage")
        self.assertEqual(row["device"], "Main Meter (ID 519)")
        self.assertEqual(row["occurrences"], 2)
        self.assertEqual(row["resolved"], 2)
        self.assertIn("11 Oct 2026", row["latest"])

    def test_alarm_table_matches_alarm_frequency_response_fields_and_leaves_missing_blank(self):
        tables = self.env["build_tables_from_historical_results"]([{
            "operation_id": "alarm_frequency_summary",
            "arguments": {"timezone": "Asia/Kuala_Lumpur", "limit": 1},
            "result": {"structuredContent": {"data": {"rows": [{
                "alarm_id": 2026093000140,
                "alarm_name": "Device Offline",
                "alarm_type": "Device Offline",
                "device_id": 492,
                "device_name": "MhsOffice_DB_LP_2",
                "event_count": 1,
                "first_seen": "2026-10-06T04:03:00+00:00",
                "last_seen": "2026-10-06T04:03:00+00:00",
                "active_count": 0,
                "acknowledged_count": 0,
                "resolved_count": 1,
                "closed_count": 0,
            }], "row_count": 1, "truncated": False}}},
        }], "List latest alarm")
        table = tables[0]
        row = table["rows"][0]
        self.assertEqual([column["key"] for column in table["columns"]], [
            "rank", "alarm", "device", "severity", "first_seen", "latest", "occurrences",
            "active", "acknowledged", "resolved", "closed", "alarm_id",
        ])
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["alarm"], "Device Offline")
        self.assertEqual(row["device"], "MhsOffice_DB_LP_2 (ID 492)")
        self.assertEqual(row["severity"], "")
        self.assertEqual(row["active"], 0)
        self.assertEqual(row["resolved"], 1)
        self.assertEqual(row["alarm_id"], 2026093000140)

    def test_ranking_sorts_numeric_usage_and_joins_current_status(self):
        self.env["historical_result_data"] = lambda result: result["data"]
        self.env["first_value"] = lambda row, keys: next((row[key] for key in keys if row.get(key) is not None), None)
        self.env["format_time_window"] = lambda data, arguments: "last 7 days"
        self.env["format_number"] = lambda value, precision=2: f"{value:.2f}"
        self.env["reading_detail"] = lambda *args: None
        results = [
            {"operation_id": "device_energy_breakdown", "arguments": {"limit": 5}, "result": {"data": {
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

    def test_top_consumers_falls_back_to_device_energy_breakdown(self):
        self.env["resolve_follow_up_message"] = Mock(side_effect=lambda turn, message, context, conversation, request: (message, context))
        self.env["request_daxview_data_plan"] = Mock(return_value={
            "authorization_id": "breakdown-auth",
            "arguments": {"site_id": 17, "start_time": "start", "end_time": "end", "timezone": "Asia/Kuala_Lumpur", "group_by": "device", "limit": 5},
        })
        self.env["call_authorized_historical_tool"] = Mock(return_value=
            {"data": {"rows": [
                {"device_id": 1, "device_name": "High meter", "energy_kwh": 30},
                {"device_id": 2, "device_name": "Low meter", "energy_kwh": 12},
            ], "unit": "kWh"}},
        )
        self.env["format_time_window"] = lambda data, arguments: "last 3 days"
        self.env["format_number"] = lambda value, precision=2: f"{value:.2f}"
        self.env["reading_detail"] = lambda *args: None
        self.env["refine_historical_answer_with_model"] = Mock(side_effect=lambda message, results, draft, request: draft)
        self.env["build_charts_from_historical_results"] = Mock(return_value=[])
        self.env["debug_trace_event"] = Mock()
        self.env["save_resolved_turn_context"] = Mock()
        result = self.env["run_daxview_integration_turn"](
            "turn", "What is the top 5 highest consumption device for 3 days", {"site_id": 17}, "request", "conversation"
        )
        self.assertIn("High meter", result["reply"])
        self.assertIn("30.00 kWh", result["reply"])
        self.assertEqual(result["resolved_plan"]["status"], "ok")
        self.assertEqual(result["resolved_plan"]["tools"][0]["tool"], "device_energy_breakdown")
        self.env["call_authorized_historical_tool"].assert_any_call(
            "device_energy_breakdown",
            "breakdown-auth",
            {"site_id": 17, "start_time": "start", "end_time": "end", "timezone": "Asia/Kuala_Lumpur", "group_by": "device", "limit": 5},
            "request",
        )

    def test_top_consumers_can_disable_fallback_for_raw_mcp_testing(self):
        self.env["AI_MCP_FALLBACKS_ENABLED"] = False
        self.env["resolve_follow_up_message"] = Mock(side_effect=lambda turn, message, context, conversation, request: (message, context))
        self.env["request_daxview_data_plan"] = Mock(return_value={
            "authorization_id": "top-auth",
            "arguments": {"site_id": 17, "start_time": "start", "end_time": "end", "timezone": "Asia/Kuala_Lumpur", "metric": "energy", "limit": 5},
        })
        self.env["call_authorized_historical_tool"] = Mock(
            side_effect=RuntimeError("device_energy_ranking could not return valid data: BACKEND_UNAVAILABLE")
        )
        self.env["log_event"] = Mock()
        self.env["debug_trace_event"] = Mock()
        self.env["save_resolved_turn_context"] = Mock()
        result = self.env["run_daxview_integration_turn"](
            "turn", "List out top 5 highest energy consumption device and it value", {"site_id": 17}, "request", "conversation"
        )
        self.assertIn("device_energy_breakdown is currently unavailable", result["reply"])
        self.env["request_daxview_data_plan"].assert_called_once()

    def test_generic_summary_prioritizes_values_before_metadata_overflow(self):
        answer = self.env["summarize_generic_tool"]("latest_telemetry_snapshot", {
            "readings": [{
                "site_id": 17,
                "building_id": 11,
                "device_id": 410,
                "device_name": "UMG Meter",
                "metric": "voltage",
                "canonical_metric": "voltage",
                "source_metric": "Voltage L1",
                "timestamp": "2026-10-07T06:00:00+00:00",
                "value": 241.2,
                "unit": "V",
            }]
        })
        self.assertIn("value=241.2", answer)
        self.assertIn("unit=V", answer)
        self.assertIn("timestamp=2026-10-07T06:00:00+00:00", answer)

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

    def test_explicit_mcp_tool_request_uses_only_named_tool(self):
        question = "Test MCP tool device_data metrics for this site and show devices and metrics"
        self.assertEqual(self.env["explicit_mcp_tool_request"](question), "telemetry_metric_catalog")
        self.assertEqual(self.env["select_historical_operations"](question), ["telemetry_metric_catalog"])

    def test_device_energy_breakdown_uses_new_time_scope(self):
        self.env["requested_historical_range"] = Mock(return_value={
            "start": "2026-10-01T00:00:00+00:00",
            "end": "2026-10-02T00:00:00+00:00",
            "timezone": "Asia/Kuala_Lumpur",
        })
        arguments = self.env["build_historical_arguments"](
            "device_energy_breakdown",
            {"site_id": 17},
            "Break down this site's energy by device for last 7 days",
        )
        self.assertIn("start_time", arguments)
        self.assertIn("end_time", arguments)
        self.assertNotIn("start", arguments)
        self.assertNotIn("end", arguments)

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

    def test_site_wide_demand_scan_uses_configured_power_and_excludes_current_only_devices(self):
        question = "Find the highest maximum demand across all devices at site 17 for the last 7 days"
        self.assertEqual(self.env["select_historical_operations"](question), ["demand_peak_summary"])
        self.env["build_historical_arguments"] = lambda operation, context, message="": {
            "site_id": context["site_id"],
            **({"device_id": context["device_id"]} if context.get("device_id") else {}),
            **{key: context[key] for key in ("metric", "bucket", "aggregation", "value_mode", "limit") if key in context},
            "start_time": "2026-10-05T00:00:00+08:00",
            "end_time": "2026-10-12T00:00:00+08:00",
            "timezone": "Asia/Kuala_Lumpur",
            "limit": 100,
        }

        def call(turn_id, operation, arguments, request_id, message=""):
            if operation == "site_device_list":
                return {"data": {"devices": [
                    {"device_id": 519, "device_name": "Main Meter", "status": "online"},
                    {"device_id": 520, "device_name": "Current-only Meter", "status": "online"},
                    {"device_id": 521, "device_name": "Energy-only Meter", "status": "online"},
                ], "row_count": 3, "truncated": False}}
            if operation == "telemetry_metric_catalog" and arguments["device_id"] == 519:
                return {"data": {"metrics": [{"metric": "active_power", "source_metric": "Active Power Total", "unit": "kW", "status": "OK", "enabled": True}]}}
            if operation == "telemetry_metric_catalog":
                if arguments["device_id"] == 521:
                    return {"data": {"metrics": [{"metric": "energy", "source_metric": "Active Energy Consumed", "unit": "kWh", "status": "OK", "enabled": True}]}}
                return {"data": {"metrics": [{"metric": "current", "source_metric": "Current L1", "unit": "A", "status": "OK", "enabled": True}]}}
            if operation == "telemetry_timeseries":
                if arguments["device_id"] == 521:
                    return {"structuredContent": {"data": {"series": [{
                        "metric": "energy", "unit": "kWh", "value_mode": "consumption_delta", "points": [
                            {"timestamp": "2026-10-11T10:00:00+08:00", "value": 2.4},
                        ],
                    }]}}}
                return {"structuredContent": {"data": {"series": [{
                    "metric": "active_power", "unit": "kW", "points": [
                        {"timestamp": "2026-10-11T09:00:00+08:00", "value": 42.5},
                        {"timestamp": "2026-10-11T10:00:00+08:00", "value": 48.2},
                    ],
                }]}}}
            raise AssertionError(f"Unexpected operation: {operation}")

        self.env["call_authorized_internal_operation"] = call
        self.env["debug_trace_event"] = Mock()
        results = self.env["run_highest_demand_device_ranking"](
            "turn", question, {"site_id": 17, "_time_window": {
                "start": "2026-10-05T00:00:00+08:00",
                "end": "2026-10-12T00:00:00+08:00",
                "timezone": "Asia/Kuala_Lumpur",
            }}, "request"
        )
        data = results[0]["result"]["structuredContent"]["data"]
        self.assertEqual(data["ranked_devices"][0]["device_id"], 519)
        self.assertEqual(data["ranked_devices"][0]["peak_kw"], 48.2)
        self.assertEqual(data["ranked_devices"][0]["peak_time"], "11 Oct 2026, 10:00 AM MYT")
        energy_row = next(row for row in data["ranked_devices"] if row["device_id"] == 521)
        self.assertEqual(energy_row["peak_kw"], 9.6)
        self.assertEqual(energy_row["metric"], "energy")
        self.assertEqual(data["excluded_devices"][0]["device_id"], 520)
        self.assertIn("No configured demand", data["excluded_devices"][0]["reason"])

    def test_interval_energy_delta_converts_to_average_kw(self):
        result = self.env["calculate_demand_peak_from_energy"](
            {"structuredContent": {"data": {"series": [{
                "unit": "kWh", "value_mode": "consumption_delta", "points": [
                    {"timestamp": "2026-10-11T10:00:00+08:00", "value": 1.2},
                    {"timestamp": "2026-10-11T10:15:00+08:00", "value": 2.4},
                ],
            }]}}},
            {"bucket": "15m", "value_mode": "consumption_delta", "device_id": 521},
        )
        self.assertEqual(result["data"]["peak_kw"], 9.6)
        self.assertEqual(result["data"]["peak_time"], "2026-10-11T10:15:00+08:00")

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

    def test_demand_no_data_triggers_manual_fallback(self):
        self.assertTrue(self.env["demand_peak_needs_manual_fallback"]({
            "structuredContent": {"status": "error", "error_code": "NO_DATA"}
        }))
        self.assertFalse(self.env["demand_peak_needs_manual_fallback"]({
            "structuredContent": {"data": {"peak_kw": 42}}
        }))

    def test_manual_demand_fallback_calculates_peak_from_telemetry(self):
        result = self.env["calculate_demand_peak_from_telemetry"](
            {"structuredContent": {"data": {"rows": [
                {"timestamp": "2026-10-01T08:00:00+08:00", "value": 12.5},
                {"timestamp": "2026-10-01T09:00:00+08:00", "value": 44.25},
                {"timestamp": "2026-10-01T10:00:00+08:00", "value": 18.0},
            ]}}},
            {"site_id": 17, "start_time": "start", "end_time": "end"},
        )
        self.assertEqual(result["data"]["peak_kw"], 44.25)
        self.assertEqual(result["data"]["peak_time"], "2026-10-01T09:00:00+08:00")
        self.assertEqual(result["data"]["sample_count"], 3)
        self.assertEqual(result["data"]["source"], "deterministic maximum of device_data.values demand points")

    def test_demand_peak_reads_series_points_and_converts_watts_to_kw(self):
        result = self.env["calculate_demand_peak_from_telemetry"](
            {"structuredContent": {"data": {"series": [{
                "metric": "demand", "unit": "W", "points": [
                    {"timestamp": "2026-10-11T08:00:00Z", "value": 240800},
                    {"timestamp": "2026-10-11T09:00:00Z", "value": 428600},
                    {"timestamp": "2026-10-11T10:00:00Z", "value": 390000},
                ],
            }]}}},
            {"device_id": 519, "site_id": 17},
        )
        self.assertEqual(result["data"]["peak_kw"], 428.6)
        self.assertEqual(result["data"]["peak_time"], "2026-10-11T09:00:00Z")
        self.assertEqual(result["data"]["unit"], "kW")

    def test_standards_are_not_added_to_routine_site_answer(self):
        self.assertEqual(self.env["build_compliance_context"]("Show site energy for the last 7 days", ["site_energy_summary"]), [])
        self.assertTrue(self.env["build_compliance_context"]("How does ISO 50001 use this energy data?", ["site_energy_summary"]))
        self.assertFalse(self.env["needs_ems_library"]("Rank the top alarm types this week"))


if __name__ == "__main__":
    unittest.main()
