import json
import unittest

from contracts.tool_schemas import TOOL_SCHEMAS, validate_tool_arguments
from memory.conversation import truncate_reply
from planner.hybrid_planner import parse_planner_response


class HybridPlannerMemoryTests(unittest.TestCase):
    def test_all_contracts_strip_unknown_arguments(self):
        for tool in TOOL_SCHEMAS:
            with self.subTest(tool=tool):
                args = self.valid_args(tool)
                args["unexpected"] = "bad"
                clean = validate_tool_arguments(tool, args)
                self.assertNotIn("unexpected", clean)

    def test_energy_forecast_uses_reference_contract_names(self):
        args = validate_tool_arguments(
            "energy_forecast",
            {
                "site_id": 17,
                "forecast_start": "2026-10-06T00:00:00+00:00",
                "forecast_end": "2026-10-13T00:00:00+00:00",
                "training_days": 35,
                "timezone": "Asia/Kuala_Lumpur",
            },
        )
        self.assertIn("forecast_start", args)
        self.assertNotIn("start", args)
        self.assertNotIn("forecast_days", args)

    def test_planner_rejects_non_allowlisted_tool(self):
        payload = {
            "message": {
                "content": json.dumps({
                    "intent": "data_query",
                    "is_follow_up": False,
                    "tools": ["drop_database"],
                    "slots": {},
                    "inherit_from_previous": [],
                    "clarification": None,
                    "confidence": 0.9,
                })
            }
        }
        with self.assertRaises(ValueError):
            parse_planner_response(payload, {"site_energy_summary"})

    def test_memory_truncates_assistant_reply(self):
        reply = "x" * 900
        self.assertEqual(len(truncate_reply(reply)), 800)
        self.assertTrue(truncate_reply(reply).endswith("..."))

    @staticmethod
    def valid_args(tool):
        base = {"site_id": 17}
        window = {**base, "start": "2026-10-01T00:00:00+00:00", "end": "2026-10-02T00:00:00+00:00", "timezone": "Asia/Kuala_Lumpur"}
        time_scope = {**base, "start_time": window["start"], "end_time": window["end"], "timezone": "Asia/Kuala_Lumpur"}
        examples = {
            "telemetry_top_consumers": {**window, "limit": 5},
            "site_energy_summary": {**window, "bucket": "day"},
            "alarm_frequency_summary": {**window, "limit": 5},
            "site_metadata_summary": base,
            "site_device_list": {**base, "limit": 100},
            "telemetry_timeseries": {**time_scope, "device_id": 12, "metric": "voltage", "bucket": "1h", "aggregation": "auto", "limit": 500},
            "active_alarm_summary": {**base, "limit": 50},
            "meter_status_summary": {**base, "limit": 100},
            "energy_comparison_summary": {**base, "period_a_start": window["start"], "period_a_end": window["end"], "period_b_start": "2026-10-02T00:00:00+00:00", "period_b_end": "2026-10-03T00:00:00+00:00", "timezone": "Asia/Kuala_Lumpur"},
            "data_availability_summary": {**time_scope, "metric": "energy"},
            "alarm_detail_lookup": {**base, "alarm_id": 123},
            "power_quality_summary": {**time_scope, "limit": 100},
            "demand_peak_summary": time_scope,
            "tariff_cost_summary": time_scope,
            "device_energy_breakdown": {**window, "limit": 20},
            "energy_forecast": {**base, "forecast_start": "2026-10-06T00:00:00+00:00", "forecast_end": "2026-10-13T00:00:00+00:00", "training_days": 35, "timezone": "Asia/Kuala_Lumpur"},
            "anomaly_detection_summary": {**window, "limit": 20},
            "report_summary": window,
        }
        return dict(examples[tool])


if __name__ == "__main__":
    unittest.main()
