import unittest

from contracts.mcp_manifest import (
    PUBLIC_MCP_TOOLS,
    build_public_calls,
    normalize_mcp_arguments,
    public_tool_for_internal,
)


class MpcManifestTests(unittest.TestCase):
    def test_only_three_public_tools_are_discoverable(self):
        self.assertEqual(PUBLIC_MCP_TOOLS, {"device_data", "alarm_data", "site_summary"})

    def test_internal_intents_map_to_operation_specific_public_tools(self):
        self.assertEqual(public_tool_for_internal("demand_peak_summary"), ("device_data", "values"))
        self.assertEqual(public_tool_for_internal("device_energy_ranking"), ("device_data", "energy_rows"))
        self.assertEqual(public_tool_for_internal("alarm_frequency_summary"), ("alarm_data", "frequency"))
        self.assertEqual(public_tool_for_internal("site_energy_summary"), ("site_summary", "energy"))

    def test_demand_request_uses_maximum_demand_metric_and_exact_public_tool(self):
        calls = build_public_calls("demand_peak_summary", {
            "site_id": 17,
            "device_id": 519,
            "start_time": "2026-10-10T00:00:00+00:00",
            "end_time": "2026-10-11T00:00:00+00:00",
            "timezone": "UTC",
        })
        self.assertEqual(len(calls), 1)
        tool, args = calls[0]
        self.assertEqual(tool, "device_data")
        self.assertEqual(args["operation"], "values")
        self.assertEqual(args["metric"], "demand")
        self.assertEqual(args["aggregation"], "max")
        self.assertEqual(args["value_mode"], "maximum_demand")
        normalized = normalize_mcp_arguments(tool, args)
        self.assertEqual(normalized["device_id"], 519)

    def test_unknown_public_tools_operations_and_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_mcp_arguments("telemetry_timeseries", {"operation": "values"})
        with self.assertRaises(ValueError):
            normalize_mcp_arguments("device_data", {"operation": "unknown", "site_id": 17})
        with self.assertRaises(ValueError):
            normalize_mcp_arguments("device_data", {"operation": "list", "site_id": 17, "sql": "select 1"})

    def test_values_requires_device_metric_and_timezone_qualified_range(self):
        base = {
            "operation": "values", "site_id": 17, "device_id": 519,
            "metric": "voltage", "start_time": "2026-10-10T00:00:00Z",
            "end_time": "2026-10-11T00:00:00Z", "timezone": "UTC",
        }
        self.assertEqual(normalize_mcp_arguments("device_data", base)["bucket"], "1h")
        invalid = {**base, "start_time": "2026-10-10T00:00:00"}
        with self.assertRaises(ValueError):
            normalize_mcp_arguments("device_data", invalid)

    def test_metrics_defaults_to_last_24_hours(self):
        normalized = normalize_mcp_arguments("device_data", {"operation": "metrics", "site_id": 17, "device_id": 519})
        self.assertIn("start_time", normalized)
        self.assertIn("end_time", normalized)
        self.assertEqual(normalized["timezone"], "Asia/Kuala_Lumpur")


if __name__ == "__main__":
    unittest.main()
