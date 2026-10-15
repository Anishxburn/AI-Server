import ast
import unittest
from pathlib import Path
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NODES = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}


class ClarificationHelperTests(unittest.TestCase):
    def test_explicit_historical_reading_clarification_request_is_detected(self):
        module = ast.Module(body=[NODES["is_historical_readings_clarification_request"]], type_ignores=[])
        env = {"__builtins__": __builtins__, "re": __import__("re")}
        exec(compile(module, str(SOURCE), "exec"), env)
        question = (
            "I need historical device readings at Mun Hean Singapore HQ. "
            "Please ask which metric and period I want before fetching readings."
        )
        self.assertTrue(env["is_historical_readings_clarification_request"](question))

    def test_explicit_clarification_routes_to_telemetry_without_planner(self):
        module = ast.Module(body=[NODES["select_historical_operations"]], type_ignores=[])
        env = {
            "__builtins__": __builtins__,
            "is_historical_readings_clarification_request": lambda _message: True,
        }
        exec(compile(module, str(SOURCE), "exec"), env)
        self.assertEqual(env["select_historical_operations"]("ignored"), ["telemetry_timeseries"])

    def test_persisted_trace_redacts_free_text_and_secret_fields(self):
        module = ast.Module(body=[NODES["mcp_result_shape"], NODES["sanitize_persisted_trace"]], type_ignores=[])
        env = {"__builtins__": __builtins__}
        exec(compile(module, str(SOURCE), "exec"), env)
        result = env["sanitize_persisted_trace"]({
            "event": "planner_response", "message": "private operator message",
            "authorization_id": "secret", "duration_ms": 14,
        })
        self.assertEqual(result["message"], {"redacted": True, "length": 24})
        self.assertEqual(result["authorization_id"], "[redacted]")
        self.assertEqual(result["duration_ms"], 14)

    def test_clarification_extraction_preserves_original_question(self):
        module = ast.Module(body=[NODES["extract_clarification_details"]], type_ignores=[])
        env = {
            "__builtins__": __builtins__, "json": __import__("json"),
            "ollama_json": Mock(return_value={"message": {"content": '{"query":"rewritten","metric":"energy","time_phrase":"last 7 days"}'}}),
            "PLANNER_MODEL": "planner", "PLANNER_TIMEOUT": 1,
            "log_event": Mock(), "question_metrics": lambda _text: [],
            "re": __import__("re"),
            "has_named_device_reference": lambda _text: False,
            "has_time_scope": lambda text: "last 7 days" in text,
        }
        exec(compile(module, str(SOURCE), "exec"), env)
        query, context, _slots = env["extract_clarification_details"](
            "historical readings at HQ", "energy, last 7 days", {}, "request-1"
        )
        self.assertIn("historical readings at HQ", query)
        self.assertNotEqual(query, "rewritten")
        self.assertEqual(context["metric"], "energy")
        self.assertIn("last 7 days", query)

    def test_partial_clarification_keeps_original_question_stable(self):
        module = ast.Module(body=[NODES["extract_clarification_details"]], type_ignores=[])
        env = {
            "__builtins__": __builtins__, "json": __import__("json"),
            "ollama_json": Mock(return_value={"message": {"content": '{"metric":"energy","time_phrase":"last 7 days","device_ref":"Main Meter"}'}}),
            "PLANNER_MODEL": "planner", "PLANNER_TIMEOUT": 1, "log_event": Mock(),
            "re": __import__("re"),
            "question_metrics": lambda _text: [], "has_named_device_reference": lambda _text: False,
            "has_time_scope": lambda text: "last 7 days" in text,
        }
        exec(compile(module, str(SOURCE), "exec"), env)
        original = "I need historical device readings at Mun Hean Singapore HQ."
        query, context, slots = env["extract_clarification_details"](original, "Main Meter, energy, last 7 days", {}, "request-2")
        self.assertEqual(query.count("Additional details"), 0)
        self.assertIn(original, query)
        self.assertEqual(context["metric"], "energy")
        self.assertEqual(context["_clarification_device_ref"], "Main Meter")
        self.assertEqual(slots["time_phrase"], "last 7 days")


if __name__ == "__main__":
    unittest.main()
