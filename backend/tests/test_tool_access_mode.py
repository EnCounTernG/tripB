import unittest

from app.agents.langgraph_trip_planner import LangGraphTripPlanner


class ToolAccessModeTests(unittest.TestCase):
    def setUp(self):
        self.planner = LangGraphTripPlanner.__new__(LangGraphTripPlanner)

    def test_run_tool_call_uses_registry_for_mcp(self):
        class _RegistryStub:
            def execute_with_fallback(self, action_name, action_input):
                return {"ok": True, "from": "registry", "action": action_name, "input": action_input}

        self.planner.tool_registry = _RegistryStub()
        self.planner._execute_direct_http_tool = lambda action_name, action_input: {"ok": True, "from": "direct"}

        result = self.planner._run_tool_call("mcp", "map.weather", {"city": "北京"})
        self.assertEqual(result["from"], "registry")

    def test_run_tool_call_uses_direct_http(self):
        class _RegistryStub:
            def execute_with_fallback(self, action_name, action_input):
                return {"ok": True, "from": "registry"}

        self.planner.tool_registry = _RegistryStub()
        self.planner._execute_direct_http_tool = lambda action_name, action_input: {"ok": True, "from": "direct", "action": action_name}

        result = self.planner._run_tool_call("direct_http", "map.search_poi", {"city": "北京", "keywords": "景点"})
        self.assertEqual(result["from"], "direct")
        self.assertEqual(result["action"], "map.search_poi")


if __name__ == "__main__":
    unittest.main()
