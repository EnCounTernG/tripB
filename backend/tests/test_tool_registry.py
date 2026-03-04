import unittest

from app.services.tool_registry import ToolExecutionError, ToolRegistry


class ToolRegistryFallbackTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_rate_limit_fallback(self):
        self.registry._specs["map.search_poi"].handler = lambda args: (_ for _ in ()).throw(ToolExecutionError("rate_limit", "429"))
        result = self.registry.execute_with_fallback("map.search_poi", {"city": "北京", "keywords": "景点"})
        self.assertTrue(result["degraded"])
        self.assertEqual(result["error_category"], "rate_limit")

    def test_timeout_fallback(self):
        self.registry._specs["map.weather"].handler = lambda args: (_ for _ in ()).throw(ToolExecutionError("timeout", "timeout"))
        result = self.registry.execute_with_fallback("map.weather", {"city": "北京"})
        self.assertTrue(result["degraded"])
        self.assertEqual(result["error_category"], "timeout")

    def test_bad_request_fallback(self):
        self.registry._specs["map.route"].handler = lambda args: (_ for _ in ()).throw(ToolExecutionError("bad_request", "bad req"))
        result = self.registry.execute_with_fallback(
            "map.route",
            {"city": "北京", "origin": "a", "destination": "b", "mode": "driving"},
        )
        self.assertTrue(result["degraded"])
        self.assertEqual(result["error_category"], "bad_request")


if __name__ == "__main__":
    unittest.main()
