import unittest

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripRequest


class DataDrivenFallbackTests(unittest.TestCase):
    def setUp(self):
        self.planner = LangGraphTripPlanner.__new__(LangGraphTripPlanner)

        class _RegistryStub:
            def execute_with_fallback(self, tool_name, action_input):
                return {
                    "ok": True,
                    "degraded": False,
                    "data": {
                        "pois": [
                            {
                                "name": "将军山滑雪场",
                                "address": "阿勒泰市将军山路",
                                "location": "88.137,47.848",
                                "type": "风景名胜",
                            }
                        ]
                    },
                }

        self.planner.tool_registry = _RegistryStub()
        self.request = TripRequest(
            city="阿勒泰",
            start_date="2026-03-08",
            end_date="2026-03-10",
            travel_days=3,
            transportation="混合",
            accommodation="经济型酒店",
            preferences=["自然风光"],
            free_text_input="",
            runtime="langgraph",
        )

    def test_data_driven_plan_prefers_real_poi(self):
        gathered = {
            "poi": {
                "data": {
                    "pois": [
                        {
                            "name": "喀纳斯景区",
                            "address": "布尔津县喀纳斯",
                            "location": "87.019,48.708",
                            "type": "风景名胜",
                        }
                    ]
                }
            },
            "route": {"data": {"distance": "1000", "duration": "600"}},
            "weather": {
                "data": {
                    "forecasts": [
                        {
                            "casts": [
                                {
                                    "date": "2026-03-08",
                                    "dayweather": "晴",
                                    "nightweather": "多云",
                                    "daytemp": "1",
                                    "nighttemp": "-8",
                                    "daywind": "南",
                                    "daypower": "1-3",
                                }
                            ]
                        }
                    ]
                }
            },
        }
        plan = self.planner._create_data_driven_plan(self.request, gathered, "测试")
        self.assertEqual(plan.days[0].attractions[0].name, "喀纳斯景区")
        self.assertIn("工具真实结果", plan.overall_suggestions)


if __name__ == "__main__":
    unittest.main()
