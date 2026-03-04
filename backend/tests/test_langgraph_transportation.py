import unittest

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripRequest


class LangGraphTransportationTests(unittest.TestCase):
    def setUp(self):
        self.planner = LangGraphTripPlanner.__new__(LangGraphTripPlanner)
        self.request = TripRequest(
            city="北京",
            start_date="2026-06-01",
            end_date="2026-06-03",
            travel_days=3,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=["历史文化"],
            free_text_input="",
            runtime="langgraph",
        )

    def test_transportation_mode_mapping(self):
        self.assertEqual(self.planner._route_mode_from_transportation("公共交通"), "transit")
        self.assertEqual(self.planner._route_mode_from_transportation("自驾"), "driving")
        self.assertEqual(self.planner._route_mode_from_transportation("步行"), "walking")
        self.assertEqual(self.planner._route_mode_from_transportation("骑行"), "riding")

    def test_finish_will_be_forced_to_missing_tool(self):
        action = {"thought": "够了", "action": "finish", "action_input": {}}
        enforced = self.planner._enforce_react_action_rules(
            action,
            self.request,
            preferred_mode="transit",
            executed_actions={"map.search_poi"},
            executed_route_modes=set(),
        )
        self.assertIn(enforced["action"], {"map.weather", "map.route"})

    def test_route_mode_comes_from_transportation(self):
        action = {"thought": "查路线", "action": "map.route", "action_input": {"origin": "天安门", "destination": "故宫"}}
        enforced = self.planner._enforce_react_action_rules(
            action,
            self.request,
            preferred_mode="transit",
            executed_actions={"map.search_poi", "map.weather"},
            executed_route_modes=set(),
        )
        self.assertEqual(enforced["action_input"]["mode"], "transit")

    def test_mixed_transportation_requires_walking_route(self):
        mixed_request = self.request.model_copy(update={"transportation": "混合"})
        action = {"thought": "结束", "action": "finish", "action_input": {}}
        enforced = self.planner._enforce_react_action_rules(
            action,
            mixed_request,
            preferred_mode="transit",
            executed_actions={"map.search_poi", "map.weather", "map.route"},
            executed_route_modes={"transit"},
        )
        self.assertEqual(enforced["action"], "map.route")
        self.assertEqual(enforced["action_input"]["mode"], "walking")


if __name__ == "__main__":
    unittest.main()
