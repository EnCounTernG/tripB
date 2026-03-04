"""基于LangGraph的旅行规划运行时（阶段B）。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from ..models.schemas import Attraction, DayPlan, Location, Meal, TripPlan, TripRequest
from ..services.langchain_service import get_langchain_llm
from ..services.tool_registry import ToolRegistry, ToolTrace


class TripPlanningState(TypedDict, total=False):
    request: TripRequest
    gathered_data: Dict[str, Any]
    planner_raw: str
    trip_plan: TripPlan
    runtime: str
    errors: List[str]
    tool_traces: List[Dict[str, Any]]
    degraded_messages: List[str]


class LangGraphTripPlanner:
    """LangGraph + ReAct + MCP工具编排。"""

    def __init__(self) -> None:
        self.runtime_name = "langgraph"
        self.llm = get_langchain_llm()
        self.tool_registry = ToolRegistry()
        self.graph = self._build_graph()
        print("✅ LangGraph 阶段B初始化成功（ReAct + ToolRegistry）")

    def _build_graph(self) -> Any:
        graph: StateGraph = StateGraph(TripPlanningState)
        graph.add_node("information_react", self._node_information_react)
        graph.add_node("plan_trip", self._node_plan_trip)
        graph.set_entry_point("information_react")
        graph.add_edge("information_react", "plan_trip")
        graph.add_edge("plan_trip", END)
        return graph.compile()

    def plan_trip(self, request: TripRequest) -> TripPlan:
        init_state: TripPlanningState = {
            "request": request,
            "runtime": self.runtime_name,
            "errors": [],
            "tool_traces": [],
            "degraded_messages": [],
        }
        try:
            final_state = self.graph.invoke(init_state)
            if final_state.get("trip_plan"):
                return final_state["trip_plan"]
            return self._create_fallback_plan(request, "未生成结构化行程")
        except Exception as exc:
            return self._create_fallback_plan(request, f"LangGraph执行异常: {exc}")

    def _node_information_react(self, state: TripPlanningState) -> TripPlanningState:
        request = state["request"]
        observations: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        degraded_messages = list(state.get("degraded_messages", []))
        preferred_mode = self._route_mode_from_transportation(request.transportation)
        executed_actions: set[str] = set()
        executed_route_modes: set[str] = set()

        for step in range(1, 6):
            action = self._next_react_action(
                request,
                observations,
                step,
                preferred_mode,
                executed_actions,
                executed_route_modes,
            )
            thought = action.get("thought", "")
            action_name = action.get("action", "finish")
            action_input = action.get("action_input", {})

            if action_name == "finish":
                break

            executed_actions.add(action_name)
            if action_name == "map.route":
                executed_route_modes.add(action_input.get("mode", preferred_mode))

            start = time.time()
            tool_result = self.tool_registry.execute_with_fallback(action_name, action_input)
            duration = int((time.time() - start) * 1000)
            observation_text = json.dumps(tool_result, ensure_ascii=False)
            trace = ToolTrace(
                step=step,
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation_text,
                status="degraded" if tool_result.get("degraded") else "ok",
                error_category=tool_result.get("error_category"),
                duration_ms=duration,
            )
            traces.append(trace.to_dict())
            observations.append({"action": action_name, "result": tool_result})
            if tool_result.get("degraded"):
                degraded_messages.append(tool_result.get("message", "工具降级"))

        gathered = self._collect_gathered_data(observations, request)
        print("🧪 [ReAct] action/observation轨迹:")
        for item in traces:
            print(json.dumps(item, ensure_ascii=False))

        return {
            **state,
            "tool_traces": traces,
            "gathered_data": gathered,
            "degraded_messages": degraded_messages,
        }

    def _next_react_action(
        self,
        request: TripRequest,
        observations: List[Dict[str, Any]],
        step: int,
        preferred_mode: str,
        executed_actions: set[str],
        executed_route_modes: set[str],
    ) -> Dict[str, Any]:
        prompt = f"""你是旅行信息获取ReAct Agent。必须按JSON格式回复：
{{"thought":"...","action":"map.search_poi|map.weather|map.route|finish","action_input":{{...}}}}

约束：
1. 最多5步，每步仅一个 action。
2. 至少覆盖 map.search_poi、map.weather、map.route 三类工具后才允许 finish。
3. action_input 必须符合工具参数Schema。
4. 若已有足够信息，则输出 finish。

工具Schema：{self.tool_registry.get_tool_schemas_text()}

用户请求：
- 城市: {request.city}
- 日期: {request.start_date} 到 {request.end_date}
- 天数: {request.travel_days}
- 交通: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {','.join(request.preferences) if request.preferences else '无'}

已有观察：{json.dumps(observations, ensure_ascii=False)}
当前步数: {step}
"""
        try:
            content = getattr(self.llm.invoke(prompt), "content", "")
            parsed = self._safe_parse_json(content)
            if isinstance(parsed, dict) and parsed.get("action"):
                normalized = self._enforce_react_action_rules(
                    parsed,
                    request,
                    preferred_mode,
                    executed_actions,
                    executed_route_modes,
                )
                return normalized
        except Exception:
            pass

        defaults = [
            {"thought": "先找景点", "action": "map.search_poi", "action_input": {"city": request.city, "keywords": request.preferences[0] if request.preferences else "热门景点"}},
            {"thought": "查天气", "action": "map.weather", "action_input": {"city": request.city}},
            {"thought": "估算路线", "action": "map.route", "action_input": {"city": request.city, "origin": f"{request.city}站", "destination": f"{request.city}中心", "mode": preferred_mode}},
            {"thought": "信息够了", "action": "finish", "action_input": {}},
        ]
        candidate = defaults[min(step - 1, len(defaults) - 1)]
        return self._enforce_react_action_rules(candidate, request, preferred_mode, executed_actions, executed_route_modes)

    def _route_mode_from_transportation(self, transportation: str) -> str:
        mapping = {
            "公共交通": "transit",
            "自驾": "driving",
            "步行": "walking",
            "骑行": "riding",
            "混合": "transit",
        }
        return mapping.get(transportation, "driving")

    def _enforce_react_action_rules(
        self,
        action: Dict[str, Any],
        request: TripRequest,
        preferred_mode: str,
        executed_actions: set[str],
        executed_route_modes: set[str],
    ) -> Dict[str, Any]:
        required = {"map.search_poi", "map.weather", "map.route"}
        missing = [item for item in required if item not in executed_actions]
        action_name = action.get("action")

        if action_name == "finish" and missing:
            force = missing[0]
            action = {
                "thought": f"必须先补齐工具调用: {force}",
                "action": force,
                "action_input": {},
            }
            action_name = force

        if action_name == "finish" and request.transportation == "混合" and "walking" not in executed_route_modes:
            action = {
                "thought": "混合交通需要补充步行路线",
                "action": "map.route",
                "action_input": {
                    "city": request.city,
                    "origin": f"{request.city}中心",
                    "destination": f"{request.city}热门景点",
                    "mode": "walking",
                },
            }
            action_name = "map.route"

        if action_name == "map.route":
            route_input = {
                "city": request.city,
                "origin": f"{request.city}站",
                "destination": f"{request.city}中心",
                "mode": preferred_mode,
            }
            route_input.update(action.get("action_input", {}))
            if request.transportation == "混合" and executed_actions.issuperset({"map.search_poi", "map.weather"}):
                # 混合模式默认优先公交 + 步行兜底，这里在首次route调用时强制公交。
                route_input["mode"] = route_input.get("mode") or "transit"
            action["action_input"] = route_input

        if action_name == "map.search_poi":
            poi_input = {"city": request.city, "keywords": request.preferences[0] if request.preferences else "热门景点"}
            poi_input.update(action.get("action_input", {}))
            action["action_input"] = poi_input

        if action_name == "map.weather":
            weather_input = {"city": request.city}
            weather_input.update(action.get("action_input", {}))
            action["action_input"] = weather_input

        return action

    def _collect_gathered_data(self, observations: List[Dict[str, Any]], request: TripRequest) -> Dict[str, Any]:
        gathered: Dict[str, Any] = {
            "poi": {"data": {"pois": []}},
            "weather": {"data": {"forecasts": []}},
            "route": {"data": {}},
        }
        for obs in observations:
            action = obs.get("action", "")
            if action == "map.search_poi":
                gathered["poi"] = obs["result"]
            elif action == "map.weather":
                gathered["weather"] = obs["result"]
            elif action == "map.route":
                gathered["route"] = obs["result"]
        if not observations:
            gathered["route"] = {
                "ok": False,
                "degraded": True,
                "message": "无工具结果，使用静态路线",
                "data": {"origin": f"{request.city}站", "destination": f"{request.city}中心", "distance": None, "duration": None},
            }
        return gathered

    def _node_plan_trip(self, state: TripPlanningState) -> TripPlanningState:
        request = state["request"]
        gathered = state.get("gathered_data", {})
        degraded = state.get("degraded_messages", [])

        prompt = f"""你是旅行规划专家。请根据工具结果输出JSON行程，不要markdown。
并在 overall_suggestions 中解释景点与路线选择理由，且明确提示任何降级信息。

用户请求：{json.dumps(request.model_dump(), ensure_ascii=False)}
工具结果：{json.dumps(gathered, ensure_ascii=False)}
降级提示：{json.dumps(degraded, ensure_ascii=False)}

输出结构必须符合 TripPlan 模型字段。"""

        try:
            content = getattr(self.llm.invoke(prompt), "content", "")
            plan = self._parse_plan(content, request)
            if degraded:
                plan.overall_suggestions = f"{plan.overall_suggestions}\n\n工具降级说明: {'；'.join(degraded)}"
            return {**state, "planner_raw": str(content), "trip_plan": plan}
        except Exception as exc:
            plan = self._create_fallback_plan(request, f"LLM规划失败: {exc}")
            return {**state, "trip_plan": plan}

    def _safe_parse_json(self, text: Any) -> Any:
        if not isinstance(text, str):
            return {}
        clean = text.strip()
        if "```json" in clean:
            start = clean.find("```json") + 7
            end = clean.find("```", start)
            clean = clean[start:end].strip()
        elif clean.startswith("```"):
            start = clean.find("\n") + 1
            end = clean.rfind("```")
            clean = clean[start:end].strip()
        try:
            return json.loads(clean)
        except Exception:
            left = clean.find("{")
            right = clean.rfind("}")
            if left >= 0 and right > left:
                return json.loads(clean[left : right + 1])
        return {}

    def _parse_plan(self, content: Any, request: TripRequest) -> TripPlan:
        data = self._safe_parse_json(content)
        if isinstance(data, dict) and data:
            try:
                return TripPlan(**data)
            except Exception:
                pass
        return self._create_fallback_plan(request, "JSON解析失败")

    def _create_fallback_plan(self, request: TripRequest, reason: str) -> TripPlan:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            days.append(
                DayPlan(
                    date=current_date.strftime("%Y-%m-%d"),
                    day_index=i,
                    description=f"第{i + 1}天行程（降级方案）",
                    transportation=request.transportation,
                    accommodation=request.accommodation,
                    attractions=[
                        Attraction(
                            name=f"{request.city}推荐景点{i + 1}",
                            address=f"{request.city}市区",
                            location=Location(longitude=116.397128 + i * 0.01, latitude=39.916527 + i * 0.01),
                            visit_duration=120,
                            description="系统降级占位景点",
                            category="景点",
                        )
                    ],
                    meals=[
                        Meal(type="breakfast", name="早餐", description="酒店附近"),
                        Meal(type="lunch", name="午餐", description="景区周边"),
                        Meal(type="dinner", name="晚餐", description="本地餐厅"),
                    ],
                )
            )

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"系统返回可执行降级方案。原因: {reason}",
        )
