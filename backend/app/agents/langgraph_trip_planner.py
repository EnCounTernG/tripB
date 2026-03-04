"""基于LangGraph的旅行规划运行时（阶段A）。

架构说明：
- 编排层：LangGraph StateGraph（景点 → 天气 → 酒店 → 规划，线性流程）
- LLM 层：LangChain ChatOpenAI（统一模型调用）
- 地图数据层：直接调用高德 REST API（amap_http_service），不依赖 hello_agents

与 HelloAgents 路径的关系：
- HelloAgents 路径（trip_planner_agent.py）保持不变，仍通过 MCPTool 调用地图
- LangGraph 路径完全独立，两条路径互不影响
- 后续如需接入 LangChain MCP 工具，只需替换 amap_http_service 中的实现
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..models.schemas import Attraction, DayPlan, Location, Meal, TripPlan, TripRequest
from ..services.amap_http_service import amap_geocode, amap_text_search, amap_weather
from ..services.langchain_service import get_langchain_llm

# 主要中国城市中心坐标缓存，用于 fallback 方案（避免网络依赖）
_CITY_COORDS: Dict[str, Dict[str, float]] = {
    "北京": {"longitude": 116.397128, "latitude": 39.916527},
    "上海": {"longitude": 121.473701, "latitude": 31.230416},
    "广州": {"longitude": 113.264385, "latitude": 23.129112},
    "深圳": {"longitude": 114.057868, "latitude": 22.543099},
    "成都": {"longitude": 104.066541, "latitude": 30.572269},
    "杭州": {"longitude": 120.153576, "latitude": 30.287459},
    "西安": {"longitude": 108.948024, "latitude": 34.263161},
    "武汉": {"longitude": 114.298572, "latitude": 30.584355},
    "南京": {"longitude": 118.796877, "latitude": 32.060255},
    "重庆": {"longitude": 106.551557, "latitude": 29.563009},
    "天津": {"longitude": 117.200983, "latitude": 39.084158},
    "石家庄": {"longitude": 114.502461, "latitude": 38.045474},
    "青岛": {"longitude": 120.382639, "latitude": 36.067082},
    "大连": {"longitude": 121.614682, "latitude": 38.914003},
    "厦门": {"longitude": 118.089425, "latitude": 24.479834},
    "苏州": {"longitude": 120.619585, "latitude": 31.299379},
    "长沙": {"longitude": 112.938814, "latitude": 28.228209},
    "郑州": {"longitude": 113.625368, "latitude": 34.746599},
    "哈尔滨": {"longitude": 126.642464, "latitude": 45.756967},
    "沈阳": {"longitude": 123.429096, "latitude": 41.796767},
    "济南": {"longitude": 117.000923, "latitude": 36.675807},
    "合肥": {"longitude": 117.227239, "latitude": 31.820587},
    "福州": {"longitude": 119.306239, "latitude": 26.075302},
    "乌鲁木齐": {"longitude": 87.617733, "latitude": 43.792818},
    "昆明": {"longitude": 102.712251, "latitude": 25.040609},
    "南宁": {"longitude": 108.320004, "latitude": 22.82402},
    "贵阳": {"longitude": 106.713478, "latitude": 26.578343},
    "兰州": {"longitude": 103.834303, "latitude": 36.061089},
    "银川": {"longitude": 106.278179, "latitude": 38.46637},
    "西宁": {"longitude": 101.778916, "latitude": 36.623178},
    "呼和浩特": {"longitude": 111.670801, "latitude": 40.818311},
    "南昌": {"longitude": 115.858197, "latitude": 28.682892},
    "太原": {"longitude": 112.549248, "latitude": 37.857014},
    "海口": {"longitude": 110.331227, "latitude": 20.031971},
    "拉萨": {"longitude": 91.117212, "latitude": 29.646923},
}


def _get_city_coords(city: str) -> Dict[str, float]:
    """获取城市中心坐标。优先查缓存，其次调用高德 geocode，最后降级为默认坐标。"""
    if city in _CITY_COORDS:
        return _CITY_COORDS[city]

    # 尝试实时 geocode
    try:
        result = amap_geocode(city)
        if result:
            return result
    except Exception:
        pass

    # 最终兜底：使用北京作为默认（并打印警告）
    print(f"⚠️  无法获取 [{city}] 的坐标，使用默认坐标（北京）")
    return {"longitude": 116.397128, "latitude": 39.916527}


# ---------------------------------------------------------------------------
# LangGraph 状态定义
# ---------------------------------------------------------------------------

class TripPlanningState(TypedDict, total=False):
    """在各节点之间传递上下文与中间结果。"""

    request: TripRequest
    attractions_raw: str
    weather_raw: str
    hotels_raw: str
    planner_raw: str
    trip_plan: TripPlan
    runtime: str
    errors: List[str]


# ---------------------------------------------------------------------------
# 规划器主类
# ---------------------------------------------------------------------------

class LangGraphTripPlanner:
    """基于 LangGraph 的旅行规划器。

    地图数据通过 amap_http_service 同步获取（高德 REST API），
    行程规划通过 LangChain ChatOpenAI 生成。
    """

    def __init__(self) -> None:
        self.runtime_name = "langgraph"
        self.llm = get_langchain_llm()
        self.graph = self._build_graph()
        print("✅ LangGraph 旅行规划器初始化成功（使用高德 HTTP API）")

    def _build_graph(self) -> Any:
        """构建 LangGraph 流程图（线性流程）。"""
        graph: StateGraph = StateGraph(TripPlanningState)

        graph.add_node("search_attractions", self._node_search_attractions)
        graph.add_node("query_weather", self._node_query_weather)
        graph.add_node("search_hotels", self._node_search_hotels)
        graph.add_node("plan_trip", self._node_plan_trip)

        graph.set_entry_point("search_attractions")
        graph.add_edge("search_attractions", "query_weather")
        graph.add_edge("query_weather", "search_hotels")
        graph.add_edge("search_hotels", "plan_trip")
        graph.add_edge("plan_trip", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """执行 LangGraph 流程，输出 TripPlan。"""
        init_state: TripPlanningState = {
            "request": request,
            "runtime": self.runtime_name,
            "errors": [],
        }

        try:
            final_state = self.graph.invoke(init_state)
            if final_state.get("trip_plan"):
                return final_state["trip_plan"]
            return self._create_fallback_plan(request, "未生成结构化行程，使用降级方案")
        except Exception as exc:
            return self._create_fallback_plan(request, f"LangGraph执行异常: {exc}")

    # ------------------------------------------------------------------
    # 节点实现
    # ------------------------------------------------------------------

    def _node_search_attractions(self, state: TripPlanningState) -> TripPlanningState:
        """节点1：通过高德 REST API 查询景点 POI。"""
        request = state["request"]
        keyword = request.preferences[0] if request.preferences else "热门景点"

        print(f"📍 [LangGraph] 搜索景点: city={request.city}, keyword={keyword}")
        try:
            raw = amap_text_search(keywords=keyword, city=request.city)
            print(f"   景点数据长度: {len(raw)} 字节")
        except Exception as exc:
            raw = f"景点检索失败: {exc}"
            state.setdefault("errors", []).append(raw)
            print(f"   ❌ {raw}")

        return {**state, "attractions_raw": raw}

    def _node_query_weather(self, state: TripPlanningState) -> TripPlanningState:
        """节点2：通过高德 REST API 查询天气预报。"""
        request = state["request"]

        print(f"🌤️  [LangGraph] 查询天气: city={request.city}")
        try:
            raw = amap_weather(city=request.city)
            print(f"   天气数据长度: {len(raw)} 字节")
        except Exception as exc:
            raw = f"天气检索失败: {exc}"
            state.setdefault("errors", []).append(raw)
            print(f"   ❌ {raw}")

        return {**state, "weather_raw": raw}

    def _node_search_hotels(self, state: TripPlanningState) -> TripPlanningState:
        """节点3：通过高德 REST API 查询酒店 POI。"""
        request = state["request"]
        hotel_keyword = f"{request.accommodation} 酒店"

        print(f"🏨 [LangGraph] 搜索酒店: city={request.city}, keyword={hotel_keyword}")
        try:
            raw = amap_text_search(keywords=hotel_keyword, city=request.city)
            print(f"   酒店数据长度: {len(raw)} 字节")
        except Exception as exc:
            raw = f"酒店检索失败: {exc}"
            state.setdefault("errors", []).append(raw)
            print(f"   ❌ {raw}")

        return {**state, "hotels_raw": raw}

    def _node_plan_trip(self, state: TripPlanningState) -> TripPlanningState:
        """节点4：用 LangChain LLM 整合数据并输出结构化行程 JSON。"""
        request = state["request"]

        print("📋 [LangGraph] 调用 LLM 生成行程计划...")

        prompt = f"""你是旅行规划专家。请根据以下信息生成旅行计划，只返回 JSON，不要有任何 markdown 代码块包裹。

输入信息：
- 城市：{request.city}
- 开始日期：{request.start_date}
- 结束日期：{request.end_date}
- 天数：{request.travel_days}
- 交通方式：{request.transportation}
- 住宿偏好：{request.accommodation}
- 偏好：{', '.join(request.preferences) if request.preferences else '无'}
- 用户额外要求：{request.free_text_input or '无'}

景点检索结果（高德地图 POI 数据）：
{state.get('attractions_raw', '无数据')}

天气检索结果（高德地图天气数据）：
{state.get('weather_raw', '无数据')}

酒店检索结果（高德地图 POI 数据）：
{state.get('hotels_raw', '无数据')}

JSON 结构要求（严格按此格式，字段名不得改变）：
{{
  "city": "城市名",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {{
        "name": "酒店名称（来自检索结果）",
        "address": "酒店地址",
        "location": {{"longitude": 116.0, "latitude": 39.0}},
        "price_range": "200-400元",
        "rating": "4.5",
        "distance": "距景点约1公里",
        "type": "经济型酒店",
        "estimated_cost": 300
      }},
      "attractions": [
        {{
          "name": "景点名称（来自检索结果）",
          "address": "详细地址",
          "location": {{"longitude": 116.0, "latitude": 39.0}},
          "visit_duration": 120,
          "description": "景点描述",
          "category": "景点类别",
          "ticket_price": 50
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐", "description": "描述", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐", "description": "描述", "estimated_cost": 60}},
        {{"type": "dinner", "name": "晚餐", "description": "描述", "estimated_cost": 80}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 20,
      "night_temp": 12,
      "wind_direction": "南风",
      "wind_power": "3级"
    }}
  ],
  "overall_suggestions": "旅行总体建议",
  "budget": {{
    "total_attractions": 300,
    "total_hotels": 900,
    "total_meals": 510,
    "total_transportation": 200,
    "total": 1910
  }}
}}

重要要求：
1. 每天安排 2-3 个景点，景点必须来自检索结果中的真实 POI
2. location 的经纬度必须使用检索结果中的真实坐标
3. 酒店必须来自酒店检索结果
4. weather_info 必须包含每一天的天气，温度为纯数字（不带°C）
5. 只输出 JSON，不要有任何解释文字或代码块标记"""

        try:
            message = self.llm.invoke(prompt)
            content = getattr(message, "content", "")
            print(f"   LLM 响应长度: {len(str(content))} 字节")
            plan = self._parse_plan(content, request)
            return {**state, "planner_raw": str(content), "trip_plan": plan}
        except Exception as exc:
            err = f"LLM规划失败: {exc}"
            state.setdefault("errors", []).append(err)
            print(f"   ❌ {err}")
            plan = self._create_fallback_plan(request, err)
            return {**state, "trip_plan": plan}

    # ------------------------------------------------------------------
    # 解析与降级
    # ------------------------------------------------------------------

    def _parse_plan(self, content: Any, request: TripRequest) -> TripPlan:
        """将 LLM 输出解析为 TripPlan。"""
        if not isinstance(content, str):
            return self._create_fallback_plan(request, "LLM输出非字符串")

        text = content.strip()

        # 去除常见的 markdown 代码块包裹
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif text.startswith("```"):
            start = text.find("\n") + 1  # 跳过 ```xxx 首行
            end = text.rfind("```")
            text = text[start:end].strip()

        # 尝试直接解析
        try:
            data = json.loads(text)
            return TripPlan(**data)
        except Exception:
            pass

        # 兜底：截取首尾大括号之间的内容
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return TripPlan(**data)
        except Exception:
            pass

        return self._create_fallback_plan(request, "JSON解析失败")

    def _create_fallback_plan(self, request: TripRequest, reason: str) -> TripPlan:
        """当图执行或解析失败时，返回基于城市真实坐标的降级行程。"""
        print(f"⚠️  [LangGraph] 触发降级方案，原因: {reason}")

        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        # 获取目标城市的真实坐标
        city_coords = _get_city_coords(request.city)

        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            # 在城市中心附近偏移，模拟不同景点位置
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i + 1}天行程（降级方案，建议重新生成）",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}推荐景点{i + 1}",
                        address=f"{request.city}市区",
                        location=Location(
                            longitude=city_coords["longitude"] + i * 0.01,
                            latitude=city_coords["latitude"] + i * 0.01,
                        ),
                        visit_duration=120,
                        description="系统降级时生成的占位景点，建议稍后重试获取实时推荐",
                        category="景点",
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐", description="酒店或附近早餐"),
                    Meal(type="lunch", name="午餐", description="景区周边简餐"),
                    Meal(type="dinner", name="晚餐", description="本地特色餐厅"),
                ],
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"系统已返回可执行降级方案。原因: {reason}",
        )
