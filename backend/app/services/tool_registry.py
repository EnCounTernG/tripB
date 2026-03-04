"""MCP工具统一注册与执行层。"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from ..config import get_settings

ToolErrorCategory = Literal["rate_limit", "timeout", "bad_request", "server_error", "unknown"]


class ToolExecutionError(Exception):
    """可分类的工具执行错误。"""

    def __init__(self, category: ToolErrorCategory, message: str):
        self.category = category
        self.message = message
        super().__init__(message)


@dataclass
class ToolTrace:
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    status: Literal["ok", "degraded", "error"]
    error_category: Optional[ToolErrorCategory] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "status": self.status,
            "error_category": self.error_category,
            "duration_ms": self.duration_ms,
        }


class SearchPOIArgs(BaseModel):
    city: str = Field(..., description="城市")
    keywords: str = Field(..., description="搜索关键词")
    limit: int = Field(default=8, ge=1, le=20, description="返回条数")


class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市")


class RouteArgs(BaseModel):
    city: str = Field(..., description="城市")
    origin: str = Field(..., description="起点")
    destination: str = Field(..., description="终点")
    mode: Literal["walking", "driving", "transit", "riding"] = Field(default="driving", description="路线类型")


class _ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[[BaseModel], Dict[str, Any]]


class ToolRegistry:
    """封装 map.search_poi / map.weather / map.route。"""

    def __init__(self, timeout_seconds: float = 8.0, retries: int = 1, rate_limit_seconds: float = 0.2):
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.rate_limit_seconds = rate_limit_seconds
        self._last_call: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._amap_key = get_settings().amap_api_key
        self._specs: Dict[str, _ToolSpec] = {
            "map.search_poi": _ToolSpec(
                name="map.search_poi",
                description="搜索城市中 POI 景点或地点信息",
                args_schema=SearchPOIArgs,
                handler=self._search_poi,
            ),
            "map.weather": _ToolSpec(
                name="map.weather",
                description="查询城市天气预报",
                args_schema=WeatherArgs,
                handler=self._weather,
            ),
            "map.route": _ToolSpec(
                name="map.route",
                description="规划起终点路线信息",
                args_schema=RouteArgs,
                handler=self._route,
            ),
        }

    def get_tool_schemas_text(self) -> str:
        payload = []
        for spec in self._specs.values():
            payload.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "args_schema": spec.args_schema.model_json_schema(),
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    def get_langchain_tools(self) -> List[StructuredTool]:
        tools: List[StructuredTool] = []
        for spec in self._specs.values():
            tool_name = spec.name
            tools.append(
                StructuredTool.from_function(
                    func=lambda _tool_name=tool_name, **kwargs: json.dumps(self.execute(_tool_name, kwargs), ensure_ascii=False),
                    name=tool_name,
                    description=spec.description,
                    args_schema=spec.args_schema,
                )
            )
        return tools

    def _classify_request_error(self, exc: Exception) -> ToolExecutionError:
        if isinstance(exc, requests.exceptions.Timeout):
            return ToolExecutionError("timeout", "请求超时")
        if isinstance(exc, requests.exceptions.HTTPError):
            code = exc.response.status_code if exc.response else 500
            if code == 429:
                return ToolExecutionError("rate_limit", "上游限流(429)")
            if 400 <= code < 500:
                return ToolExecutionError("bad_request", f"请求参数错误({code})")
            return ToolExecutionError("server_error", f"上游服务错误({code})")
        return ToolExecutionError("unknown", f"未知异常: {exc}")

    def _wait_rate_limit(self, tool_name: str) -> None:
        with self._lock:
            now = time.time()
            last = self._last_call.get(tool_name, 0)
            remaining = self.rate_limit_seconds - (now - last)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call[tool_name] = time.time()

    def execute(self, tool_name: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self._specs:
            raise ToolExecutionError("bad_request", f"未知工具: {tool_name}")

        spec = self._specs[tool_name]
        try:
            args_obj = spec.args_schema(**action_input)
        except ValidationError as exc:
            raise ToolExecutionError("bad_request", f"参数校验失败: {exc.errors()}") from exc

        last_err: Optional[ToolExecutionError] = None
        for _ in range(self.retries + 1):
            self._wait_rate_limit(tool_name)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(spec.handler, args_obj)
                    return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as exc:
                last_err = ToolExecutionError("timeout", f"工具执行超时({self.timeout_seconds}s)")
            except ToolExecutionError as exc:
                last_err = exc
            except Exception as exc:
                last_err = self._classify_request_error(exc)

            if last_err and last_err.category in {"bad_request", "rate_limit"}:
                break

        assert last_err is not None
        raise last_err

    def execute_with_fallback(self, tool_name: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self.execute(tool_name, action_input)
            return {"ok": True, "data": result, "degraded": False, "message": ""}
        except ToolExecutionError as exc:
            return {
                "ok": False,
                "degraded": True,
                "message": f"{tool_name} 调用失败({exc.category})，已降级",
                "error_category": exc.category,
                "data": self._fallback_data(tool_name, action_input, exc.category),
            }

    def _fallback_data(self, tool_name: str, action_input: Dict[str, Any], category: ToolErrorCategory) -> Dict[str, Any]:
        if tool_name == "map.weather":
            return {"city": action_input.get("city", ""), "forecasts": [], "note": f"天气接口不可用: {category}"}
        if tool_name == "map.route":
            return {
                "origin": action_input.get("origin", ""),
                "destination": action_input.get("destination", ""),
                "distance": None,
                "duration": None,
                "note": f"路线接口不可用: {category}",
            }
        return {"pois": [], "city": action_input.get("city", ""), "note": f"POI接口不可用: {category}"}

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._amap_key:
            raise ToolExecutionError("bad_request", "未配置 AMAP_API_KEY")
        resp = requests.get(
            f"https://restapi.amap.com{path}",
            params={"key": self._amap_key, **params, "output": "json"},
            timeout=6,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "0":
            info = payload.get("info", "未知错误")
            if "DAILY_QUERY_OVER_LIMIT" in info:
                raise ToolExecutionError("rate_limit", info)
            raise ToolExecutionError("bad_request", info)
        return payload

    def _search_poi(self, args: SearchPOIArgs) -> Dict[str, Any]:
        payload = self._request("/v3/place/text", {"city": args.city, "keywords": args.keywords, "offset": args.limit, "extensions": "all"})
        pois = payload.get("pois", [])
        return {
            "city": args.city,
            "keywords": args.keywords,
            "pois": [
                {
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "location": poi.get("location", ""),
                    "type": poi.get("type", ""),
                }
                for poi in pois
            ],
        }

    def _weather(self, args: WeatherArgs) -> Dict[str, Any]:
        payload = self._request("/v3/weather/weatherInfo", {"city": args.city, "extensions": "all"})
        forecasts = payload.get("forecasts", [])
        return {"city": args.city, "forecasts": forecasts}

    def _route(self, args: RouteArgs) -> Dict[str, Any]:
        mode_to_path = {
            "walking": "/v3/direction/walking",
            "driving": "/v3/direction/driving",
            "transit": "/v3/direction/transit/integrated",
            "riding": "/v4/direction/bicycling",
        }
        origin_loc = self._geocode(args.origin, args.city)
        destination_loc = self._geocode(args.destination, args.city)
        payload = self._request(mode_to_path[args.mode], {"origin": origin_loc, "destination": destination_loc, "city": args.city})
        route = payload.get("route", {})
        paths = route.get("paths") or route.get("transits") or []
        first = paths[0] if paths else {}
        return {
            "origin": args.origin,
            "destination": args.destination,
            "mode": args.mode,
            "distance": first.get("distance"),
            "duration": first.get("duration"),
        }

    def _geocode(self, address: str, city: str) -> str:
        payload = self._request("/v3/geocode/geo", {"address": address, "city": city})
        geocodes = payload.get("geocodes", [])
        if not geocodes:
            raise ToolExecutionError("bad_request", f"地址解析失败: {address}")
        return geocodes[0].get("location", "")
