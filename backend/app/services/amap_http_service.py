"""高德地图 HTTP REST API 同步服务。

用于 LangGraph 路径：完全绕开 hello_agents.MCPTool 的异步依赖，
直接通过 requests 调用高德 Web 服务 REST API（需要 Web 服务类型 API Key）。

后续如需重新接入 MCP，可替换此模块的实现，接口签名保持不变。
"""

from typing import Dict, Optional

import requests

from ..config import get_settings


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _key() -> str:
    """读取高德 Web 服务 API Key。"""
    return get_settings().amap_api_key


_AMAP_BASE = "https://restapi.amap.com"


def _get(path: str, params: Dict) -> str:
    """发起 GET 请求，返回响应原始 JSON 字符串；失败时返回错误描述。"""
    try:
        resp = requests.get(
            f"{_AMAP_BASE}{path}",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.Timeout:
        return f"高德API请求超时: {path}"
    except requests.exceptions.HTTPError as e:
        return f"高德API HTTP错误 {e.response.status_code}: {path}"
    except Exception as e:
        return f"高德API调用失败: {e}"


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def amap_text_search(keywords: str, city: str, citylimit: bool = True) -> str:
    """POI 文本搜索。

    对应高德 API: GET /v3/place/text
    文档: https://lbs.amap.com/api/webservice/guide/api/search

    Returns:
        高德 API 返回的原始 JSON 字符串（供 LLM 直接阅读）。
    """
    return _get(
        "/v3/place/text",
        {
            "key": _key(),
            "keywords": keywords,
            "city": city,
            "citylimit": "true" if citylimit else "false",
            "output": "json",
            "offset": 10,
            "extensions": "all",
        },
    )


def amap_weather(city: str) -> str:
    """天气预报查询（最多 4 天）。

    对应高德 API: GET /v3/weather/weatherInfo
    文档: https://lbs.amap.com/api/webservice/guide/api/weatherinfo

    Returns:
        高德 API 返回的原始 JSON 字符串。
    """
    return _get(
        "/v3/weather/weatherInfo",
        {
            "key": _key(),
            "city": city,
            "extensions": "all",
            "output": "json",
        },
    )


def amap_geocode(address: str, city: Optional[str] = None) -> Optional[Dict[str, float]]:
    """地理编码：将地址/城市名转换为经纬度坐标。

    对应高德 API: GET /v3/geocode/geo
    文档: https://lbs.amap.com/api/webservice/guide/api/georegeo

    Returns:
        {'longitude': float, 'latitude': float} 或 None（失败时）。
    """
    params: Dict = {
        "key": _key(),
        "address": address,
        "output": "json",
    }
    if city:
        params["city"] = city

    try:
        resp = requests.get(
            f"{_AMAP_BASE}/v3/geocode/geo",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        geocodes = data.get("geocodes", [])
        if geocodes:
            loc: str = geocodes[0].get("location", "")
            if loc and "," in loc:
                lng_str, lat_str = loc.split(",", 1)
                return {
                    "longitude": float(lng_str),
                    "latitude": float(lat_str),
                }
    except Exception:
        pass
    return None

