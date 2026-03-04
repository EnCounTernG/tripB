"""旅行规划运行时管理器。

阶段A通过该模块并存两套运行时:
- helloagents: 历史实现
- langgraph: 新实现(默认)
"""

from typing import Literal, Optional

from .langgraph_trip_planner import LangGraphTripPlanner
from .trip_planner_agent import MultiAgentTripPlanner

RuntimeType = Literal["langgraph", "helloagents"]

_langgraph_runtime: Optional[LangGraphTripPlanner] = None
_helloagents_runtime: Optional[MultiAgentTripPlanner] = None


def get_trip_planner_runtime(runtime: RuntimeType = "langgraph"):
    """按名称获取运行时实例(单例模式)。"""
    global _langgraph_runtime, _helloagents_runtime

    if runtime == "helloagents":
        if _helloagents_runtime is None:
            _helloagents_runtime = MultiAgentTripPlanner()
        return _helloagents_runtime

    if _langgraph_runtime is None:
        _langgraph_runtime = LangGraphTripPlanner()
    return _langgraph_runtime
