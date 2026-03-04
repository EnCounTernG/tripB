"""LangChain LLM服务模块。

该模块在阶段A中引入,用于提供基于LangChain的聊天模型实例。
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

# 全局实例,避免重复初始化
_langchain_llm_instance: Optional[ChatOpenAI] = None


def get_langchain_llm() -> ChatOpenAI:
    """获取LangChain聊天模型实例(单例模式)。"""
    global _langchain_llm_instance

    if _langchain_llm_instance is None:
        # 兼容当前项目已有的环境变量命名
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        if not api_key:
            raise ValueError("缺少LLM_API_KEY或OPENAI_API_KEY,无法初始化LangChain模型")

        _langchain_llm_instance = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )

        print("✅ LangChain LLM服务初始化成功")
        print(f"   模型: {model}")
        if base_url:
            print(f"   Base URL: {base_url}")

    return _langchain_llm_instance


def reset_langchain_llm() -> None:
    """重置LangChain LLM实例(用于测试或重新配置)。"""
    global _langchain_llm_instance
    _langchain_llm_instance = None
