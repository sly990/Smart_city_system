"""
查询分类工具函数
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

@tool
def classify_query(query: str, llm=None) -> str:
    """根据客户查询内容分类查询类型，返回：urban_planning / info_query / general"""


    system_prompt = """你是一个查询分类专家。请根据客户查询内容，将查询分类为以下三种类型之一：
- urban_planning: 与智慧城市规划、建设、评价、安全、合规等城市建设相关的问题
- info_query: 查询天气等信息（如“今天天气怎么样？”“明天会下雨吗？”等）
- general: 其他所有通用问题（问候、闲聊、简单咨询、与城市无关的问题等）

只返回分类标签，不要有任何其他解释或标点符号。"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请分类以下查询：{query}")
    ]
    
    try:
        response = llm.invoke(messages)
        result = response.content.strip().lower()
        # 确保返回值合法
        if result not in ["urban_planning", "info_query", "general"]:
            result = "general"
        return result
    except Exception as e:
        print(f"分类工具出错: {e}")
        return "general"