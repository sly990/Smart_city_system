"""
智慧城市全域智能协同系统
使用LangGraph构建，包含多个专门的智能体来处理不同类型的客户查询
基于OpenAI兼容API提供LLM能力
支持多轮对话和会话管理
"""

import os
os.environ["BLOCKBUSTER_DISABLE"] = "1"


import json
import requests
import time
from typing import Dict, List, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.memory import BaseMemory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel
# 导入 RAG 模块
from rag.retriever import hybrid_retrieve, load_vector_store, connect_es
from rag.reranker import rerank_docs
import uuid
import asyncio
import uuid
from copy import deepcopy
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

# 导入分类工具
from tools.query_tools import classify_query




# 加载环境变量
load_dotenv()

# 导入配置
from config import *

# 导入智能体和工具
from agents import (UrbanPlanAgent, EvaluationAgent, 
                        SecurityAgent, ComplianceAgent, SmartServiceAgent, InfoQueryAgent)

from tools import classify_query

# 导入会话管理器
from session_manager import LangChainSessionManager, default_session_manager

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str # urban_planning / weather / general
    response: str
    tools_used: List[str]
    next_agent: str
    conversation_history: List[Any]
    memory: Any
    context: str
    documents: List[Any]
    # 新增字段
    expert_reports: Dict[str, str]
    reviewer_feedback: str
    final_plan: str


# OpenAI兼容API客户端类
class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = HTTP_TIMEOUT
        self.max_retries = HTTP_MAX_RETRIES
        self.headers = HTTP_HEADERS.copy()
        self.headers["Authorization"] = f"Bearer {api_key}"

        # 添加LangChain回调管理器所需的属性
        self.parent_run_id = None
        self.run_id = None
        self.tags = []
        self.metadata = {}
        self.handlers = []
        self.callback_manager = None
        self.inheritable_handlers = []
        self.inheritable_tags = []
        self.inheritable_metadata = {}

    def invoke(self, messages, tools=None, **kwargs):
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        formatted_messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        # 【修改点 1】保留 LLM 第一轮返回的 tool_calls
                        msg_dict = {"role": "assistant", "content": msg.content or ""}
                        if hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                            msg_dict["tool_calls"] = msg.additional_kwargs["tool_calls"]
                        formatted_messages.append(msg_dict)
                    elif msg.type == 'tool':
                        # 【修改点 2】正确识别并格式化 ToolMessage
                        formatted_messages.append({
                            "role": "tool",
                            "content": msg.content,
                            "tool_call_id": getattr(msg, "tool_call_id", "")
                        })
                    elif msg.type == 'system':
                        formatted_messages.append({"role": "system", "content": msg.content})
                    else:
                        formatted_messages.append({"role": "user", "content": msg.content})
                else:
                    formatted_messages.append({"role": "user", "content": msg.content})
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})

        # 构建请求payload
        payload = {
            "model": self.model,
            "messages": formatted_messages
        }


        # 新增：如果传入了工具，加入 payload
        if tools:
            payload["tools"] = tools
            # 如果强制要求调用某个工具，可在此处理 tool_choice


        # 添加调试信息
        print(f"🔍 Debug: API request:")
        print(f"   URL: {self.base_url}/chat/completions")
        print(f"   Model: {self.model}")
        print(f"   Messages: {len(formatted_messages)}")
        print(f"   Format: {formatted_messages[:2]}...")  # 只显示前两条

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                # # 提取响应内容
                # if "choices" in result and len(result["choices"]) > 0:
                #     message = result["choices"][0].get("message", {})
                #     content = message.get("content", "")
                #     return CustomResponse(content)
                # else:
                #     return CustomResponse("API response format error")
                
                # 提取响应内容
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    # 新增：检查 LLM 是否决定调用工具
                    if "tool_calls" in message:
                        # 返回包含 tool_calls 的 AIMessage 以兼容 LangChain
                        return AIMessage(content="", additional_kwargs={"tool_calls": message["tool_calls"]})
                    else:
                        content = message.get("content", "")
                        return CustomResponse(content)
                else:
                    return CustomResponse("API response format error")



            except requests.exceptions.RequestException as e:
                print(f"❌ Debug: Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"API call failed: {e}")
                time.sleep(2 ** attempt)  # 指数退避

    def chat(self, messages):
        """兼容LangChain的chat方法"""
        return self.invoke(messages)

    # 添加LangChain回调管理器接口
    def bind(self, **kwargs):
        """绑定参数到客户端"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def with_config(self, config):
        """设置配置"""
        if hasattr(config, 'get'):
            for key, value in config.items():
                setattr(self, key, value)
        return self

class CustomResponse:
    def __init__(self, content):
        self.content = content

# 全局会话管理器（使用 LangChain 标准接口）
session_manager = default_session_manager

# 延迟初始化LLM
_llm_instance = None

def initialize_llm_client():
    """初始化OpenAI兼容API客户端"""
    if not OPENAI_API_KEY:
        raise ValueError("API密钥未设置")

    return OpenAICompatibleClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL
    )

def get_llm():
    """获取LLM实例，延迟初始化"""
    global _llm_instance
    if _llm_instance is None:
        try:
            if not OPENAI_API_KEY:
                print("❌ 错误: API密钥未设置，无法初始化LLM")
                _llm_instance = None
            else:
                _llm_instance = initialize_llm_client()
                print(f"✅ 成功初始化API客户端")
        except Exception as e:
            print(f"❌ 初始化API客户端失败: {e}")
            print("将使用模拟响应模式")
            _llm_instance = None
    return _llm_instance

# 初始化智能体
def initialize_agents():
    from agents import (UrbanPlanAgent, EvaluationAgent, 
                        SecurityAgent, ComplianceAgent, SmartServiceAgent, InfoQueryAgent)
    agents = {
        "urban_plan_agent": UrbanPlanAgent(),
        "evaluation_agent": EvaluationAgent(),
        "security_agent": SecurityAgent(),
        "compliance_agent": ComplianceAgent(),
        "smart_service_agent": SmartServiceAgent(),
        "info_query_agent": InfoQueryAgent()
    }
    for agent in agents.values():
        agent.set_llm(get_llm())
        agent.set_session_manager(default_session_manager)
    return agents




# ==================== 分类节点（调用工具） ====================
def classify_query_node(state: AgentState) -> AgentState:
    """使用 classify_query 工具进行三分类"""
    # 初始化默认字段
    if "session_id" not in state:
        state["session_id"] = str(uuid.uuid4())
    if "tools_used" not in state:
        state["tools_used"] = []
    if "conversation_history" not in state:
        state["conversation_history"] = []
    if "memory" not in state:
        state["memory"] = None
    if "next_agent" not in state:
        state["next_agent"] = ""
    if "messages" not in state:
        state["messages"] = []
    if "expert_reports" not in state:
        state["expert_reports"] = {}
    if "reviewer_feedback" not in state:
        state["reviewer_feedback"] = ""
    if "final_plan" not in state:
        state["final_plan"] = ""

    customer_query = state.get("customer_query", "")
    if not customer_query:
        state["response"] = "Error: No customer query provided"
        state["query_type"] = "general"
        return state

    # 调用工具
    llm = get_llm()
    try:
        # 注意：classify_query 是一个被 @tool 装饰的函数，可以直接调用
        query_type = classify_query.invoke({"query": customer_query, "llm": llm})
    except Exception as e:
        print(f"工具调用失败: {e}")
        query_type = "general"

    state["query_type"] = query_type
    state["tools_used"].append("classify_query")
    
    # 记录用户消息
    try:
        session_manager.add_message(state["session_id"], customer_query, is_user=True)
    except Exception as e:
        print(f"记录消息失败: {e}")
    
    return state

# ==================== 编排者节点 ====================
def orchestrator_node(state: AgentState) -> AgentState:
    """确定需要调用的专家列表（固定四个）"""
    print("🎯 编排者：准备调用四个领域专家")
    state["next_agent"] = "parallel_experts"
    state["tools_used"].append("orchestrator")
    return state

# ==================== 并行执行节点 ====================
async def parallel_experts_node(state: AgentState) -> AgentState:
    """并发调用四个专家"""
    print("⚡ 并行执行：四个专家同时工作...")
    agents = initialize_agents()
    expert_names = ["urban_plan_agent", "evaluation_agent", "security_agent", "compliance_agent"]
    
    # async def call_expert(agent_name, state_copy):
    #     agent = agents.get(agent_name)
    #     if not agent:
    #         return agent_name, f"错误：未找到专家 {agent_name}"
    #     try:
    #         result = agent.process(state_copy)
    #         report = result.get("response", "专家未返回有效报告")
    #         return agent_name, report
    #     except Exception as e:
    #         return agent_name, f"报告生成失败：{str(e)}"
    
    # 3) 在 async 节点里，把同步 agent.process 放到线程里执行
    async def call_expert(agent_name, state_copy):
        agent = agents.get(agent_name)
        if not agent:
            return agent_name, f"错误：未找到专家 {agent_name}"
        try:
            result = await asyncio.to_thread(agent.process, state_copy)
            return agent_name, result.get("response", "专家未返回有效报告")
        except Exception as e:
            return agent_name, f"报告生成失败：{str(e)}"
    
    tasks = []
    for expert in expert_names:
        state_copy = deepcopy(state)
        state_copy.pop("response", None)
        tasks.append(call_expert(expert, state_copy))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            print(f"并行任务异常: {result}")
            continue
        agent_name, report = result
        state["expert_reports"][agent_name] = report
        state["tools_used"].append(agent_name)
    
    print(f"✅ 并行完成，收到 {len(state['expert_reports'])} 份报告")
    return state

# ==================== 评审者节点 ====================
def reviewer_node(state: AgentState) -> AgentState:
    """识别冲突并给出修改意见"""
    print("🔍 评审者：分析冲突...")
    reports = state.get("expert_reports", {})
    if len(reports) < 4:
        state["reviewer_feedback"] = "警告：未收到全部专家报告，评审可能不完整。"
        return state
    
    prompt = f"""
你是首席评审专家。以下四个专家针对用户查询给出了报告：

【城市规划】{reports.get("urban_plan_agent", "")}
【评价体系】{reports.get("evaluation_agent", "")}
【安全】{reports.get("security_agent", "")}
【合规】{reports.get("compliance_agent", "")}

用户查询：{state.get("customer_query", "")}

任务：
综合各专业报告，识别规划、安全、合规等冲突点，
评估严重程度，提出具体修改建议，最终输出无矛盾、
可执行的整合方案。遵循安全优先原则，依据标准裁决。

直接输出结果。
"""
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        feedback = response.content
    except Exception as e:
        feedback = f"评审出错：{str(e)}"
    
    state["reviewer_feedback"] = feedback
    state["final_plan"] = feedback
    state["tools_used"].append("reviewer")
    return state

# ==================== 最终响应节点 ====================
def final_response_node(state: AgentState) -> AgentState:
    qtype = state.get("query_type")
    query = state.get("customer_query", "")
    
    if qtype == "urban_planning":
        final_msg = f"""
📋 **智慧城市综合方案**（基于四专家并行评审）

您的问题：{query}

【冲突分析与修改意见】
{state.get("reviewer_feedback", "无")}

【最终建议】
{state.get("final_plan", "请稍后重试")}

---
*本方案由城市规划、评价体系、安全、合规四位专家联合制定，经首席评审优化。*
"""
    elif qtype == "weather":
        final_msg = state.get("response", "🌤️ 天气服务暂时不可用。")
    else:
        final_msg = state.get("response", "您好，请问有什么可以帮您？")
    
    state["response"] = final_msg
    return state

# ==================== 辅助节点工厂 ====================
def create_agent_node(agent_name: str):
    def agent_node(state: AgentState) -> AgentState:
        agents = initialize_agents()
        agent = agents.get(agent_name)
        if agent:
            session_id = state["session_id"]
            conv_ctx = session_manager.get_conversation_context(session_id)
            state["conversation_history"] = conv_ctx
            result = agent.process(state)
            state["response"] = result.get("response", "无响应")
            state["current_agent"] = agent_name
            # 记录AI回复
            try:
                session_manager.add_message(session_id, state["response"], is_user=False)
            except Exception as e:
                print(f"记录AI消息失败: {e}")
            return state
        else:
            state["response"] = f"错误：未找到智能体 {agent_name}"
            return state
    return agent_node

# ==================== 路由函数 ====================
def route_after_classify(state: AgentState) -> str:
    qtype = state.get("query_type", "general")
    if qtype == "urban_planning":
        return "orchestrator"
    elif qtype == "info_query":
        return "info_query_agent"
    else:
        return "smart_service_agent"


# 图表入口点
# 使用方式：在langgraph.json文件中增加以下配置，声明构建图的方式，硬编码方式实现。
# "graphs": {
#     "customer_service": "./multi_agent_customer_service.py:make_graph"
# },
# 也可以在langgraph.json文件中使用workflow配置化的方式定义图的结构，但功能相对简单，无法实现复杂的逻辑
def make_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("parallel_experts", parallel_experts_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("info_query_agent", create_agent_node("info_query_agent"))
    workflow.add_node("smart_service_agent", create_agent_node("smart_service_agent"))
    workflow.add_node("final_response", final_response_node)
    
    workflow.set_entry_point("classify_query")
    
    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "orchestrator": "orchestrator",
            "info_query_agent": "info_query_agent",
            "smart_service_agent": "smart_service_agent"
        }
    )
    
    workflow.add_edge("orchestrator", "parallel_experts")
    workflow.add_edge("parallel_experts", "reviewer")
    workflow.add_edge("reviewer", "final_response")
    workflow.add_edge("info_query_agent", "final_response")
    workflow.add_edge("smart_service_agent", "final_response")
    
    workflow.set_finish_point("final_response")
    
    return workflow.compile()

# 创建默认工作流实例
if __name__ == "__main__":
    # 1. 编译图
    app = make_graph()
    
    # # 2. 准备测试用例
    # test_queries = [
    #     {
    #         "desc": "测试 MCP 实时工具调用 (天气查询)",
    #         "data": {
    #             "customer_query": "广州今天的实时天气怎么样？适合出门散步吗？", 
    #             "session_id": "test_mcp_001",
    #             "tools_used": [] # 确保 state 中有这个初始化列表
    #              }
    #     }
    # ]

    # # 修改后的遍历执行逻辑（建议加上对 tools_used 的打印，方便观察 MCP 是否生效）
    # for i, test in enumerate(test_queries):
    #     print(f"\n{'='*20} 运行测试 {i+1}: {test['desc']} {'='*20}")
        
    #     # 初始状态
    #     initial_state = {
    #         "session_id": test["data"]["session_id"],
    #         "customer_query": test["data"]["customer_query"],
    #         "messages": [],
    #         "query_type": None,
    #         "context": [],
    #         "response": "",
    #         "current_agent": "",
    #         "tools_used": []
    #     }
        
    #     # 执行工作流 (如果你的节点改成了异步，这里记得用 await app.ainvoke)
    #     # result = await app.ainvoke(initial_state) 
    #     result = app.invoke(initial_state)
        
    #     print(f"用户问题: {test['data']['customer_query']}")
    #     print(f"最终响应: {result['response']}")
    #     print(f"调用链路: {result['tools_used']}")
    #     print(f"{'='*60}\n")
