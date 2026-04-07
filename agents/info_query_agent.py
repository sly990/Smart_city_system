"""
智慧城市信息查询与便民服务智能体
为市民及城市访客提供便捷、准确的城市运行信息查询服务，支持天气、交通、设施、办事指南、文体活动等查询，并结合位置与偏好提供个性化推荐
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent
import json
import asyncio
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import threading
import os
import logging
import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class InfoQueryAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市信息查询与便民服务专家",
            role="为市民及城市访客提供便捷、准确的城市运行信息查询服务。支持实时查询天气状况、空气质量、交通路况、公共交通实时到站信息、公共设施分布（如停车场、公厕、充电桩）、政务服务办事指南、文体活动安排等。通过自然语言交互快速理解用户意图，结合多源动态数据与知识库直观呈现结果，并可根据用户位置与偏好提供个性化推荐。",
            expertise=["天气与空气质量查询", "交通路况与公交到站", "公共设施分布查询", "政务服务办事指南", "文体活动安排", "个性化推荐"],
            
        )
        

        # TODO: 城市信息应从实时数据接口（气象、交通、GIS）获取，这里只是模拟数据
        # 实际应用中应连接城市大数据平台、地图API、政务服务平台等
        self.city_info_knowledge_base = {
            "天气与空气质量": {
                "查询方式": "提供城市名称或区域，可获取实时天气",
                "示例": "北京朝阳区：晴，24℃~30℃³",
                "预警信息": "暴雨、雷电、大风、高温、雾霾等气象预警实时推送",
                "生活指数": "穿衣、洗车、运动、感冒、钓鱼等指数建议"
            },
            "交通路况与公交到站": {
                "实时路况": "城市主要道路拥堵指数、事故路段、施工绕行提示",
                "公交到站": "支持查询指定公交线路的下一班到站时间、车辆位置、拥挤度",
                "地铁运营": "首末班车时间、换乘指引、站点出口信息、限流通知",
                "共享交通": "共享单车、共享汽车停放点分布及车辆数量"
            },
            "公共设施分布": {
                "停车场": "周边停车场位置、剩余车位、收费标准、开放时间",
                "公共厕所": "附近公厕位置、是否无障碍、母婴室配置",
                "充电桩": "电动汽车充电桩位置、接口类型、空闲桩数、电费标准",
                "其他设施": "加油站、公园、医院、派出所、行政服务中心等"
            },
            "政务服务办事指南": {
                "办事事项": "户籍、出入境、社保、公积金、不动产、税务等高频事项",
                "所需材料": "每项业务需提供的证件、表格、复印件清单",
                "办理流程": "预约 → 提交 → 审核 → 领证，各环节时限",
                "线上渠道": "政务App、小程序、一网通办平台入口"
            },
            "文体活动安排": {
                "活动类型": "展览、演出、讲座、体育赛事、亲子活动、社区活动",
                "时间地点": "活动日期、具体时段、场馆地址、交通指引",
                "参与方式": "免费/收费、预约方式、名额限制、联系方式",
                "推荐专题": "本周热门、周末去哪儿、亲子优选、免费活动"
            },
            "个性化推荐": {
                "推荐依据": "用户历史查询、当前位置、时间（早晚/节假日）、偏好标签（如亲子、运动、艺术）",
                "推荐内容": "附近餐饮、景点、活动、便民服务点、出行方案",
                "隐私保护": "位置信息仅用于查询，不上传服务器，可随时关闭",
                "反馈机制": "用户可对推荐结果点赞/点踩，持续优化模型"
            }
        }
    async def _call_mcp_weather_tool(self, location: str) -> str:
        """使用官方 MCP Streamable HTTP 客户端调用天气工具"""
        try:
           # 兼容本地运行与 LangGraph Docker 环境
            mcp_url = os.getenv("MCP_WEATHER_URL", "http://127.0.0.1:8001/mcp")
            async with streamable_http_client(mcp_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "get_realtime_weather",
                        arguments={"location": location}
                    )

                    if result.content:
                        return result.content[0].text
                    return "未能获取到天气数据"

        except Exception as e:
            import traceback
            print(f"❌ MCP 调用异常:\n{traceback.format_exc()}")
            return f"MCP 工具执行失败: {str(e)}"
                    
    @staticmethod
    def run_async_in_thread(coro):
        result = []
        exception = []
        def runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result.append(loop.run_until_complete(coro))
            except Exception as e:
                exception.append(e)
            finally:
                loop.close()
                
        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        
        if exception:
            raise exception[0]
        return result[0] 
    
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("tools_used", [])
        print("INFO_QUERY_AGENT_VERSION_20260402")
        """处理智慧城市信息查询（集成 MCP 天气工具）"""
        
        customer_query = state["customer_query"]
        conversation_context = state.get("conversation_history", "")

        # 1. 定义发送给 LLM 的工具描述 (OpenAI 格式)
        weather_tool_schema = {
            "type": "function",
            "function": {
                "name": "get_realtime_weather",
                "description": "查询指定城市的实时天气和空气质量。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市名称，如 '北京' 或 'beijing'"}
                    },
                    "required": ["location"]
                }
            }
        }

        # 2. 构建初始消息列表
        base_system_prompt = f"你是{self.name}，专门负责{self.role}。请注意：严格基于工具返回的数据回答，不要编造气象细节"
        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)
        
        messages = [SystemMessage(content=system_prompt)]
        if conversation_context:
            messages.append(SystemMessage(content=f"对话历史：\n{conversation_context}"))
        
        # 融合本地匹配的知识库信息
        matched_info = self._match_city_info(customer_query)
        user_input = f"城市服务参考信息：\n{matched_info}\n\n当前查询：{customer_query}" if matched_info else customer_query
        messages.append(HumanMessage(content=user_input))

        try:
            # 3. 第一轮调用：LLM 判断是否需要查天气
            # 注意：此处要求你的 OpenAICompatibleClient.invoke 支持 tools 参数
            response = self.llm.invoke(messages, tools=[weather_tool_schema])
            # 4. 检查是否有工具调用请求
            # 这里的判断逻辑取决于你 OpenAICompatibleClient 的返回类型
            # 假设它返回的是 AIMessage 且包含 tool_calls
            if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
                tool_calls = response.additional_kwargs["tool_calls"]
                state["tools_used"].append("mcp_weather_tool_triggered")

                # 将 LLM 的决策加入上下文
                messages.append(AIMessage(content="", additional_kwargs={"tool_calls": tool_calls}))

                # 执行工具调用
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "get_realtime_weather":
                        args = json.loads(tool_call["function"]["arguments"])
                        city = args.get("location", "北京")
                        
                      
                        weather_result = InfoQueryAgent.run_async_in_thread(
                                        self._call_mcp_weather_tool(city)
                                                                            )

                        # 将工具结果反馈给 LLM
                        messages.append(ToolMessage(
                            content=weather_result,
                            tool_call_id=tool_call["id"]
                        ))

                # 5. 第二轮调用：LLM 根据天气结果生成最终回复
                final_response = self.llm.invoke(messages)
                response_content = final_response.content
            else:
                # 如果不需要工具调用，直接使用第一轮的回复
                response_content = response.content

        except Exception as e:
            print(f"InfoQueryAgent 运行出错: {e}")
            response_content = "抱歉，我在查询实时天气或处理信息时遇到了技术问题。"

        # 6. 更新状态
        state["response"] = response_content
        state["current_agent"] = self.name
        return state

    def _match_city_info(self, query: str) -> str:
        """匹配查询中的城市信息类型（更稳的版本）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        # 每个类别对应更贴近业务的关键词
        category_keywords = {
            "城市服务": ["天气", "空气质量", "路况", "公交", "地铁", "停车场", "公厕", "充电桩"],
            "政务服务": ["办事", "政务", "行政", "审批", "窗口", "材料", "流程"],
            "文旅活动": ["活动", "展览", "演出", "推荐", "导航", "附近", "周边", "场馆", "景点", "旅游"]
        }

        # 先做相对精确的匹配
        for category, policies in self.city_info_knowledge_base.items():
            keywords = category_keywords.get(category, [category])
            if any(kw.lower() in query_lower for kw in keywords):
                if category not in seen_categories:
                    seen_categories.add(category)

                    info_text = f"""【{category}】
    """
                    for policy, description in policies.items():
                        info_text += f"• {policy}：{description}\n"
                    matched_info.append(info_text.strip())

        # 如果没有精确匹配，再做宽泛兜底匹配
        if not matched_info:
            fallback_keywords = [
                "天气", "空气质量", "路况", "公交", "地铁", "停车场", "公厕", "充电桩",
                "办事", "政务", "活动", "展览", "演出", "推荐", "导航", "附近", "周边", "场馆"
            ]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.city_info_knowledge_base.items():
                    if category not in seen_categories:
                        seen_categories.add(category)

                        info_text = f"""相关服务：{category}
    """
                        for i, (policy, description) in enumerate(policies.items()):
                            if i < 2:
                                info_text += f"• {policy}：{description}\n"
                        info_text += "..."
                        matched_info.append(info_text.strip())

        return "\n\n".join(matched_info) if matched_info else ""