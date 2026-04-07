"""
智慧城市综合智慧服务智能体
负责解答不涉及智慧城市相关信息的一般性通用问题，精准识别并转接具体的业务需求
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent


class SmartServiceAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市综合智慧服务专家",
            role="解答不涉及城市有关的问题的其他通用问题，精准识别并转接具体的业务需求",
            expertise=["通用问答", "常识解答", "生活服务", "信息查询", "基础计算", "日常帮助"],
            
        )

        # TODO: 通用知识应从知识图谱或公开API获取，这里只是模拟数据
        # 实际应用中可连接维基百科、天气API、新闻API等服务
        self.general_knowledge_base = {
            
            "日期时间常识": {
                "当前时间": "可根据用户所在时区提供参考，默认北京时间（UTC+8）",
                "节假日": "包含国家法定节假日（元旦、春节、清明、劳动节、端午、中秋、国庆）及调休安排",
                "计算功能": "日期加减、两个日期之间的天数差、星期几推算（不依赖城市）",
                "时区知识": "介绍全球24个时区的基本概念，不涉及具体城市转换"
            },
            "数学计算": {
                "基础运算": "加减乘除、百分比、幂运算",
                "科学计算": "三角函数、对数、指数、开方",
                "单位换算": "长度、重量、面积、体积、温度、货币汇率",
                "表达式解析": "支持括号和多步运算"
            },
            "常识百科": {
                "知识范围": "历史、地理、文化、科学、艺术、体育等常见事实知识",
                "信息验证": "多源比对，优先采用权威来源（如百度百科、维基百科）",
                "不确定时": "明确告知用户信息存疑，建议核实",
                "不宜回答": "涉及敏感、违法、个人隐私或未经验证的信息拒绝回答"
            },
            "生活服务": {
                "快递查询": "支持主流快递公司单号查询，提供物流轨迹",
                "票务信息": "火车票、飞机票、电影票的查询与比价提示",
                "餐饮指南": "推荐餐厅、菜谱做法、饮食禁忌",
                "健康建议": "一般性健康知识（非医疗诊断），如运动、睡眠、营养"
            },
            "新闻资讯": {
                "热点新闻": "提供近期国内外的热点事件摘要",
                "分类新闻": "科技、体育、财经、娱乐等分类浏览",
                "获取方式": "可指定关键词或时间范围",
                "免责声明": "新闻来源为公开媒体，不代表本智能体观点"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用问题（不涉及智慧城市）"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")
        
        

        # --- 修改：直接从 state 获取历史，而不是重新查库 ---
        conversation_context = state.get("conversation_history", "")

        # 从通用知识库中匹配相关信息
        matched_info = self._match_general_info(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的通用问题提供专业、友好的解答：
        1. 对于非智慧城市的通用问题，提供准确、简洁的信息
        2. 对于无法回答的问题，诚实告知并给出替代建议

        回答要清晰、有用，避免过度承诺。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的解答。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的通用信息，添加到上下文中
        if matched_info:
            general_context = f"""通用知识信息：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=general_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM
        try:
            response = self.llm.invoke(messages)
            response_content = response.content
        except Exception as e:
            print(f"智慧城市综合智慧服务专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的问题时遇到系统错误，请稍后重试。"



        # 更新状态
        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_general_info(self, query: str) -> str:
        """匹配查询中的通用信息（更稳的版本）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        category_keywords = {
            "时间日期": ["日期", "时间", "节假日", "日历", "今天", "明天", "周几"],
            "计算换算": ["计算", "数学", "单位", "换算", "公式", "数值", "金额"],
            "生活资讯": ["快递", "票务", "餐饮", "健康", "新闻", "资讯", "百科", "常识"]
        }

        # 先做相对精确的匹配
        for category, policies in self.general_knowledge_base.items():
            keywords = category_keywords.get(category, [category])
            if any(kw.lower() in query_lower for kw in keywords):
                if category not in seen_categories:
                    seen_categories.add(category)

                    info_text = f"""【{category}】
    """
                    for policy, description in policies.items():
                        info_text += f"• {policy}：{description}\n"
                    matched_info.append(info_text.strip())

        # 再做宽泛兜底匹配
        if not matched_info:
            fallback_keywords = [
               "日期", "时间", "计算", "数学", "单位", "百科", "常识",
                "快递", "票务", "餐饮", "健康", "新闻", "资讯", "换算", "节假日"
            ]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.general_knowledge_base.items():
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