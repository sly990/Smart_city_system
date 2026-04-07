"""
智慧城市评价体系指导智能体
专门负责建立评分标准、权重分配、成果衡量、定制化评价方法等问题处理
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent
from rag.retriever import hybrid_retrieve, load_vector_store, connect_es
from rag.reranker import rerank_docs

vector_store = load_vector_store()
es_client = connect_es()
class EvaluationAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市评价体系专家",
            role="智慧城市评价体系指导",
            expertise=["建立评分标准", "权重分配", "成果衡量", "定制化评价方法"],
            
        )

        # TODO: 评价体系知识应该从专业智库系统获取，这里只是模拟数据
        # 实际应用中应该连接评价指标库或调用专业API服务
        self.evaluation_database = {
            "评分标准": {
                "基础设施": "包括网络覆盖、感知终端密度、计算资源等，满分25分",
                "智慧应用": "涵盖政务、交通、环保等领域系统成熟度，满分30分",
                "成效体验": "市民满意度、企业获得感、政府效能提升，满分25分",
                "安全保障": "网络安全、数据安全、隐私保护水平，满分20分"
            },
            "权重分配": {
                "层次分析法(AHP)": "通过专家打分构建判断矩阵，计算各指标权重",
                "熵权法": "基于数据离散程度客观赋权，减少主观偏差",
                "德尔菲法": "多轮专家咨询，形成共识性权重方案",
                "组合赋权": "主观+客观结合，兼顾专业判断与数据特征"
            },
            "成果衡量": {
                "年度对比": "建立基准年数据，逐年对比进步幅度",
                "标杆对标": "选取先进城市进行横向比较，定位差距",
                "目标完成率": "对照规划目标，计算各项指标的完成比例",
                "效益评估": "量化投资回报率、碳减排量、时间节约等价值"
            },
            "定制化评价方法": {
                "智慧政务": "重点考察一网通办覆盖率、数据共享率、在线服务深度",
                "智慧社区": "侧重安防智能、物业数字化、适老化服务、居民参与度",
                "智慧交通": "关注拥堵指数下降、公交准点率、信号灯优化水平",
                "智慧环保": "聚焦空气质量、水质监测、垃圾分类、能源消耗"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理评价体系相关查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")
        
       # --- 核心修改：在 Agent 内部执行 RAG 检索 ---
        rag_context = ""
        if customer_query and len(customer_query) >= 2:
            print(f"🔍 {self.name} 正在为查询检索知识库: {customer_query[:20]}...")
            
            # 1. 混合检索[cite: 3]
            initial_docs = hybrid_retrieve(
                query=customer_query,
                vector_store=vector_store,
                es=es_client,
                k=15
            )
            
            # 2. 重排[cite: 3]
            if initial_docs:
                reranked_docs = rerank_docs(query=customer_query, docs=initial_docs, top_k=5)
                rag_context = "\n\n".join([
                    f"--- 资料片段 [{i+1}] ---\n{doc.page_content}" 
                    for i, doc in enumerate(reranked_docs)
                ])
                # 将结果存入 state，方便后续工具审计[cite: 3]
                state["documents"] = reranked_docs
                state["tools_used"].append(f"{self.name}_rag_retrieval")
            else:
                rag_context = "未找到相关背景资料。"
        
        # 将最新的检索结果赋值给 state 的 context[cite: 1]
        state["context"] = rag_context
        # ---------------------------------------



        # --- 修改：直接从 state 获取历史，而不是重新查库 ---
        conversation_context = state.get("conversation_history", "")
        
        # 从评价数据库中匹配相关信息
        matched_info = self._match_evaluation_info(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的评价体系问题提供专业的解答：
        1. 仔细分析客户的具体问题（评分标准、权重分配、成果衡量或定制化评价方法）
        2. 提供明确的指标框架、赋权方法和衡量周期建议
        3. 说明不同领域的评价差异和定制化要点
        4. 如果问题复杂，建议开展专家研讨会或委托专业评估机构

        回答要准确、专业，涉及指标、权重和评价方法的信息要具体明确。如果问题超出你的权限范围，请说明并建议转接给城市绩效评估总师。"""

        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据以下提供的参考资料（知识库）和你的专业背景来回答:
        1. 仔细分析客户的具体问题（如监测异常、数据调度、安全事件、风险预警等）
        2. 提供明确的处置流程和时间预期
        3. 说明需要准备的材料或协调的部门
        4. 如果问题复杂或涉及重大风险，建议升级处理或联系安全运维中心

        回答要准确、专业，涉及指标、流程、时间的信息要具体明确。如果问题超出你的权限范围，请说明并建议转接给相关安全专家。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构造发送给 LLM 的内容
        # 我们将 RAG 上下文、模拟数据、对话历史整合在一起
        knowledge_bundle = ""
        if rag_context:
            knowledge_bundle += f"【核心知识库资料】:\n{rag_context}\n\n"
        if matched_info:
            knowledge_bundle += f"【专家参考信息】:\n{matched_info}\n\n"

        messages = [
            SystemMessage(content=system_prompt)
        ]

        # 注入上下文和历史
        combined_query = f"""
对话历史：
{conversation_context if conversation_context else "暂无对话历史"}

已知参考资料：
{knowledge_bundle if knowledge_bundle else "未找到直接相关的文档资料，请基于专业知识回答。"}

用户当前问题：{customer_query}
"""
        messages.append(HumanMessage(content=combined_query))

        # 调用LLM
        try:
            # 这里的 self.llm 是你在 multi_agent_customer_service.py 中注入的 OpenAICompatibleClient
            response = self.llm.invoke(messages)
            response_content = response.content
        except Exception as e:
            print(f"❌ {self.name} 调用LLM出错: {e}")
            response_content = "抱歉，由于规划知识库系统连接超时，我暂时无法给出详细方案，请您稍后再试。"

        # 更新会话和状态

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_rag_processed")

        return state

    def _match_evaluation_info(self, query: str) -> str:
        """匹配查询中的评价体系信息（更稳的版本）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        # 不同类别对应更有针对性的关键词
        category_keywords = {
            "评价指标体系": ["评分", "权重", "指标", "衡量", "评价", "考核", "打分", "量化"],
            "指标定制": ["定制", "自定义", "定制化", "指标体系", "评价体系", "个性化"],
            "标准对标": ["标准", "对标", "规范", "基准", "参照", "标杆"],
            "结果分析": ["结果", "分析", "评估", "报告", "结论", "反馈", "统计"]
        }

        # 先做相对精确的匹配
        for category, policies in self.evaluation_database.items():
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
            fallback_keywords = ["评分", "权重", "衡量", "评价", "指标", "定制", "标准", "考核", "打分", "评估"]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.evaluation_database.items():
                    if category not in seen_categories:
                        seen_categories.add(category)

                        info_text = f"""相关领域：{category}
    """
                        for i, (policy, description) in enumerate(policies.items()):
                            if i < 2:
                                info_text += f"• {policy}：{description}\n"
                        info_text += "..."
                        matched_info.append(info_text.strip())

        return "\n\n".join(matched_info) if matched_info else ""