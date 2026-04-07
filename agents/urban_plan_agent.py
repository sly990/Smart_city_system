"""
智慧城市规划设计咨询智能体
专门负责发展蓝图、技术选型、建设步骤、跨领域协同等问题处理
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent
from rag.retriever import hybrid_retrieve, load_vector_store, connect_es
from rag.reranker import rerank_docs

vector_store = load_vector_store()
es_client = connect_es()

class UrbanPlanAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市规划设计专家",
            role="智慧城市规划设计咨询",
            expertise=["发展蓝图", "技术选型", "建设步骤", "跨领域协同"],
           
        )

        # TODO: 城市规划知识应该从专业智库系统获取，这里只是模拟数据
        # 实际应用中应该连接城市规划知识库或调用专业API服务
        self.urban_plan_database = {
            "发展蓝图": {
                "愿景规划": "制定城市未来10-15年发展愿景，包括经济、社会、环境三维目标",
                "战略定位": "明确城市在区域中的功能定位与核心竞争力",
                "空间布局": "划分功能分区，优化居住、商业、生态用地比例",
                "时序安排": "分近期（1-3年）、中期（4-7年）、远期（8年以上）推进"
            },
            "技术选型": {
                "感知层技术": "推荐部署5G+物联网传感器，覆盖交通、环保、安防等场景",
                "数据中台": "采用云计算+大数据架构，支持跨部门数据共享与治理",
                "AI平台": "建议使用主流深度学习框架，实现视频分析、预测预警等功能",
                "数字孪生": "基于BIM/CIM技术构建城市数字孪生底座，支持模拟推演"
            },
            "建设步骤": {
                "顶层设计阶段": "完成需求调研、可行性分析、总体方案设计（约3个月）",
                "试点建设阶段": "选择核心片区先行示范，验证技术方案（约6个月）",
                "全面推广阶段": "按优先级分批推进，覆盖全市主要领域（约1-2年）",
                "优化运营阶段": "建立持续运维机制，根据数据反馈迭代升级"
            },
            "跨领域协同": {
                "交通与安防协同": "共享视频监控与车流数据，实现应急联动",
                "环保与能源协同": "监测污染源与能耗数据，优化低碳调度",
                "数据标准统一": "制定跨部门数据接口规范，避免信息孤岛",
                "项目统筹机制": "建立联席会议制度，统一审批避免重复建设"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理城市规划设计相关查询"""
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

        # 从城市规划数据库中匹配相关信息
        matched_info = self._match_urban_plan_info(customer_query)

        # 构建增强后的系统提示
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
            你的专业领域包括：{', '.join(self.expertise)}

            请根据以下提供的参考资料（知识库）和你的专业背景来回答。
            1. 如果参考资料中有具体案例、技术标准或规划步骤，请优先引用。
            2. 回答要具体、专业，涉及时间表和技术选型时要给出明确建议。
            3. 如果参考资料不足以回答，请结合智慧城市规划的通用标准（如GB/T系列标准）进行解答。
            4. 说明不同领域（如交通与安防）的协同要点。"""

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

    def _match_urban_plan_info(self, query: str) -> str:
        """匹配查询中的城市规划信息（更稳的版本）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        category_keywords = {
            "总体规划": ["蓝图", "规划", "总体", "顶层设计", "框架", "布局"],
            "建设方案": ["建设", "方案", "实施", "步骤", "推进", "落地"],
            "技术架构": ["技术", "架构", "系统", "平台", "协同", "集成"],
            "协同机制": ["协同", "联动", "配合", "机制", "部门", "治理"]
        }

        # 先做相对精确的匹配
        for category, policies in self.urban_plan_database.items():
            keywords = category_keywords.get(category, [category])
            if any(kw.lower() in query_lower for kw in keywords):
                if category not in seen_categories:
                    seen_categories.add(category)

                    info_text = f"""【{category}】"""
                    for policy, description in policies.items():
                        info_text += f"• {policy}：{description}\n"
                    matched_info.append(info_text.strip())

        # 再做宽泛兜底匹配
        if not matched_info:
            fallback_keywords = ["蓝图", "技术", "步骤", "协同", "规划", "建设", "方案"]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.urban_plan_database.items():
                    if category not in seen_categories:
                        seen_categories.add(category)

                        info_text = f"""相关领域：{category} """
                        for i, (policy, description) in enumerate(policies.items()):
                            if i < 2:
                                info_text += f"• {policy}：{description}\n"
                        info_text += "..."
                        matched_info.append(info_text.strip())

        return "\n\n".join(matched_info) if matched_info else ""