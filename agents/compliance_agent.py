"""
智慧城市建设合规落地智能体
负责匹配建设标准、落实合规要求、划分责任、保障工程推进
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent
from rag.retriever import hybrid_retrieve, load_vector_store, connect_es
from rag.reranker import rerank_docs

vector_store = load_vector_store()
es_client = connect_es()

class ComplianceAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市建设合规落地专家",
            role="确保建设过程合法合规、不跑偏，帮助项目精准匹配各项建设标准，将合规要求落实到每一步，清晰划分各参与方的责任，从而保障工程顺利推进",
            expertise=["匹配建设标准", "落实合规要求", "划分责任", "保障工程推进"],
            
        )

        # TODO: 合规知识应从实际标准库或法规系统获取，这里只是模拟数据
        # 实际应用中应连接标准数据库或调用合规API服务
        self.compliance_knowledge_base = {
            "建设标准匹配": {
                "国家标准": "GB/T 34678-2017《智慧城市技术参考模型》、GB/T 36333-2018《智慧城市顶层设计指南》",
                "行业标准": "CJJ/T 312-2021《城市运行管理服务平台技术标准》、YD/T 3865-2021《物联网安全技术要求》",
                "地方标准": "依据项目所在省市发布的智慧城市相关建设规范执行",
                "匹配流程": "需求分析 → 标准检索 → 差异比对 → 标准适配 → 合规确认"
            },
            "合规要求落实": {
                "合规清单": "包含政策合规、数据安全合规、环保合规、招投标合规、工程验收合规等",
                "落实机制": "逐项分解 → 责任到岗 → 节点检查 → 闭环管理",
                "文档要求": "合规自评报告、第三方合规审查意见、过程留痕记录",
                "数字化工具": "合规管理平台、智能检查表、合规风险预警模块"
            },
            "责任划分": {
                "参与方角色": "建设方、设计方、施工方、监理方、审计方、运营方",
                "责任矩阵": "RACI模型（负责、批准、咨询、知情），明确每项任务的权责",
                "合同约束": "在招标文件和合同中嵌入合规责任条款及违约处罚措施",
                "争议解决": "设立合规争议协调小组，定期召开合规例会"
            },
            "工程推进保障": {
                "里程碑管控": "立项、设计、招标、施工、验收、运营各阶段合规审查节点",
                "进度与合规双线": "进度滞后需重新评估合规风险，合规问题未解决不得进入下一阶段",
                "保障机制": "合规专班驻场、合规月报制度、合规问题红黄绿灯预警",
                "应急处理": "出现合规偏差时启动纠偏程序，必要时暂停施工进行合规整改"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理智慧城市建设合规相关查询"""
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

        # 从合规知识库中匹配相关信息
        matched_info = self._match_compliance_info(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据以下提供的参考资料（知识库）和你的专业背景来回答:
        1. 仔细分析客户的具体问题（如标准匹配、合规要求落实、责任划分、工程推进保障等）
        2. 提供明确的合规操作流程和关键节点
        3. 说明需要哪些参与方协同以及所需材料
        4. 如果问题复杂或涉及重大合规风险，建议升级处理或转接法律合规部门

        回答要准确、专业，涉及标准编号、条款、流程的信息要具体明确。如果问题超出你的权限范围，请说明并建议转接给合规专家。"""

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


    def _match_compliance_info(self, query: str) -> str:
        """匹配查询中的合规相关信息（更稳的版本，避免 split 越界）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        # 每个类别对应一组更有针对性的关键词
        category_keywords = {
            "建设标准匹配": ["标准", "国标", "行标", "地标", "匹配", "规范", "参考模型", "顶层设计"],
            "合规要求落实": ["合规", "落实", "清单", "审查", "整改", "风险", "留痕", "文档", "自评"],
            "责任划分": ["责任", "raci", "合同", "约束", "争议", "建设方", "施工方", "监理方", "运营方"],
            "工程推进保障": ["工程", "推进", "里程碑", "验收", "进度", "暂停", "应急", "施工", "运营"]
        }

        # 精确/半精确匹配
        for category, policies in self.compliance_knowledge_base.items():
            keywords = category_keywords.get(category, [category])
            if any(kw.lower() in query_lower for kw in keywords):
                if category not in seen_categories:
                    seen_categories.add(category)

                    info_text = f"""【{category}】
    """
                    for policy, description in policies.items():
                        info_text += f"• {policy}：{description}\n"
                    matched_info.append(info_text.strip())

        # 如果还没有匹配到，再做更宽泛的兜底匹配
        if not matched_info:
            fallback_keywords = [
                "标准", "合规", "责任", "工程", "推进", "验收", "招标", "合同",
                "法规", "国标", "行标", "地标", "raci", "里程碑", "审查", "整改", "风险", "约束"
            ]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.compliance_knowledge_base.items():
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