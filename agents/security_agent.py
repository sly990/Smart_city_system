"""
智慧城市运营与安全指导智能体
负责网络设备监测、数据资源整合、运维协调、安全防护、风险预警、事件处置及策略优化
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent
from rag.retriever import hybrid_retrieve, load_vector_store, connect_es
from rag.reranker import rerank_docs

vector_store = load_vector_store()
es_client = connect_es()

class SecurityAgent(BaseAgent):
    def __init__(self,session_manager=None):
        super().__init__(
            name="智慧城市运营与安全指导专家",
            role="网络设备与系统监测、数据资源整合调度、日常运维协调、安全体系构建、风险预警、安全事件处置、运营与安全策略动态优化",
            expertise=["网络设备监测", "数据资源整合", "运维协调", "安全防护", "风险预警", "安全事件处置", "策略优化"],
           
        )

        # TODO: 安全与运维知识应从实际系统获取，这里只是模拟数据
        # 实际应用中应连接安全运维数据库或调用相关API服务
        self.security_knowledge_base = {
            "网络设备与系统监测": {
                "监测范围": "涵盖路由器、交换机、服务器、存储设备、物联网终端等关键设备",
                "监测指标": "CPU使用率、内存占用、网络流量、连接数、错误日志、设备可用性",
                "告警机制": "异常指标超过阈值时自动告警，支持短信、邮件、平台通知",
                "巡检周期": "每日自动巡检，每周生成设备健康报告"
            },
            "数据资源整合调度": {
                "数据来源": "交通、安防、能源、环境、政务等多源城市数据",
                "整合方式": "数据湖+数据仓库混合架构，实时流处理与批量处理结合",
                "调度策略": "优先级调度、负载均衡、故障转移，保障关键业务数据优先",
                "资源目录": "提供统一数据资源目录，支持跨系统查询与订阅"
            },
            "日常运维协调": {
                "运维流程": "事件申报 → 分派 → 处理 → 验证 → 关闭",
                "协调机制": "跨部门运维联席会议，日常沟通群，7×24小时值班制",
                "运维工具": "集中监控平台、自动化运维平台（Ansible）、日志分析系统",
                "文档管理": "运维手册、应急预案、变更记录统一归档"
            },
            "安全体系构建": {
                "安全框架": "等保2.0三级+ISO 27001融合架构",
                "技术防护": "下一代防火墙、入侵检测/防御、端点检测与响应、数据加密",
                "管理策略": "最小权限原则、定期安全审计、人员安全培训",
                "合规要求": "满足网络安全法、数据安全法、个人信息保护法"
            },
            "风险预警": {
                "预警等级": "红（严重）、橙（高危）、黄（中危）、蓝（低危）",
                "预警来源": "漏洞扫描、威胁情报、异常行为分析、态势感知平台",
                "发布渠道": "短信、邮件、政务钉钉、大屏弹窗",
                "响应时限": "红色预警5分钟内响应，橙色15分钟，黄色30分钟"
            },
            "安全事件处置": {
                "事件类型": "网络攻击、数据泄露、勒索病毒、物理入侵、系统瘫痪",
                "处置流程": "隔离 → 取证 → 分析 → 清除 → 恢复 → 复盘",
                "应急小组": "技术处置组、溯源取证组、对外联络组、后勤保障组",
                "报告要求": "事件发生后2小时内提交初报，24小时内提交详细报告"
            },
            "运营与安全策略动态优化": {
                "优化依据": "运行态势数据、安全事件统计、风险评估结果、成本效益分析",
                "调整频率": "每月一次常规优化，重大变更或事件后即时优化",
                "优化方法": "A/B测试、模拟推演、灰度发布",
                "效果评估": "关键性能指标（KPI）与安全指标（KRI）双维度评估"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理智慧城市运营与安全相关查询"""
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

        # 从安全知识库中匹配相关信息
        matched_info = self._match_security_info(customer_query)

        # 构建系统提示并增强对话上下文说明
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


        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_rag_processed")

        return state

    def _match_security_info(self, query: str) -> str:
        """匹配查询中的安全与运维相关信息（更稳的版本）"""
        query_lower = query.lower()
        matched_info = []
        seen_categories = set()

        # 每个类别对应更贴近业务的关键词
        category_keywords = {
            "网络安全": ["网络", "防火墙", "入侵", "攻击", "漏洞", "等保", "安全", "防护"],
            "设备运维": ["设备", "巡检", "告警", "运维", "调度", "监测", "资源", "优化"],
            "事件处置": ["事件", "处置", "应急", "预警", "风险", "响应", "策略"],
            "数据安全": ["数据", "合规", "风险", "防护", "安全", "隐私", "访问控制"]
        }

        # 先做相对精确的匹配
        for category, policies in self.security_knowledge_base.items():
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
                "网络", "设备", "监测", "数据", "资源", "整合", "运维", "协调", "安全", "防护",
                "风险", "预警", "事件", "处置", "策略", "优化", "漏洞", "攻击", "入侵", "防火墙",
                "巡检", "告警", "调度", "应急", "等保", "合规"
            ]

            if any(kw in query_lower for kw in fallback_keywords):
                for category, policies in self.security_knowledge_base.items():
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