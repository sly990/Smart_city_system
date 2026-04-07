# 智慧城市全域智能协同系统



## 项目概述

智慧城市全域智能协同系统是一个基于 LangGraph 构建的多智能体协同平台，专为城市管理者提供从规划设计、
合规落地、安全运维到便民服务的全流程智能化支撑。系统采用模块化架构，
通过六大专业智能体（城市规划、评价体系、安全运维、合规落地、综合服务、信息查询）覆盖智慧城市建设的核心业务场景，
并深度融合 RAG 知识库与 MCP 工具适配能力，实现跨部门、跨系统的数据与业务协同。





## 项目结构
```
customer-service-ai-agent/
├── agents/ # 智能体模块
│ ├── init.py # 智能体包初始化
│ ├── base_agent.py # 基础智能体类
│ ├── urban_plan_agent.py      # 城市规划设计咨询智能体
│ ├── evaluation_agent.py      # 评价体系指导智能体
│ ├── security_agent.py        # 安全运维智能体
│ ├── compliance_agent.py      # 合规落地智能体
│ ├── smart_service_agent.py   # 综合服务智能体
│ └── info_query_agent.py      # 信息查询与便民服务智能体
├── rag/ # rag相关函数文件夹
├── tools/ # 工具函数模块
│ ├── init.py # 工具包初始化
│ └── query_tools.py # 查询分类工具
├── templates/ # Web界面模板
│ └── index.html # 主页面HTML模板
├── config.py # 基础配置文件
├── multi_agent_customer_service.py # 主程序文件（LangGraph工作流）
├── session_manager.py # 会话管理器（LangChain标准接口）
├── web_app.py # Web应用（Flask + LangGraph API）
├── langgraph.json # LangGraph工作流配置
├── requirements.txt # 项目依赖
├── .env # 环境变量配置
├── README.md # 项目说明文档
└── README_LangGraph_CLI.md # LangGraph CLI使用说明
```


## 主要特性

### 六大专业智能体，覆盖智慧城市核心场景
- **城市规划设计咨询智能体**：	提供发展蓝图、技术选型、建设步骤、跨领域协同建议
- **评价体系指导智能体**：	建立评分标准、权重分配、成果衡量、定制化评价方法
- **安全运维智能体**：	网络设备与系统监测、数据资源整合、风险预警、安全事件处置、策略动态优化
- **合规落地智能体**：	匹配建设标准、落实合规要求、划分责任、保障工程推进
- **综合服务智能体**：	解答通用问题，处理非专业领域咨询
- **信息查询与便民服务智能体**：	查询天气/交通/设施/办事指南，提供个性化推荐

### 基于 LangGraph 的多 Agent 协同架构
- **动态路由**：根据用户意图自动将任务路由至对应专业智能体
- **任务拆解与协作**：复杂任务可拆分为子任务，由多个智能体协同完成




### RAG 知识库构建与优化
- **多源数据整合**：汇聚国家标准、政策文件
- **向量数据库**：采用 FAISS实现语义检索，支持混合检索（关键词 + 语义 + 多模态特征）
- **重排序机制**：使用 Cross-Encoder 提升检索准确率



### MCP 工具适配集成
- **统一工具调用**：基于 langchain-mcp-adapters 对接天气等现有业务系统数据
- **多协议支持**：支持 stdio 和 streamable-http 传输协议



### 智能对话与上下文管理
多会话并发，每个会话独立管理历史记录。
基于 LangChain 记忆组件，支持长期对话记忆。
智能体结合历史上下文生成连贯、个性化的回答。


### 并行专家分析与评审
- **编排者智能体**：自动识别复杂问题，动态确定需要并行调用的专家领域（如城市规划、评价体系、安全运维、合规落地等）
- **并行执行引擎**：基于 LangGraph 的异步并发能力，同时调用多个专业智能体，各自独立生成分析报告，显著缩短总体响应时间
- **评审者智能体**：综合多份专家报告，智能识别冲突点（例如规划方案与安全要求矛盾），提出具体修改意见，并输出整合后的最终方案
- **价值体现**：实现“多人会诊”式的协同决策，提升跨领域问题的解决质量与效率，特别适用于智慧城市顶层设计、重大项目评审等复杂场景


## 安装与配置


### 环境要求
Python 3.9+



### 安装依赖
```bash
pip install -r requirements.txt
```


### 进入 elasticsearch 的 bin 目录
```
python G:\Agent\customer-service-ai-agent2\rag\delete.py
cd G:\Agent\customer-service-ai-agent2\elasticsearch-9.0.1-windows-x86_64\elasticsearch-9.0.1\bin
.\elasticsearch.bat
```



### LLM 模型配置（推荐使用硅基流动或其他 OpenAI 兼容服务）
```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 初始化知识库
```bash
python G:\Agent\customer-service-ai-agent2\rag\ingest.py
```


### 启动系统
```
python multi_agent_customer_service.py
python G:\Agent\customer-service-ai-agent2\weather_mcp_server.py
```

### 终端 1：启动 LangGraph 服务
```bash
langgraph dev
```

### 终端 2：启动 Flask Web 应用
```bash
python web_app.py
```

访问 http://localhost:5000 使用聊天界面。
