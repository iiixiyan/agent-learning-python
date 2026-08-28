# LangFuse：开源可内网部署的 Agent 全链路监测方案

> **Python 版** | 基于 LangFuse + LangChain + Python 技术栈
> 前置知识：[LangChain 基础](../01-第一阶段-Agent基础入门/03_LangChain基础.md)、[Agent 基础](../01-第一阶段-Agent基础入门/08_Agent基础.md)

---

## 为什么需要 LangFuse？

Agent 系统复杂，调用链很长（Prompt → LLM → Tool → LLM → ...），没有监测工具会导致：

| 问题 | 说明 |
|------|------|
| **成本不可控** | 不知道每次调用花了多少钱、多少 token |
| **问题难定位** | 出问题无法定位是哪一步出错 |
| **效果难评估** | 无法评估 Prompt 改动的效果 |
| **优化无依据** | 没有数据支撑优化决策 |
| **用户无反馈** | 不知道用户对回答是否满意 |

**LangFuse** 是开源的 LLM 应用观测平台，支持私有化部署，解决以上所有问题。

### 可观测性架构

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   Agent 应用                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Prompt  │───▶│   LLM    │───▶│   Tool   │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│         │               │               │             │
│         └───────────────┼───────────────┘             │
│                         ▼                              │
│              ┌──────────────────┐                     │
│              │  LangFuse SDK    │  ← 自动/手动采集   │
│              └────────┬─────────┘                     │
└───────────────────────┼───────────────────────────────┘
                        │ HTTP API
                        ▼
┌─────────────────────────────────────────────────────┐
│              LangFuse 服务（Docker 部署）             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Trace    │  │ 成本统计 │  │ Prompt 管理      │  │
│  │ 追踪     │  │          │  │                  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 评分反馈 │  │ 数据集   │  │ 评估对比         │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
              PostgreSQL（数据存储）
```

---

## LangSmith vs LangFuse

| 对比维度 | LangSmith | LangFuse |
|----------|-----------|----------|
| **开源** | ❌ 闭源 SaaS | ✅ 开源可自建 |
| **部署方式** | 只能用官方云 | Docker 一键自建 |
| **数据安全** | 数据在官方服务器 | 数据在自己服务器 |
| **成本** | 按调用量收费 | 免费（自己出服务器费） |
| **功能完善度** | 非常完善 | 核心功能齐全，持续更新 |
| **社区生态** | 成熟 | 活跃，快速增长 |
| **适用场景** | 个人/小团队快速上手 | 企业内网/数据敏感场景 |

**企业内网/数据敏感场景选 LangFuse，个人快速验证选 LangSmith。**

---

## Docker 部署

### docker-compose.yml

```yaml
version: '3.8'

services:
  langfuse:
    image: langfuse/langfuse:latest
    container_name: langfuse
    restart: always
    environment:
      # 数据库连接
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/langfuse
      # 认证密钥（生产环境务必修改）
      - NEXTAUTH_SECRET=your-secret-key-change-in-production-32chars
      - NEXTAUTH_URL=http://localhost:3000
      # 关闭遥测
      - TELEMETRY_ENABLED=false
      # 环境变量
      - NODE_ENV=production
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: postgres:16-alpine
    container_name: langfuse-postgres
    restart: always
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=langfuse
      - POSTGRES_USER=postgres
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pg_data:
```

### 启动和初始化

```bash
# 1. 启动服务
docker compose up -d

# 2. 等待启动完成（约30秒）
docker compose logs -f langfuse

# 3. 访问界面
# 浏览器打开 http://localhost:3000

# 4. 注册账号 → 创建项目 → 获取 API Key
# Settings → API Keys → Create new API keys
# 会得到：
#   Public Key: pk-lf-xxx
#   Secret Key: sk-lf-xxx
```

---

## Python 集成

### 安装依赖

```bash
pip install langfuse langchain langchain-openai python-dotenv
```

### 完整示例

创建 `langfuse_demo.py`：

```python
"""
langfuse_demo.py - LangFuse 完整示例
包含：LangChain 自动追踪 + 手动追踪 + Prompt 管理 + 评分反馈
"""
import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ========== 1. 初始化 LangFuse ==========
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-xxx"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-xxx"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# LangChain 回调处理器（自动追踪）
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-xxx"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-xxx"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# 初始化 LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ========== 2. 方式一：LangChain 自动追踪 ==========
def langchain_auto_trace():
    """
    LangChain 自动追踪
    传入 CallbackHandler，自动记录整个链路（Prompt → LLM → 输出）
    """
    print("\n" + "="*60)
    print("方式一：LangChain 自动追踪")
    print("="*60)

    # 创建 Chain
    prompt = ChatPromptTemplate.from_template("用中文解释：{topic}")
    chain = prompt | llm | StrOutputParser()

    # 调用时传入 handler，自动记录整个链路
    result = chain.invoke(
        {"topic": "RAG检索增强生成"},
        config={"callbacks": [langfuse_handler]}
    )

    print(f"回答: {result}")
    print("\n✅ 已自动记录到 LangFuse，去界面查看 Trace 详情")


# ========== 3. 方式二：手动追踪（更灵活） ==========
def manual_trace_rag(question: str):
    """
    手动追踪 RAG 查询流程
    Trace → Span（检索）→ Span（LLM生成）
    适合自定义流程，不依赖 LangChain
    """
    print("\n" + "="*60)
    print(f"方式二：手动追踪 RAG 查询")
    print(f"问题: {question}")
    print("="*60)

    # 创建 Trace（一次完整请求）
    trace = langfuse.trace(
        name="rag_query",
        metadata={
            "question": question,
            "user_id": "user_001",
            "session_id": "session_001",
        },
        tags=["RAG", "production"],
    )

    # ===== 阶段1：检索 =====
    span_retrieve = trace.span(
        name="retrieve_documents",
        metadata={"top_k": 3, "index": "knowledge_base"},
    )

    # 模拟检索（实际调用向量数据库）
    import time
    time.sleep(0.5)  # 模拟耗时
    retrieved_docs = [
        {"id": "doc_001", "content": "RAG是检索增强生成技术..."},
        {"id": "doc_002", "content": "向量数据库用于存储文本向量..."},
        {"id": "doc_003", "content": "Embedding模型将文本转为向量..."},
    ]

    span_retrieve.end(
        output={
            "docs_count": len(retrieved_docs),
            "doc_ids": [d["id"] for d in retrieved_docs],
        }
    )
    print(f"  📚 检索到 {len(retrieved_docs)} 篇文档")

    # ===== 阶段2：LLM 生成 =====
    span_llm = trace.span(
        name="llm_generate",
        metadata={"model": "gpt-4o-mini", "temperature": 0},
    )

    # 构造 Prompt
    context = "\n".join([d["content"] for d in retrieved_docs])
    prompt_text = f"""根据以下资料回答问题。

资料：
{context}

问题：{question}

回答："""

    # 调用 LLM
    llm_result = llm.invoke(prompt_text)

    span_llm.end(
        output={"answer": llm_result.content},
        usage={
            "input": llm_result.usage_metadata["input_tokens"],
            "output": llm_result.usage_metadata["output_tokens"],
            "total": llm_result.usage_metadata["total_tokens"],
            "unit": "TOKENS",
        }
    )
    print(f"  🤖 LLM 回答: {llm_result.content[:100]}...")

    # 更新 Trace 输出
    trace.update(
        output={"answer": llm_result.content},
        metadata={"total_docs": len(retrieved_docs)},
    )

    # ===== 评分（用户反馈） =====
    trace.score(
        name="relevance",
        value=4,  # 1-5 分
        comment="回答相关但不够详细",
    )
    print("  ⭐ 已评分: 4/5")

    return llm_result.content


# ========== 4. Prompt 管理（从 LangFuse 获取） ==========
def prompt_from_langfuse(topic: str):
    """
    从 LangFuse 获取 Prompt（不用硬编码在代码里）
    运营人员可以在界面修改 Prompt，不用改代码
    """
    print("\n" + "="*60)
    print("方式三：从 LangFuse 获取 Prompt")
    print("="*60)

    try:
        # 从 LangFuse 获取 Prompt（需要先在界面创建）
        prompt_client = langfuse.get_prompt("explain_topic", version=1)

        # 编译 Prompt（替换变量）
        prompt = prompt_client.compile(topic=topic)
        print(f"Prompt: {prompt[:100]}...")

        # 调用 LLM
        result = llm.invoke(prompt)
        print(f"回答: {result.content}")

        return result.content

    except Exception as e:
        print(f"⚠️  获取 Prompt 失败（请先在 LangFuse 界面创建 'explain_topic' Prompt）: {e}")
        # 降级使用本地 Prompt
        prompt = ChatPromptTemplate.from_template("用中文解释：{topic}")
        chain = prompt | llm
        result = chain.invoke({"topic": topic})
        print(f"回答（降级）: {result.content}")
        return result.content


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 方式一：LangChain 自动追踪
    langchain_auto_trace()

    # 方式二：手动追踪 RAG
    manual_trace_rag("什么是RAG？它有什么优势？")

    # 方式三：从 LangFuse 获取 Prompt
    prompt_from_langfuse("Transformer架构")

    # 刷新（确保所有数据发送到 LangFuse）
    langfuse.flush()
    print("\n✅ 所有示例完成！去 LangFuse 界面查看 Trace 详情")
```

### 运行示例

```bash
# 1. 配置 .env
cat > .env << EOF
LANGFUSE_PUBLIC_KEY=pk-lf-你的public-key
LANGFUSE_SECRET_KEY=sk-lf-你的secret-key
LANGFUSE_HOST=http://localhost:3000
OPENAI_API_KEY=你的-openai-api-key
LLM_MODEL=gpt-4o-mini
EOF

# 2. 运行示例
python langfuse_demo.py

# 3. 去 LangFuse 界面查看
# 浏览器打开 http://localhost:3000
# Traces 菜单查看所有 Trace
```

---

## 核心功能详解

### 1. Trace 追踪

每次请求是一个 **Trace**，包含多个 **Span**（阶段），可以看到完整调用链、耗时、token 消耗。

```
Trace: rag_query (总耗时: 2.3s, 总token: 450)
├── Span: retrieve_documents (耗时: 0.5s)
│   └── 输出: 3篇文档
├── Span: llm_generate (耗时: 1.8s)
│   ├── 输入: 300 tokens
│   ├── 输出: 150 tokens
│   └── 模型: gpt-4o-mini
└── Score: relevance = 4/5
```

| 概念 | 说明 |
|------|------|
| **Trace** | 一次完整请求，包含所有阶段 |
| **Span** | 请求中的一个阶段（如检索、LLM调用） |
| **Generation** | LLM 调用（特殊的 Span，包含 token 统计） |
| **Score** | 对 Trace 的评分（用户反馈） |
| **Event** | Trace 中的事件（如错误、日志） |

### 2. 成本统计

自动统计每次调用的 token 数和费用，支持按时间、模型、用户维度分析。

| 分析维度 | 说明 |
|----------|------|
| **按时间** | 每天/每周/每月的 token 消耗和费用 |
| **按模型** | 不同模型的调用次数和费用占比 |
| **按用户** | 每个用户的调用次数和费用 |
| **按 Trace** | 每次请求的详细 token 消耗 |
| **按标签** | 按自定义标签分组统计 |

### 3. Prompt 管理

在 LangFuse 界面管理 Prompt 版本，支持 A/B 测试，不用改代码就能切换 Prompt。

```python
# 从 LangFuse 获取 Prompt（不用硬编码）
prompt_client = langfuse.get_prompt("rag_prompt", version=1)
prompt = prompt_client.compile(topic="RAG", context="...")

# 也可以获取最新版本
prompt_client = langfuse.get_prompt("rag_prompt")  # 不指定版本，获取最新
```

**优势**：
- 运营人员不用改代码就能优化 Prompt
- 支持版本管理，随时回滚
- 支持 A/B 测试，对比不同 Prompt 效果
- Prompt 变更有记录，可追溯

### 4. 评分与反馈

支持对回答打分（1-5星），收集用户反馈，用于评估和优化。

```python
# 数值评分（1-5）
trace.score(name="relevance", value=4, comment="回答相关但不够详细")

# 布尔评分（是/否）
trace.score(name="correct", value=True, comment="回答正确")

# 分类评分
trace.score(name="sentiment", value="positive", data_type="CATEGORICAL")
```

| 评分类型 | 说明 | 示例 |
|----------|------|------|
| **NUMERIC** | 数值评分 | 1-5 分、0-100 分 |
| **BOOLEAN** | 布尔评分 | 正确/错误、有用/无用 |
| **CATEGORICAL** | 分类评分 | 正面/中性/负面 |

### 5. 数据集与评估

上传测试数据集，批量运行评估，对比不同 Prompt/模型的效果。

```python
# 创建数据集
dataset = langfuse.create_dataset(name="rag_evaluation")

# 添加数据项
dataset.add_item(
    input={"question": "什么是RAG？"},
    expected_output={"answer": "RAG是检索增强生成..."},
)

# 运行评估
from langfuse.model import DatasetItem
for item in dataset.items:
    # 运行你的应用
    result = rag_query(item.input["question"])
    # 记录评估结果
    item.link(
        trace_or_observation=trace,
        run_name="prompt_v2_evaluation",
    )
```

---

## FastAPI 集成示例

```python
"""
langfuse_fastapi.py - FastAPI + LangFuse 集成示例
每个请求自动创建 Trace，记录完整调用链
"""
import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LangFuse 集成示例")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 初始化 LangFuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"


@app.post("/api/chat")
async def chat(req: QueryRequest, request: Request):
    """
    聊天接口
    每个请求自动创建 Trace，记录完整调用链
    """
    # 1. 创建 Trace
    trace = langfuse.trace(
        name="chat_query",
        metadata={
            "question": req.question,
            "user_id": req.user_id,
            "ip": request.client.host,
        },
        session_id=req.user_id,
        tags=["chat", "v1"],
    )

    try:
        # 2. LLM 调用（记录为 Generation）
        generation = trace.generation(
            name="llm_call",
            model="gpt-4o-mini",
            input={"prompt": req.question},
            metadata={"temperature": 0},
        )

        prompt = ChatPromptTemplate.from_template("你是一个 helpful 的助手，请回答：{question}")
        chain = prompt | llm
        result = chain.invoke({"question": req.question})

        # 更新 Generation 输出
        generation.end(
            output={"answer": result.content},
            usage={
                "input": result.usage_metadata["input_tokens"],
                "output": result.usage_metadata["output_tokens"],
                "total": result.usage_metadata["total_tokens"],
                "unit": "TOKENS",
            }
        )

        # 3. 更新 Trace
        trace.update(output={"answer": result.content})

        return {
            "answer": result.content,
            "trace_id": trace.id,
            "trace_url": f"{os.getenv('LANGFUSE_HOST')}/trace/{trace.id}",
        }

    except Exception as e:
        # 记录错误
        trace.update(
            level="ERROR",
            status_message=str(e),
        )
        raise


@app.post("/api/chat/{trace_id}/score")
async def score_chat(trace_id: str, score: int, comment: str = ""):
    """用户对回答评分"""
    langfuse.score(
        trace_id=trace_id,
        name="user_satisfaction",
        value=score,
        comment=comment,
    )
    return {"success": True, "message": "评分已记录"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 学习要点

1. **LangFuse** 是开源可自建的 LLM 观测平台，适合企业内网和数据敏感场景
2. **LangSmith vs LangFuse**：LangSmith 闭源 SaaS 功能完善，LangFuse 开源可自建数据安全
3. **两种集成方式**：LangChain 自动追踪（传 CallbackHandler）+ 手动追踪（Trace + Span，更灵活）
4. **Trace + Span 模型**：Trace 是一次完整请求，Span 是请求中的一个阶段，Generation 是 LLM 调用
5. **核心功能**：Trace 追踪、成本统计、Prompt 管理、评分反馈、数据集评估
6. **Prompt 管理**：运营人员不用改代码就能优化 Prompt，支持版本管理和 A/B 测试
7. **评分反馈**：支持数值、布尔、分类三种评分类型，收集用户反馈用于优化
8. **FastAPI 集成**：每个请求自动创建 Trace，记录完整调用链，支持用户评分
9. **生产环境**：一定要配观测，否则出问题无法排查，成本不可控
10. **数据安全**：LangFuse 私有化部署，数据在自己服务器，适合敏感数据场景

## 扩展方向

- 学习 LangFuse 的高级功能（Annotations、Sessions、Playground）
- 探索 LangFuse 与 LangGraph 的集成（图编排的全链路追踪）
- 学习 LangFuse 的数据集评估和 A/B 测试
- 探索 LangFuse 的告警和通知配置（异常调用、成本超预算）
- 学习 LangFuse 的团队协作和权限管理
- 探索 LangFuse 与 OpenTelemetry 的集成
- 学习 LangFuse 的数据导出和 API 调用
- 探索 LangFuse 的高可用部署（集群、负载均衡）
- 学习 LangFuse 的自定义指标和 Dashboard
- 探索其他可观测平台（LangSmith、Helicone、Arize Phoenix、Braintrust）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/04-storage-monitoring/41-langfuse-observability

包含本文的完整可运行代码示例（LangChain自动追踪+手动追踪+Prompt管理+评分反馈+FastAPI集成）。

---

**上一篇**：[RabbitMQ 消息队列](./40_RabbitMQ.md) | **下一篇**：[图解 Transformer 架构](./42_图解Transformer架构.md)
