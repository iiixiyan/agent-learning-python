# PostgreSQL：AI 时代最适合的数据库

> **Python 版** | 基于 PostgreSQL + pgvector + SQLAlchemy 技术栈
> 前置知识：[RAG 基础](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)、[ElasticSearch 全文检索](../02-第二阶段-企业级后端与进阶框架/28_ElasticSearch全文检索.md)

---

## 为什么 PostgreSQL 适合 AI 项目？

开发 AI 应用时，通常需要多种存储：关系型数据、向量数据、全文检索、JSON 文档。如果每种都用专门的数据库，维护成本很高。

**PostgreSQL + pgvector** 一个数据库搞定所有需求：

| 能力 | PostgreSQL | MySQL | MongoDB | 专用向量库 |
|------|-----------|-------|---------|-----------|
| **关系型数据** | ✅ 强 | ✅ 强 | ❌ | ❌ |
| **JSON 存储** | ✅ JSONB（强） | ✅ JSON（弱） | ✅ 原生 | ❌ |
| **全文检索** | ✅ tsvector | ✅ 全文索引 | ❌ | ❌ |
| **向量检索** | ✅ pgvector | ❌ | ❌ | ✅ 专业 |
| **事务支持** | ✅ ACID 强 | ✅ ACID 强 | ❌ 弱 | ❌ |
| **扩展生态** | ✅ 丰富（2000+扩展） | ⚠️ 有限 | ⚠️ 有限 | ⚠️ 有限 |

**结论**：PostgreSQL + pgvector 一个数据库搞定 AI 项目的所有存储需求，不用维护多个数据库，降低运维成本。

### PostgreSQL 在 AI 项目中的典型用途

```
AI 应用
   │
   ├── 用户数据（关系表）→ users, sessions, permissions
   ├── 文档向量（pgvector）→ documents, embeddings → RAG 检索
   ├── 对话记录（JSONB）→ conversations, messages
   ├── 全文检索（tsvector）→ articles, knowledge_base → 关键词搜索
   ├── Agent 配置（JSONB）→ agent_configs, tool_configs
   └── 日志监控（关系表+JSONB）→ request_logs, error_logs
```

---

## 环境准备

### Docker 启动带 pgvector 的 PostgreSQL

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: pgvector-dev
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: agent_db
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    restart: always

volumes:
  pg_data:
```

```bash
# 启动 PostgreSQL
docker compose up -d

# 验证连接
docker exec -it pgvector-dev psql -U postgres -d agent_db -c "SELECT version();"
```

### Python 依赖

```bash
pip install psycopg2-binary sqlalchemy pgvector openai langchain langchain-community python-dotenv
```

---

## 核心功能演示

### 1. 向量检索（pgvector）

pgvector 是 PostgreSQL 的向量扩展，支持三种距离计算：

| 距离类型 | 操作符 | 说明 | 适用场景 |
|----------|--------|------|----------|
| **余弦相似度** | `<=>` | 1 - 余弦相似度，值越小越相似 | 文本相似度（最常用） |
| **L2 距离** | `<->` | 欧几里得距离，值越小越相似 | 图像、数值向量 |
| **内积** | `<#>` | 负内积，值越小越相似 | 归一化向量 |

#### 完整示例

创建 `pgvector_demo.py`：

```python
"""
pgvector_demo.py - PostgreSQL 向量检索完整示例
包含：连接、建表、插入向量、余弦相似度检索、L2距离检索
"""
import os
import json
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI

load_dotenv()

# 初始化 OpenAI 客户端（用于生成向量）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 连接 PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    dbname="agent_db",
    user="postgres",
    password="password",
    port=5432,
)
cur = conn.cursor()
print("✅ PostgreSQL 连接成功")


# ========== 1. 启用 pgvector 扩展 ==========
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
print("✅ pgvector 扩展已启用")


# ========== 2. 创建文档表 ==========
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        source VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        embedding vector(1536)  -- text-embedding-3-small 是 1536 维
    )
""")
conn.commit()
print("✅ 文档表已创建")


# ========== 3. 生成向量并插入数据 ==========
def get_embedding(text: str) -> list:
    """
    生成文本的向量表示

    Args:
        text: 输入文本

    Returns:
        list: 向量列表
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# 示例文档
docs = [
    {"content": "Python是一种简单易学的编程语言，语法简洁，拥有丰富的第三方库。", "source": "python_intro"},
    {"content": "LangChain是大模型应用开发框架，提供Chain、Agent、Tool等核心组件。", "source": "langchain_intro"},
    {"content": "PostgreSQL是功能强大的关系型数据库，支持JSONB、全文检索、向量检索等扩展。", "source": "pg_intro"},
    {"content": "FastAPI是现代Python Web框架，支持异步、自动文档、类型提示，性能优秀。", "source": "fastapi_intro"},
    {"content": "RAG（检索增强生成）结合检索和生成，提升大模型问答的准确性和时效性。", "source": "rag_intro"},
]

# 清空旧数据（演示用）
cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")

# 插入向量数据
for doc in docs:
    emb = get_embedding(doc["content"])
    cur.execute(
        "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
        (doc["content"], doc["source"], str(emb))
    )
conn.commit()
print(f"✅ 已插入 {len(docs)} 条文档向量")


# ========== 4. 向量检索（余弦相似度） ==========
def vector_search(query: str, top_k: int = 3, distance_type: str = "cosine"):
    """
    向量检索

    Args:
        query: 查询文本
        top_k: 返回数量
        distance_type: 距离类型（cosine/l2/inner_product）

    Returns:
        list: 检索结果列表
    """
    query_emb = get_embedding(query)

    # 根据距离类型选择操作符
    if distance_type == "cosine":
        # 余弦相似度：1 - (embedding <=> query)
        score_expr = "1 - (embedding <=> %s)"
        order_expr = "embedding <=> %s"
    elif distance_type == "l2":
        # L2 距离：embedding <-> query
        score_expr = "-(embedding <-> %s)"  # 取负数，越大越相似
        order_expr = "embedding <-> %s"
    else:
        # 内积：-(embedding <#> query)
        score_expr = "-(embedding <#> %s)"
        order_expr = "embedding <#> %s"

    cur.execute(f"""
        SELECT content, source, {score_expr} as score
        FROM documents
        ORDER BY {order_expr}
        LIMIT %s
    """, (str(query_emb), str(query_emb), top_k))

    results = cur.fetchall()
    return [{"content": r[0], "source": r[1], "score": float(r[2])} for r in results]


# 测试检索
print("\n" + "="*60)
print("测试1: 余弦相似度检索 - '什么是大模型框架？'")
print("="*60)
results = vector_search("什么是大模型框架？", top_k=3, distance_type="cosine")
for i, r in enumerate(results, 1):
    print(f"  [{i}] 相似度: {r['score']:.4f} | 来源: {r['source']}")
    print(f"      内容: {r['content'][:60]}...")

print("\n" + "="*60)
print("测试2: L2距离检索 - 'Python Web开发'")
print("="*60)
results = vector_search("Python Web开发", top_k=3, distance_type="l2")
for i, r in enumerate(results, 1):
    print(f"  [{i}] 分数: {r['score']:.4f} | 来源: {r['source']}")
    print(f"      内容: {r['content'][:60]}...")


# ========== 5. 创建向量索引（提升检索速度） ==========
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops)
""")
conn.commit()
print("\n✅ HNSW 向量索引已创建（提升大规模检索速度）")

# 关闭连接
cur.close()
conn.close()
print("\n✅ 演示完成，连接已关闭")
```

### 运行示例

```bash
# 1. 创建 .env 文件
echo "OPENAI_API_KEY=你的_api_key" > .env
echo "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1" >> .env

# 2. 运行向量检索演示
python pgvector_demo.py
```

---

### 2. 全文检索

PostgreSQL 内置全文检索，小项目不需要额外部署 ElasticSearch。

#### 建表和索引

```sql
-- 创建带全文检索的文章表
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    author VARCHAR(100),
    published_at DATE,
    -- tsvector 列：自动从 title 和 content 生成
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED
);

-- 创建 GIN 索引（全文检索专用索引，速度快）
CREATE INDEX idx_articles_tsv ON articles USING GIN(tsv);

-- 插入示例数据
INSERT INTO articles (title, content, author, published_at) VALUES
('Python入门教程', 'Python是一种简单易学的编程语言，语法简洁，适合新手入门。', '张三', '2024-01-15'),
('LangChain实战指南', 'LangChain是大模型应用开发框架，提供Chain、Agent、Tool等核心组件。', '李四', '2024-02-20'),
('RAG检索增强生成', 'RAG结合检索和生成，提升问答准确率。RAG通过向量检索获取相关文档。', '王五', '2024-03-10');
```

#### 全文检索查询

```sql
-- 基本全文检索
SELECT title, ts_rank(tsv, plainto_tsquery('simple', 'python 编程')) as rank
FROM articles
WHERE tsv @@ plainto_tsquery('simple', 'python 编程')
ORDER BY rank DESC
LIMIT 10;

-- 短语检索（精确匹配短语）
SELECT title
FROM articles
WHERE tsv @@ phraseto_tsquery('simple', '大模型 应用')
ORDER BY published_at DESC;

-- 高亮显示匹配词
SELECT
    title,
    ts_headline('simple', content, plainto_tsquery('simple', 'RAG 检索')) as highlighted
FROM articles
WHERE tsv @@ plainto_tsquery('simple', 'RAG 检索');

-- 统计词频
SELECT word, ndoc, nentry
FROM ts_stat('SELECT tsv FROM articles')
WHERE word IN ('python', 'langchain', 'rag')
ORDER BY ndoc DESC;
```

#### Python 调用全文检索

```python
"""
fulltext_search.py - PostgreSQL 全文检索 Python 示例
"""
import psycopg2

conn = psycopg2.connect(host="localhost", dbname="agent_db", user="postgres", password="password")
cur = conn.cursor()


def fulltext_search(query: str, top_k: int = 5) -> list:
    """
    全文检索

    Args:
        query: 搜索关键词
        top_k: 返回数量

    Returns:
        list: 检索结果
    """
    cur.execute("""
        SELECT id, title, author, published_at,
               ts_rank(tsv, plainto_tsquery('simple', %s)) as rank,
               ts_headline('simple', content, plainto_tsquery('simple', %s)) as highlight
        FROM articles
        WHERE tsv @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
    """, (query, query, query, top_k))

    results = cur.fetchall()
    return [
        {
            "id": r[0], "title": r[1], "author": r[2],
            "published_at": str(r[3]), "rank": float(r[4]),
            "highlight": r[5]
        }
        for r in results
    ]


# 测试
results = fulltext_search("Python 编程")
for r in results:
    print(f"[{r['rank']:.4f}] {r['title']} - {r['author']}")
    print(f"  高亮: {r['highlight'][:100]}...")

cur.close()
conn.close()
```

---

### 3. JSONB 存储

JSONB 适合存储灵活的 Agent 配置、对话记录、元数据等。

#### 建表和操作

```python
"""
jsonb_demo.py - PostgreSQL JSONB 存储 Python 示例
"""
import json
import psycopg2

conn = psycopg2.connect(host="localhost", dbname="agent_db", user="postgres", password="password")
cur = conn.cursor()

# 创建对话表（JSONB 存储消息和元数据）
cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL,
        title VARCHAR(255),
        messages JSONB NOT NULL DEFAULT '[]'::jsonb,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("✅ 对话表已创建")


# ========== 1. 插入 JSON 数据 ==========
messages = [
    {"role": "user", "content": "你好，帮我介绍一下Python"},
    {"role": "assistant", "content": "Python是一种简单易学的编程语言..."},
    {"role": "user", "content": "有什么学习资源推荐吗？"},
]
metadata = {
    "model": "qwen-plus",
    "tokens": 1500,
    "tags": ["python", "入门"],
    "rating": 5,
}

cur.execute("""
    INSERT INTO conversations (user_id, title, messages, metadata)
    VALUES (%s, %s, %s, %s)
    RETURNING id
""", (1, "Python学习咨询", json.dumps(messages, ensure_ascii=False),
      json.dumps(metadata, ensure_ascii=False)))
conv_id = cur.fetchone()[0]
conn.commit()
print(f"✅ 对话已创建，ID: {conv_id}")


# ========== 2. 查询 JSON 字段 ==========
# 查询第一条消息的内容
cur.execute("""
    SELECT messages->0->>'content' as first_msg
    FROM conversations WHERE id = %s
""", (conv_id,))
print(f"\n第一条消息: {cur.fetchone()[0]}")

# 查询所有用户消息
cur.execute("""
    SELECT jsonb_path_query_array(messages, '$[*] ? (@.role == "user").content') as user_msgs
    FROM conversations WHERE id = %s
""", (conv_id,))
print(f"用户消息: {cur.fetchone()[0]}")

# 查询元数据中的标签
cur.execute("""
    SELECT metadata->'tags' as tags, metadata->>'model' as model
    FROM conversations WHERE id = %s
""", (conv_id,))
row = cur.fetchone()
print(f"标签: {row[0]}, 模型: {row[1]}")


# ========== 3. 更新 JSON 字段 ==========
# 追加一条新消息
new_msg = {"role": "assistant", "content": "推荐你看Python官方教程和菜鸟教程。"}
cur.execute("""
    UPDATE conversations
    SET messages = messages || %s::jsonb,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = %s
""", (json.dumps(new_msg, ensure_ascii=False), conv_id))
conn.commit()
print(f"\n✅ 已追加新消息")

# 更新元数据（添加字段）
cur.execute("""
    UPDATE conversations
    SET metadata = jsonb_set(metadata, '{status}', '"completed"'::jsonb)
    WHERE id = %s
""", (conv_id,))
conn.commit()
print("✅ 已更新元数据状态")


# ========== 4. JSONB 条件查询 ==========
# 查询包含特定标签的对话
cur.execute("""
    SELECT id, title FROM conversations
    WHERE metadata @> '{"tags": ["python"]}'::jsonb
""")
print(f"\n包含python标签的对话: {cur.fetchall()}")

# 查询消息数大于2的对话
cur.execute("""
    SELECT id, title, jsonb_array_length(messages) as msg_count
    FROM conversations
    WHERE jsonb_array_length(messages) > 2
""")
print(f"消息数>2的对话: {cur.fetchall()}")

cur.close()
conn.close()
print("\n✅ JSONB 演示完成")
```

---

## LangChain 集成

### PGVector 向量库

```python
"""
langchain_pgvector.py - LangChain 集成 PGVector
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 连接字符串
CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/agent_db"

# 初始化 Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 创建 PGVector 向量库
vectorstore = PGVector(
    collection_name="my_documents",
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# 添加文档
documents = [
    "Python是一种简单易学的编程语言",
    "LangChain是大模型应用开发框架",
    "PostgreSQL支持向量检索和全文检索",
    "FastAPI是高性能Python Web框架",
]
vectorstore.add_texts(documents)
print("✅ 文档已添加到向量库")

# 相似度检索
results = vectorstore.similarity_search("什么是大模型框架？", k=2)
print("\n检索结果:")
for doc in results:
    print(f"  - {doc.page_content}")

# 带分数的检索
results_with_score = vectorstore.similarity_search_with_score("Python编程", k=2)
print("\n带分数的检索结果:")
for doc, score in results_with_score:
    print(f"  分数: {score:.4f} | {doc.page_content}")
```

### SQLDatabase Chain（自然语言查 SQL）

```python
"""
nl2sql_demo.py - 自然语言转 SQL 查询
"""
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# 连接数据库
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:password@localhost:5432/agent_db")

# 初始化大模型
llm = ChatOpenAI(model="qwen-plus", temperature=0)

# 创建 SQL Agent
agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# 自然语言查询
result = agent.invoke("统计每个用户的对话数量，按数量降序排列")
print(result["output"])
```

---

## 生产环境最佳实践

### 1. 向量索引选择

| 索引类型 | 特点 | 适用场景 |
|----------|------|----------|
| **HNSW** | 近似最近邻，查询快，内存占用高 | 大规模数据（100万+），查询频繁 |
| **IVFFlat** | 倒排文件，需要先建立聚类，查询需要调参 | 中等规模，可接受调参 |
| **无索引（顺序扫描）** | 精确结果，查询慢 | 小规模数据（<1万），需要精确结果 |

```sql
-- HNSW 索引（推荐，查询快）
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- IVFFlat 索引（省内存，需要调参）
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 2. 连接池配置

```python
"""
connection_pool.py - PostgreSQL 连接池配置
生产环境必须使用连接池，避免频繁创建连接
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建引擎（带连接池配置）
engine = create_engine(
    "postgresql+psycopg2://postgres:password@localhost:5432/agent_db",
    pool_size=10,           # 连接池大小
    max_overflow=20,        # 最大溢出连接数
    pool_timeout=30,        # 获取连接超时时间（秒）
    pool_recycle=3600,      # 连接回收时间（秒）
    pool_pre_ping=True,     # 连接前检测连接是否有效
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 使用示例
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 3. 托管服务推荐

| 服务 | 说明 | 特点 |
|------|------|------|
| **Supabase** | 开源 Firebase 替代，基于 PG | 免费额度大，内置 pgvector，实时订阅 |
| **Neon** |  Serverless PostgreSQL | 按需付费，分支功能，自动扩展 |
| **AWS RDS for PG** | 云托管 PostgreSQL | 企业级，高可用，安全合规 |
| **阿里云 RDS for PG** | 云托管 PostgreSQL | 国内访问快，支持 pgvector |
| **Timescale** | 时序数据库，基于 PG | 时序数据优化，支持向量 |

---

## 学习要点

1. **PostgreSQL + pgvector** 一个数据库搞定 AI 项目所有存储需求（关系+向量+全文+JSON），降低运维成本
2. **pgvector** 支持三种距离计算：余弦相似度（`<=>`，文本最常用）、L2距离（`<->`）、内积（`<#>`）
3. **HNSW 索引**是大规模向量检索的首选，查询速度快，但内存占用较高
4. **PG 内置全文检索**（tsvector + GIN 索引），小项目不需要额外部署 ElasticSearch
5. **JSONB** 是二进制 JSON，支持索引和高效查询，适合存储灵活的 Agent 配置和对话记录
6. **LangChain 集成**：PGVector 作为向量库、SQLDatabase 实现自然语言查 SQL
7. **生产环境**必须使用连接池（SQLAlchemy），配置合理的 pool_size 和 pool_recycle
8. **托管服务**推荐 Supabase（开源、免费额度大、内置 pgvector）或云厂商 RDS
9. **向量维度**要和 Embedding 模型匹配（text-embedding-3-small 是 1536 维）
10. **混合检索**：PG 可以同时做向量检索和全文检索，实现向量+关键词的混合检索方案

## 扩展方向

- 学习 pgvector 的高级用法（半监督学习、向量压缩、乘积量化）
- 探索 PostgreSQL 的其他 AI 扩展（pgai、postgresml、pgsimilarity）
- 学习 PostgreSQL 性能优化（查询计划分析、索引优化、分区表）
- 探索 PostgreSQL 流复制和高可用架构
- 学习 PostgreSQL 安全配置（用户权限、行级安全 RLS、加密）
- 探索 PostgreSQL 实时订阅（LISTEN/NOTIFY、逻辑复制）
- 学习 PostgreSQL 全文检索的高级用法（自定义分词、同义词、停用词）
- 探索 PostgreSQL + pgvector 的混合检索方案（向量+全文+结构化过滤）
- 学习 PostgreSQL 备份恢复策略（pg_dump、WAL 归档、时间点恢复）
- 探索 PostgreSQL 向量检索的分布式方案（Citus + pgvector）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/34-postgresql-ai-database

包含本文的完整可运行代码示例（pgvector 向量检索 + 全文检索 + JSONB 存储 + LangChain 集成 + 连接池配置）。

---

**上一篇**：[DeepAgents 实战](./33_DeepAgents实战.md) | **下一篇**：[Redis 实现 Agent 短期记忆](./35_Redis-实现Agent短期记忆.md)
