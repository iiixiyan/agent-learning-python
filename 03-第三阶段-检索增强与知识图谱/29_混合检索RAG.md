# 混合检索 RAG：多路召回 + 重排模型

> **Python 版** | 基于 LangChain + ElasticSearch + Reranker 技术栈
> 前置知识：[RAG 基础](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)、[ElasticSearch 全文检索](../02-第二阶段-企业级后端与进阶框架/28_ElasticSearch全文检索.md)

---

## 为什么需要混合检索？

单一检索方式都有局限性：

| 检索方式 | 优点 | 缺点 | 擅长场景 |
|----------|------|------|----------|
| **向量检索** | 语义匹配好，能理解同义词 | 精确关键词差，可能漏关键信息 | 同义表达、语义理解 |
| **全文检索（ES/BM25）** | 关键词精确，速度快 | 语义理解差，同义词匹配不到 | 精确关键词、编号、名称 |
| **知识图谱检索** | 关系推理强 | 构建成本高，覆盖有限 | 实体关系、推理查询 |
| **混合检索** | 兼顾语义和精确 | 实现复杂，需要重排 | 企业级 RAG 标配 |

### 实际案例

| 用户问题 | 向量检索 | 全文检索 | 混合检索 |
|----------|----------|----------|----------|
| "怎么用大模型做问答？" | ✅ 能匹配"LLM应用" | ❌ 关键词不匹配 | ✅ 都能匹配 |
| "API_KEY 怎么配置？" | ❌ 语义模糊 | ✅ 精确匹配"API_KEY" | ✅ 都能匹配 |
| "LangChain 和 LlamaIndex 区别" | ✅ 语义理解 | ⚠️ 部分匹配 | ✅ 都能匹配 |

**结论**：企业级 RAG 必须用混合检索，召回率和准确率都更高。

---

## 混合检索架构

```
                        用户提问
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ 向量检索    │  │ 全文检索    │  │ 知识图谱    │
    │ (语义匹配)  │  │ (关键词匹配) │  │ (关系推理)  │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                    ┌──────────────┐
                    │  合并去重     │
                    │ (Merge & Dedup)│
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  重排模型     │
                    │  (Reranker)  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Top-K 结果   │
                    │  (3-5条)      │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  大模型生成   │
                    │  (LLM)        │
                    └──────────────┘
```

### 核心流程

| 步骤 | 说明 | 数量 |
|------|------|------|
| **多路召回** | 向量检索 + 全文检索 + 知识图谱，各自召回 Top-N | 每路 10-20 条 |
| **合并去重** | 合并所有召回结果，按内容去重 | 20-50 条 |
| **重排（Rerank）** | 用 CrossEncoder 重新计算相关性分数 | 20-50 条 → 排序 |
| **Top-K 截取** | 取重排后分数最高的 K 条 | 3-5 条 |
| **大模型生成** | 基于 Top-K 文档生成回答 | 最终答案 |

---

## 完整实现

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community faiss-cpu \
    elasticsearch sentence-transformers python-dotenv numpy
```

### 完整代码

创建 `hybrid_rag.py`：

```python
"""
hybrid_rag.py - 混合检索 RAG 完整实现
多路召回（向量 + 全文）+ 重排模型（Reranker）+ 大模型生成
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

load_dotenv()


# ============ 初始化 ============

# 大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# Embedding 模型（用于向量检索）
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ElasticSearch 客户端（用于全文检索）
es = Elasticsearch("http://localhost:9200")

# 重排模型（Reranker）
# 中文场景推荐用 BAAI/bge-reranker-base
# 英文场景可以用 cross-encoder/ms-marco-MiniLM-L-6-v2
reranker = CrossEncoder('BAAI/bge-reranker-base')


# ============ 1. 准备数据 ============

def prepare_data(file_path: str = "knowledge_base.txt"):
    """
    准备知识库数据：加载文档 → 切分 → 构建向量库和 ES 索引

    Args:
        file_path: 知识库文件路径
    """
    # 加载文档
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # 切分文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # 每块 300 字符
        chunk_overlap=50,     # 重叠 50 字符
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 文档切分完成，共 {len(chunks)} 个块")

    # 构建向量库（FAISS）
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ 向量库构建完成")

    # 构建 ES 索引（全文检索）
    index_name = "knowledge_base"
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    }
                }
            }
        }
    )

    # 批量插入 ES
    from elasticsearch.helpers import bulk
    actions = [
        {"_index": index_name, "_source": {"text": chunk.page_content}}
        for chunk in chunks
    ]
    bulk(es, actions)
    es.indices.refresh(index=index_name)
    print("✅ ES 索引构建完成")

    return vectorstore, index_name


# ============ 2. 多路召回 ============

def vector_search(query: str, vectorstore, top_k: int = 10) -> List[Dict]:
    """
    向量检索（语义匹配）

    Args:
        query: 用户查询
        vectorstore: 向量库
        top_k: 返回数量

    Returns:
        List[Dict]: 检索结果列表，每个包含 text、score、source
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {
            "text": doc.page_content,
            "score": float(score),
            "source": "vector",
        }
        for doc, score in results
    ]


def keyword_search(query: str, index_name: str = "knowledge_base", top_k: int = 10) -> List[Dict]:
    """
    全文检索（关键词匹配，BM25）

    Args:
        query: 用户查询
        index_name: ES 索引名
        top_k: 返回数量

    Returns:
        List[Dict]: 检索结果列表
    """
    result = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "boost": 1.0,
                    }
                }
            },
            "size": top_k,
        }
    )
    return [
        {
            "text": hit["_source"]["text"],
            "score": hit["_score"],
            "source": "bm25",
        }
        for hit in result["hits"]["hits"]
    ]


def hybrid_recall(
    query: str,
    vectorstore,
    index_name: str = "knowledge_base",
    top_k: int = 20,
) -> List[Dict]:
    """
    混合召回：向量检索 + 全文检索，合并去重

    Args:
        query: 用户查询
        vectorstore: 向量库
        index_name: ES 索引名
        top_k: 每路召回数量

    Returns:
        List[Dict]: 合并去重后的结果
    """
    # 1. 向量检索
    vec_results = vector_search(query, vectorstore, top_k=top_k)
    print(f"  向量检索召回: {len(vec_results)} 条")

    # 2. 全文检索
    kw_results = keyword_search(query, index_name, top_k=top_k)
    print(f"  全文检索召回: {len(kw_results)} 条")

    # 3. 合并去重（按文本内容前50字作为去重键）
    seen = set()
    merged = []
    for item in vec_results + kw_results:
        key = item["text"][:50]  # 用前50字作为去重键
        if key not in seen:
            seen.add(key)
            merged.append(item)

    print(f"  合并去重后: {len(merged)} 条")
    return merged


# ============ 3. 重排（Reranker）============

def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    用 CrossEncoder 重排候选结果

    Args:
        query: 用户查询
        candidates: 候选结果列表
        top_k: 返回 Top-K

    Returns:
        List[Dict]: 重排后的 Top-K 结果
    """
    if not candidates:
        return []

    # 构造 (query, text) 对
    pairs = [[query, item["text"]] for item in candidates]

    # 用 CrossEncoder 预测相关性分数
    scores = reranker.predict(pairs)

    # 按重排分数排序
    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    print(f"  重排后 Top-{top_k}:")
    for i, item in enumerate(candidates[:top_k]):
        print(f"    [{i+1}] 分数={item['rerank_score']:.4f} 来源={item['source']} 文本={item['text'][:30]}...")

    return candidates[:top_k]


# ============ 4. RAG 问答 ============

def rag_qa(query: str, vectorstore, index_name: str = "knowledge_base") -> str:
    """
    混合检索 RAG 问答

    Args:
        query: 用户问题
        vectorstore: 向量库
        index_name: ES 索引名

    Returns:
        str: 大模型生成的回答
    """
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"{'='*60}")

    # 1. 混合召回（多路召回 + 合并去重）
    print("\n[Step 1] 混合召回...")
    candidates = hybrid_recall(query, vectorstore, index_name, top_k=20)

    # 2. 重排（Reranker）
    print("\n[Step 2] 重排...")
    top_chunks = rerank(query, candidates, top_k=5)

    # 3. 拼接上下文
    context = "\n\n".join([
        f"[{i+1}] {c['text']}"
        for i, c in enumerate(top_chunks)
    ])

    # 4. 大模型生成
    print("\n[Step 3] 大模型生成...")
    prompt = f"""根据以下资料回答问题，引用资料编号。
如果资料中没有相关信息，请如实说明。

资料：
{context}

问题：{query}

回答："""

    response = llm.invoke(prompt)
    answer = response.content

    print(f"\n回答:\n{answer}")
    print(f"\n{'='*60}\n")

    return answer


# ============ 5. 使用示例 ============

if __name__ == "__main__":
    # 准备数据（首次运行需要，后续可以注释掉）
    vectorstore, index_name = prepare_data("knowledge_base.txt")

    # 问答示例
    questions = [
        "什么是 RAG？",
        "向量检索和全文检索有什么区别？",
        "为什么需要重排模型？",
    ]

    for q in questions:
        rag_qa(q, vectorstore, index_name)
```

### 运行示例

```bash
# 1. 准备知识库文件
echo "RAG（检索增强生成）是一种结合检索和生成的技术..." > knowledge_base.txt

# 2. 运行混合检索 RAG
python hybrid_rag.py
```

---

## 重排模型推荐

| 模型 | 说明 | 速度 | 效果 | 适用场景 |
|------|------|------|------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 轻量，英文好 | 快 | 中 | 英文场景、对速度要求高 |
| `BAAI/bge-reranker-base` | 中文效果好 | 中 | 好 | 中文场景、通用推荐 |
| `BAAI/bge-reranker-large` | 中文最佳 | 慢 | 最好 | 中文场景、对效果要求高 |
| `BAAI/bge-reranker-v2-m3` | 多语言支持 | 中 | 好 | 多语言场景 |
| Cohere Rerank API | 商用 API | 快 | 好 | 不想本地部署、商用场景 |
| Jina Reranker API | 商用 API | 快 | 好 | 不想本地部署、商用场景 |

**中文场景推荐用 `BAAI/bge-reranker-base`**，平衡速度和效果。

### 重排模型 vs Embedding 模型

| 维度 | Embedding 模型（双塔） | Reranker 模型（CrossEncoder） |
|------|----------------------|-------------------------------|
| **结构** | 分别编码 query 和 doc，计算余弦相似度 | 同时输入 query 和 doc，直接输出相关性分数 |
| **速度** | 快（可预计算 doc 向量） | 慢（每次都要前向传播） |
| **效果** | 中（粗排） | 好（精排） |
| **适用阶段** | 召回阶段（Top 100） | 重排阶段（Top 10 → Top 5） |

**为什么需要重排？**
- 向量检索是粗排，速度快但精度有限
- 重排模型是精排，速度慢但精度高
- 组合使用：向量检索召回 Top 100 → 重排模型精排 → Top 5

---

## 混合检索进阶

### 1. 结果融合算法（RRF）

除了简单的合并去重，还可以用 **RRF（Reciprocal Rank Fusion）** 算法融合多路结果：

```python
def rrf_fuse(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    RRF（Reciprocal Rank Fusion）结果融合算法

    Args:
        results_list: 多路检索结果列表
        k: RRF 参数，默认 60

    Returns:
        List[Dict]: 融合后的结果
    """
    scores = {}
    for results in results_list:
        for rank, item in enumerate(results):
            key = item["text"][:50]
            if key not in scores:
                scores[key] = {"item": item, "score": 0}
            # RRF 公式：1 / (k + rank)
            scores[key]["score"] += 1 / (k + rank + 1)

    # 按融合分数排序
    fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["item"] for item in fused]
```

### 2. 加权融合

```python
def weighted_fuse(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[Dict]:
    """
    加权融合：向量检索和全文检索按权重融合

    Args:
        vector_results: 向量检索结果
        keyword_results: 全文检索结果
        vector_weight: 向量检索权重
        keyword_weight: 全文检索权重

    Returns:
        List[Dict]: 融合后的结果
    """
    # 归一化分数
    def normalize(results):
        if not results:
            return results
        max_score = max(r["score"] for r in results)
        for r in results:
            r["norm_score"] = r["score"] / max_score if max_score > 0 else 0
        return results

    vector_results = normalize(vector_results)
    keyword_results = normalize(keyword_results)

    # 合并并加权
    merged = {}
    for r in vector_results:
        key = r["text"][:50]
        merged[key] = {"item": r, "score": r["norm_score"] * vector_weight}

    for r in keyword_results:
        key = r["text"][:50]
        if key in merged:
            merged[key]["score"] += r["norm_score"] * keyword_weight
        else:
            merged[key] = {"item": r, "score": r["norm_score"] * keyword_weight}

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
```

### 3. 查询改写（Query Rewriting）

在检索前，先用大模型改写用户查询，提高召回率：

```python
def rewrite_query(query: str) -> str:
    """
    查询改写：用大模型优化查询，提高召回率

    Args:
        query: 原始查询

    Returns:
        str: 改写后的查询
    """
    prompt = f"""请将以下用户查询改写为更适合检索的关键词组合，保留核心意图，扩展同义词。
原始查询：{query}
改写后的查询："""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    print(f"  查询改写: {query} → {rewritten}")
    return rewritten
```

---

## 学习要点

1. **混合检索 = 向量检索 + 全文检索**，兼顾语义理解和精确关键词匹配
2. **多路召回后要去重**，避免重复片段影响重排和生成
3. **重排模型（Reranker）** 大幅提升准确率，是企业级 RAG 标配
4. **召回阶段多召一些（20-50条）**，重排后取 Top-K（3-5条）
5. **RRF（Reciprocal Rank Fusion）** 是常用的结果融合算法，不需要归一化分数
6. **CrossEncoder vs Bi-Encoder**：CrossEncoder 精度高但速度慢，适合精排；Bi-Encoder 速度快但精度低，适合粗排
7. **中文场景推荐用 BAAI/bge-reranker-base**，平衡速度和效果
8. **查询改写（Query Rewriting）** 可以提高召回率，用大模型扩展同义词
9. **混合检索是企业级 RAG 的标配**，单一检索方式都有局限性
10. **知识图谱检索**可以作为第三路召回，补充关系推理能力

## 扩展方向

- 学习更多融合算法（加权融合、学习排序 LTR、深度学习融合）
- 探索多路召回（向量 + 全文 + 知识图谱 + 结构化数据）
- 学习查询改写技术（Query Rewriting、Query Expansion、HyDE）
- 探索动态路由（根据查询类型自动选择检索方式）
- 学习重排模型微调（Fine-tuning Reranker）
- 探索多模态混合检索（文本 + 图片 + 视频）
- 学习检索结果的可解释性（为什么这条结果排前面）
- 探索实时索引更新（增量索引、近实时检索）
- 学习检索评估指标（Recall@K、MRR、NDCG）
- 探索 A/B 测试和检索效果优化

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/29-hybrid-rag

包含本文的完整可运行代码示例（混合检索 RAG + 多路召回 + RRF 融合 + 重排模型 + 查询改写）。

---

**上一篇**：[ElasticSearch 全文检索](../02-第二阶段-企业级后端与进阶框架/28_ElasticSearch全文检索.md) | **下一篇**：[Neo4j 知识图谱和 Graph RAG](./30_Neo4j知识图谱和Graph-RAG.md)
