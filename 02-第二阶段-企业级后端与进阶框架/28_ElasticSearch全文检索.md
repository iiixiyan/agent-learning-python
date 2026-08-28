# ElasticSearch 全文检索：倒排索引 + IK 分词器 + BM25 算法

> **Python 版** | 基于 ElasticSearch + Python elasticsearch-py 技术栈
> 前置知识：[Docker Compose 部署](./27_基于Docker-Compose的部署.md)、[RAG 基础](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)

---

## 为什么需要 ElasticSearch？

向量检索（RAG）擅长语义匹配，但不擅长精确关键词搜索。ElasticSearch 是全文检索引擎，擅长：

| 能力 | 说明 | 示例 |
|------|------|------|
| **关键词精确匹配** | 精确匹配用户输入的关键词 | 搜索"Python"必须包含"Python" |
| **中文分词搜索** | 中文自动分词，支持部分匹配 | 搜索"大模型"能匹配"大语言模型" |
| **复杂条件过滤** | 支持范围、日期、多字段组合过滤 | 时间范围 + 作者 + 关键词 |
| **高亮显示匹配词** | 搜索结果中高亮匹配的关键词 | `<em>Python</em>入门教程` |

**混合检索**（向量 + ES）是企业级 RAG 的标配，第 29 篇会详细讲。

### 向量检索 vs 全文检索

| 维度 | 向量检索（RAG） | 全文检索（ES） |
|------|-----------------|----------------|
| **匹配方式** | 语义相似度 | 关键词匹配 |
| **擅长场景** | 同义表达、语义理解 | 精确关键词、编号、名称 |
| **中文处理** | 天然支持 | 需要分词器 |
| **可解释性** | 低（黑盒） | 高（词频、IDF） |
| **更新成本** | 需要重新嵌入 | 直接索引 |
| **组合使用** | 语义召回 | 精确过滤 + 重排序 |

---

## 核心概念

### 倒排索引

倒排索引是 ES 的核心数据结构，类似书的索引页：

```
正排索引（文档 → 词）：
  文档1: [我, 爱, 北京, 天安门]
  文档2: [我, 爱, 上海, 外滩]

倒排索引（词 → 文档列表）：
  我:    [文档1, 文档2]
  爱:    [文档1, 文档2]
  北京:  [文档1]
  天安门: [文档1]
  上海:  [文档2]
  外滩:  [文档2]
```

搜索"北京"时，直接查倒排索引就能找到包含"北京"的所有文档，速度极快。

### IK 分词器

中文没有空格分隔，需要分词器把句子拆成词语。

| 分词模式 | 说明 | 示例 |
|----------|------|------|
| `ik_max_word` | 最细粒度切分，索引时用 | "我爱北京天安门" → "我/爱/北京/天安/天安门/安门" |
| `ik_smart` | 智能切分，搜索时用 | "我爱北京天安门" → "我/爱/北京/天安门" |

**为什么索引和搜索用不同模式？**
- 索引用 `ik_max_word`：尽可能多切分，提高召回率
- 搜索用 `ik_smart`：更智能的切分，提高准确率

### BM25 算法

BM25 是 ES 默认的相关性评分算法，考虑三个因素：

| 因素 | 说明 | 影响 |
|------|------|------|
| **词频（TF）** | 词在文档中出现的次数 | 出现越多，分数越高（但有上限） |
| **逆文档频率（IDF）** | 词在多少文档中出现 | 出现越少，越重要，分数越高 |
| **文档长度** | 文档的总长度 | 文档越长，分数越低（稀释效应） |

公式：

```
BM25 = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
```

关键参数：
- `k1`：词频饱和度（默认 1.2），控制词频增加时分数增长速度
- `b`：长度归一化（默认 0.75），文档越长分数越低

### ES 基本概念

| 概念 | 类比数据库 | 说明 |
|------|-----------|------|
| **Index（索引）** | 表（Table） | 一类文档的集合 |
| **Document（文档）** | 行（Row） | 一条具体的数据 |
| **Field（字段）** | 列（Column） | 文档中的属性 |
| **Mapping（映射）** | 表结构 | 字段类型和分词器定义 |
| **Query DSL** | SQL | ES 的查询语言（JSON 格式） |

---

## Docker 启动 ES + IK

### docker-compose.yml

```yaml
version: '3.8'

services:
  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: es-dev
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    restart: always

volumes:
  es_data:
```

### 启动和安装 IK 分词器

```bash
# 启动 ES
docker compose up -d

# 等待 ES 启动完成
curl http://localhost:9200

# 安装 IK 分词器（进入容器执行）
docker exec -it es-dev bin/elasticsearch-plugin install \
  https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.11.0/elasticsearch-analysis-ik-8.11.0.zip

# 重启 ES 使插件生效
docker restart es-dev

# 验证 IK 分词器安装成功
curl -X POST "http://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "analyzer": "ik_max_word",
    "text": "我爱北京天安门"
  }'
```

### 验证分词效果

```bash
# ik_max_word 模式（最细粒度）
curl -X POST "http://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -d '{"analyzer": "ik_max_word", "text": "大语言模型"}'

# ik_smart 模式（智能切分）
curl -X POST "http://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -d '{"analyzer": "ik_smart", "text": "大语言模型"}'
```

---

## Python 操作 ES

### 安装依赖

```bash
pip install elasticsearch==8.11.0
```

### 完整示例

创建 `es_demo.py`：

```python
"""
es_demo.py - ElasticSearch Python 操作完整示例
"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


# ========== 1. 连接 ES ==========
es = Elasticsearch("http://localhost:9200")

# 验证连接
if es.ping():
    print("✅ ES 连接成功")
else:
    print("❌ ES 连接失败")
    exit(1)


# ========== 2. 创建索引（指定 IK 分词器） ==========
index_name = "articles"

# 如果索引已存在，先删除（演示用）
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"🗑️  删除已有索引: {index_name}")

# 定义索引映射
index_body = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ik_max_word",      # 索引时用最细粒度
                "search_analyzer": "ik_smart",   # 搜索时用智能切分
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "author": {"type": "keyword"},       # 精确匹配，不分词
            "tags": {"type": "keyword"},         # 标签，精确匹配
            "date": {"type": "date"},             # 日期类型
            "views": {"type": "integer"},         # 整数类型
        }
    }
}

es.indices.create(index=index_name, body=index_body)
print(f"✅ 创建索引: {index_name}")


# ========== 3. 批量插入文档 ==========
docs = [
    {
        "title": "Python入门教程",
        "content": "Python是一种简单易学的编程语言，适合新手入门。Python语法简洁，拥有丰富的第三方库。",
        "author": "张三",
        "tags": ["Python", "编程", "入门"],
        "date": "2024-01-15",
        "views": 1500,
    },
    {
        "title": "LangChain实战指南",
        "content": "LangChain是大模型应用开发框架，提供了Chain、Agent、Tool等核心组件。LangChain支持多种大模型。",
        "author": "李四",
        "tags": ["LangChain", "AI", "大模型"],
        "date": "2024-02-20",
        "views": 2300,
    },
    {
        "title": "RAG检索增强生成技术",
        "content": "RAG结合检索和生成，提升问答准确率。RAG通过向量检索获取相关文档，再让大模型基于文档生成回答。",
        "author": "王五",
        "tags": ["RAG", "AI", "检索"],
        "date": "2024-03-10",
        "views": 3100,
    },
    {
        "title": "FastAPI高性能Web框架",
        "content": "FastAPI是现代Python Web框架，支持异步、自动文档、类型提示。FastAPI性能接近Node.js和Go。",
        "author": "张三",
        "tags": ["Python", "FastAPI", "Web"],
        "date": "2024-04-05",
        "views": 1800,
    },
]

# 批量插入
actions = [{"_index": index_name, "_source": doc} for doc in docs]
success, failed = bulk(es, actions)
print(f"✅ 批量插入: 成功 {success} 条, 失败 {len(failed)} 条")

# 刷新索引使文档可搜索
es.indices.refresh(index=index_name)


# ========== 4. 全文检索（多字段匹配 + 高亮） ==========
print("\n" + "="*60)
print("📝 全文检索：搜索'大模型编程'")
print("="*60)

result = es.search(
    index=index_name,
    body={
        "query": {
            "multi_match": {
                "query": "大模型编程",
                "fields": ["title^2", "content"],  # title 权重2倍
                "type": "best_fields",
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "content": {},
            },
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
        },
        "size": 5,
    }
)

for hit in result["hits"]["hits"]:
    print(f"\n得分: {hit['_score']:.2f}")
    print(f"标题: {hit['_source']['title']}")
    print(f"作者: {hit['_source']['author']}")
    if "highlight" in hit:
        print(f"高亮: {hit['highlight']}")


# ========== 5. 布尔查询（多条件组合） ==========
print("\n" + "="*60)
print("🔍 布尔查询：作者=张三 且 标签包含Python 且 浏览量>1000")
print("="*60)

result = es.search(
    index=index_name,
    body={
        "query": {
            "bool": {
                "must": [
                    {"term": {"author": "张三"}},           # 必须匹配
                    {"term": {"tags": "Python"}},            # 必须匹配
                ],
                "filter": [
                    {"range": {"views": {"gt": 1000}}},     # 过滤，不影响评分
                ],
            }
        },
        "sort": [{"views": {"order": "desc"}}],          # 按浏览量降序
    }
)

for hit in result["hits"]["hits"]:
    print(f"\n标题: {hit['_source']['title']}")
    print(f"作者: {hit['_source']['author']}")
    print(f"浏览量: {hit['_source']['views']}")


# ========== 6. 范围查询（日期范围） ==========
print("\n" + "="*60)
print("📅 范围查询：2024年2月到4月发布的文章")
print("="*60)

result = es.search(
    index=index_name,
    body={
        "query": {
            "range": {
                "date": {
                    "gte": "2024-02-01",
                    "lte": "2024-04-30",
                }
            }
        },
        "sort": [{"date": {"order": "asc"}}],
    }
)

for hit in result["hits"]["hits"]:
    print(f"\n标题: {hit['_source']['title']}")
    print(f"日期: {hit['_source']['date']}")


# ========== 7. 聚合查询（统计） ==========
print("\n" + "="*60)
print("📊 聚合查询：按作者统计文章数量和总浏览量")
print("="*60)

result = es.search(
    index=index_name,
    body={
        "size": 0,  # 不需要文档，只要聚合结果
        "aggs": {
            "by_author": {
                "terms": {"field": "author"},
                "aggs": {
                    "total_views": {"sum": {"field": "views"}},
                    "avg_views": {"avg": {"field": "views"}},
                },
            }
        },
    }
)

for bucket in result["aggregations"]["by_author"]["buckets"]:
    print(f"\n作者: {bucket['key']}")
    print(f"文章数: {bucket['doc_count']}")
    print(f"总浏览量: {bucket['total_views']['value']}")
    print(f"平均浏览量: {bucket['avg_views']['value']:.0f}")


print("\n✅ 所有查询执行完成！")
```

### 运行示例

```bash
python es_demo.py
```

---

## 常用查询类型速查

| 查询类型 | 说明 | 示例场景 |
|----------|------|----------|
| `match` | 全文匹配，会分词 | 搜索标题包含"Python"的文章 |
| `multi_match` | 多字段匹配 | 标题和内容都搜索 |
| `term` | 精确匹配，不分词 | 作者="张三"、标签="Python" |
| `terms` | 多值精确匹配 | 作者在 ["张三", "李四"] 中 |
| `range` | 范围查询 | 日期、数字范围 |
| `bool` | 布尔组合查询 | must/should/must_not/filter 组合 |
| `exists` | 字段存在性 | 有封面图的文章 |
| `prefix` | 前缀匹配 | 标题以"Python"开头 |
| `wildcard` | 通配符匹配 | 标题包含"*模型*" |

### Bool 查询组合

```python
{
    "bool": {
        "must": [],      # 必须匹配，影响评分（AND）
        "should": [],    # 可选匹配，影响评分（OR）
        "must_not": [],  # 必须不匹配（NOT）
        "filter": [],    # 必须匹配，不影响评分（用于过滤）
    }
}
```

---

## ES 在 Agent 项目中的应用

### 场景一：知识库关键词检索

```python
def search_knowledge_base(query: str, top_k: int = 5):
    """从知识库检索相关文档（关键词检索）"""
    result = es.search(
        index="knowledge_base",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content^2", "tags"],
                }
            },
            "size": top_k,
        }
    )
    return [hit["_source"] for hit in result["hits"]["hits"]]
```

### 场景二：日志分析

```python
def search_logs(service: str, level: str, start_time: str, end_time: str):
    """检索服务日志"""
    result = es.search(
        index="logs-*",
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"service": service}},
                        {"term": {"level": level}},
                    ],
                    "filter": [
                        {"range": {"timestamp": {"gte": start_time, "lte": end_time}}},
                    ],
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": 100,
        }
    )
    return result["hits"]["hits"]
```

### 场景三：混合检索（ES + 向量）

```python
def hybrid_search(query: str, top_k: int = 5):
    """
    混合检索：ES 关键词检索 + 向量语义检索
    结果融合后返回
    """
    # 1. ES 关键词检索
    es_results = es.search(
        index="articles",
        body={"query": {"multi_match": {"query": query, "fields": ["title", "content"]}}, "size": top_k}
    )

    # 2. 向量检索（假设已有向量检索函数）
    # vector_results = vector_search(query, top_k)

    # 3. 结果融合（RRF 算法或简单加权）
    # ...

    return merged_results
```

---

## 学习要点

1. **倒排索引**是 ES 的核心，词 → 文档列表的映射，实现快速全文检索
2. **中文必须用 IK 分词器**：`ik_max_word` 索引时用（最细粒度），`ik_smart` 搜索时用（智能切分）
3. **BM25** 是默认评分算法，考虑词频（TF）、逆文档频率（IDF）、文档长度
4. **字段类型选择**：需要分词搜索的用 `text`，需要精确匹配的用 `keyword`
5. **Bool 查询**是最常用的组合查询：`must`（必须匹配，影响评分）、`filter`（必须匹配，不影响评分）、`should`（可选匹配）、`must_not`（必须不匹配）
6. **高亮显示**用 `highlight` 配置，`pre_tags` 和 `post_tags` 自定义高亮标签
7. **聚合查询**用 `aggs` 配置，支持统计、分组、平均值等，`size: 0` 只返回聚合结果
8. **ES 和向量检索结合（混合检索）**效果最好：ES 负责精确关键词和过滤，向量负责语义召回
9. **批量操作**用 `elasticsearch.helpers.bulk`，比单条插入效率高很多
10. **索引刷新**：插入文档后需要 `es.indices.refresh()` 才能立即搜索到

## 扩展方向

- 学习 ES 集群部署（主节点、数据节点、协调节点）
- 探索 ES 分片和副本机制
- 学习 ES 索引生命周期管理（ILM）
- 探索 ES 跨集群搜索（CCS）和跨集群复制（CCR）
- 学习 ES 安全配置（用户认证、权限控制、HTTPS）
- 探索 ES 性能优化（查询优化、索引优化、缓存）
- 学习 Kibana 数据可视化
- 探索混合检索算法（RRF、加权融合、学习排序 LTR）
- 学习 ES 同义词、停用词、自定义词典配置
- 探索 ES 向量搜索（dense_vector 字段类型，8.x 原生支持）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/02-enterprise-backend/28-elasticsearch-fulltext-search

包含本文的完整可运行代码示例（ElasticSearch + IK 分词器 + Python 操作 + 7种查询类型 + 聚合统计）。

---

**上一篇**：[Docker Compose 部署](./27_基于Docker-Compose的部署.md) | **下一篇**：[混合检索 RAG](../03-第三阶段-检索增强与知识图谱/29_混合检索RAG.md)
