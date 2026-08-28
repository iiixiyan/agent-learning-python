# 企业级知识库项目：文档抽取 Neo4j 知识图谱的实体

> **Python 版** | LLM 实体关系抽取 + Neo4j 知识图谱存储 + 多跳推理检索 + GraphRAG
> 前置知识：[Neo4j 知识图谱和 Graph RAG](../03-第三阶段-检索增强与知识图谱/30_Neo4j知识图谱和Graph-RAG.md)、[企业级知识库项目介绍](./48_企业级知识库项目-项目介绍.md)

---

## 为什么需要知识图谱？

传统 RAG 基于向量相似度检索，擅长语义匹配，但有短板：

| 短板 | 说明 | 示例 |
|------|------|------|
| **无法推理实体关系** | 不能回答"A和B是什么关系" | "张三和李四是什么关系？" |
| **多跳问题回答差** | 需要跨多个文档推理的问题回答差 | "A的老板的下属是谁？" |
| **实体消歧困难** | 同名实体无法区分 | "苹果"是公司还是水果？ |
| **关系路径不可追溯** | 回答缺乏推理链路 | 无法展示"因为A→B→C，所以..." |

**知识图谱（Neo4j）** 补全这些能力，和向量检索结合形成 **GraphRAG**：

- 从文档中抽取**实体**和**关系**，存入图数据库
- 用户问题抽取实体，执行**多跳推理检索**
- 把推理链路和 RAG 结果融合送入 Prompt，提升复杂问题回答质量

这是一个可以写到简历上的亮点项目。

---

## 实体关系抽取流程

```
文档分块（Chunks）
    ↓
LLM 实体关系抽取（Prompt 工程）
    ↓
实体归一化（去重、消歧、合并）
    ↓
关系校验（过滤无效关系）
    ↓
Neo4j 存储（MERGE 幂等写入）
    ↓
知识图谱构建完成
```

### 各阶段职责

| 阶段 | 技术 | 职责 | 输出 |
|------|------|------|------|
| **实体抽取** | LLM | 从文本中识别命名实体 | 实体列表 |
| **关系抽取** | LLM | 识别实体之间的关系 | 三元组（头实体, 关系, 尾实体） |
| **实体归一化** | 规则+LLM | 去重、消歧、合并同名实体 | 归一化实体 |
| **关系校验** | 规则 | 过滤无效、重复关系 | 有效关系 |
| **Neo4j 存储** | Cypher | 幂等写入节点和关系 | 知识图谱 |

---

## LLM 实体关系抽取（完整实现）

### 1. Prompt 设计

```python
"""
entity_extractor.py - LLM 实体关系抽取
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


# 定义输出结构
class Entity(BaseModel):
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型，如：人物、组织、地点、产品、技术、概念")
    description: Optional[str] = Field(description="实体描述，可选")


class Relation(BaseModel):
    head: str = Field(description="头实体名称")
    relation: str = Field(description="关系类型，如：属于、工作于、投资、合作、包含")
    tail: str = Field(description="尾实体名称")
    description: Optional[str] = Field(description="关系描述，可选")


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(description="抽取的实体列表")
    relations: List[Relation] = Field(description="抽取的关系列表")


# 初始化
llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = JsonOutputParser(pydantic_object=ExtractionResult)

# Prompt 模板
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的信息抽取专家。从给定的文本中抽取实体和实体之间的关系。

抽取要求：
1. 实体类型包括：人物、组织、地点、产品、技术、概念、时间、数字
2. 关系类型包括：属于、工作于、投资、合作、包含、使用、研发、发布、收购、竞争
3. 只抽取文本中明确提到的实体和关系，不要推测
4. 实体名称要完整，不要缩写
5. 如果实体在文本中多次出现，只抽取一次

{format_instructions}"""),
    ("human", """请从以下文本中抽取实体和关系：

文本：
{text}

文档ID：{doc_id}
分块序号：{chunk_index}"""),
])


def extract_entities_and_relations(text: str, doc_id: str, chunk_index: int) -> dict:
    """
    从文本中抽取实体和关系

    Args:
        text: 文本内容
        doc_id: 文档ID
        chunk_index: 分块序号

    Returns:
        dict: 抽取结果，包含 entities 和 relations
    """
    chain = EXTRACTION_PROMPT | llm | parser

    result = chain.invoke({
        "text": text,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "format_instructions": parser.get_format_instructions(),
    })

    return result


# 使用示例
if __name__ == "__main__":
    text = """
    张三是字节跳动的高级工程师，负责开发 AI 知识库项目。
    该项目使用了 LangChain、FastAPI 和 PostgreSQL 技术栈。
    字节跳动投资了智谱AI，双方在大模型领域有合作。
    """

    result = extract_entities_and_relations(text, "doc_001", 0)
    print("抽取的实体：")
    for entity in result["entities"]:
        print(f"  - {entity['name']} ({entity['type']})")

    print("\n抽取的关系：")
    for relation in result["relations"]:
        print(f"  - {relation['head']} --{relation['relation']}--> {relation['tail']}")
```

### 2. 批量抽取

```python
"""
batch_extractor.py - 批量实体关系抽取
"""
from entity_extractor import extract_entities_and_relations
from typing import List, Dict
import time


def batch_extract(chunks: List[Dict], doc_id: str, delay: float = 0.5) -> Dict:
    """
    批量抽取文档所有分块的实体和关系

    Args:
        chunks: 分块列表，每个包含 content 和 chunk_index
        doc_id: 文档ID
        delay: 每次调用间隔（秒），避免限流

    Returns:
        dict: 合并后的抽取结果
    """
    all_entities = []
    all_relations = []

    for chunk in chunks:
        try:
            result = extract_entities_and_relations(
                text=chunk["content"],
                doc_id=doc_id,
                chunk_index=chunk["chunk_index"],
            )

            # 添加来源信息
            for entity in result["entities"]:
                entity["source_doc_id"] = doc_id
                entity["source_chunk_index"] = chunk["chunk_index"]
                all_entities.append(entity)

            for relation in result["relations"]:
                relation["source_doc_id"] = doc_id
                relation["source_chunk_index"] = chunk["chunk_index"]
                all_relations.append(relation)

            print(f"✅ 分块 {chunk['chunk_index']} 抽取完成: "
                  f"{len(result['entities'])} 实体, {len(result['relations'])} 关系")

        except Exception as e:
            print(f"❌ 分块 {chunk['chunk_index']} 抽取失败: {e}")

        # 避免限流
        time.sleep(delay)

    # 实体归一化
    normalized_entities = normalize_entities(all_entities)

    # 关系校验
    valid_relations = validate_relations(all_relations, normalized_entities)

    return {
        "entities": normalized_entities,
        "relations": valid_relations,
        "raw_entity_count": len(all_entities),
        "raw_relation_count": len(all_relations),
    }


def normalize_entities(entities: List[Dict]) -> List[Dict]:
    """
    实体归一化：去重、消歧、合并

    Args:
        entities: 原始实体列表

    Returns:
        list: 归一化后的实体列表
    """
    # 按名称去重（简单实现，实际可用 LLM 做消歧）
    entity_map = {}

    for entity in entities:
        name = entity["name"].strip()

        if name not in entity_map:
            entity_map[name] = {
                "name": name,
                "type": entity["type"],
                "description": entity.get("description", ""),
                "sources": [],
                "occurrence_count": 0,
            }

        # 合并来源
        source = {
            "doc_id": entity.get("source_doc_id"),
            "chunk_index": entity.get("source_chunk_index"),
        }
        if source not in entity_map[name]["sources"]:
            entity_map[name]["sources"].append(source)

        entity_map[name]["occurrence_count"] += 1

    return list(entity_map.values())


def validate_relations(relations: List[Dict], entities: List[Dict]) -> List[Dict]:
    """
    关系校验：过滤无效、重复关系

    Args:
        relations: 原始关系列表
        entities: 归一化后的实体列表

    Returns:
        list: 有效关系列表
    """
    entity_names = set(e["name"] for e in entities)
    valid_relations = []
    seen = set()

    for relation in relations:
        head = relation["head"].strip()
        tail = relation["tail"].strip()
        rel_type = relation["relation"].strip()

        # 过滤无效关系（实体不在抽取结果中）
        if head not in entity_names or tail not in entity_names:
            continue

        # 过滤自环
        if head == tail:
            continue

        # 去重
        key = (head, rel_type, tail)
        if key in seen:
            continue
        seen.add(key)

        valid_relations.append({
            "head": head,
            "relation": rel_type,
            "tail": tail,
            "description": relation.get("description", ""),
            "source_doc_id": relation.get("source_doc_id"),
            "source_chunk_index": relation.get("source_chunk_index"),
        })

    return valid_relations
```

---

## Neo4j 存储

### 1. Neo4j 连接

```python
"""
neo4j_store.py - Neo4j 知识图谱存储
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional


class Neo4jKnowledgeGraph:
    """Neo4j 知识图谱存储类"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_entity(self, entity: Dict):
        """
        创建实体节点（幂等：MERGE）

        Args:
            entity: 实体字典，包含 name, type, description, sources
        """
        with self.driver.session() as session:
            session.execute_write(self._create_entity_tx, entity)

    @staticmethod
    def _create_entity_tx(tx, entity: Dict):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.description = $description,
            e.occurrence_count = COALESCE(e.occurrence_count, 0) + $occurrence_count,
            e.updated_at = datetime()
        WITH e
        UNWIND $sources AS source
        MERGE (s:Source {doc_id: source.doc_id, chunk_index: source.chunk_index})
        MERGE (e)-[:MENTIONED_IN]->(s)
        """
        tx.run(query, **entity)

    def create_relation(self, relation: Dict):
        """
        创建实体关系（幂等：MERGE）

        Args:
            relation: 关系字典，包含 head, relation, tail, description
        """
        with self.driver.session() as session:
            session.execute_write(self._create_relation_tx, relation)

    @staticmethod
    def _create_relation_tx(tx, relation: Dict):
        # 关系类型作为关系标签（需要转义特殊字符）
        rel_type = relation["relation"].upper().replace(" ", "_").replace("-", "_")

        query = f"""
        MERGE (head:Entity {{name: $head}})
        MERGE (tail:Entity {{name: $tail}})
        MERGE (head)-[r:{rel_type}]->(tail)
        SET r.description = $description,
            r.source_doc_id = $source_doc_id,
            r.source_chunk_index = $source_chunk_index,
            r.updated_at = datetime()
        """
        tx.run(query, **relation)

    def batch_create(self, entities: List[Dict], relations: List[Dict]):
        """
        批量创建实体和关系

        Args:
            entities: 实体列表
            relations: 关系列表
        """
        print(f"📥 开始存储: {len(entities)} 实体, {len(relations)} 关系")

        for entity in entities:
            self.create_entity(entity)

        for relation in relations:
            self.create_relation(relation)

        print(f"✅ 存储完成")

    def query_multi_hop(self, entity_name: str, max_hops: int = 2) -> List[Dict]:
        """
        多跳推理查询

        Args:
            entity_name: 起始实体名称
            max_hops: 最大跳数

        Returns:
            list: 推理路径列表
        """
        with self.driver.session() as session:
            result = session.execute_read(
                self._query_multi_hop_tx, entity_name, max_hops
            )
            return result

    @staticmethod
    def _query_multi_hop_tx(tx, entity_name: str, max_hops: int) -> List[Dict]:
        query = """
        MATCH path = (start:Entity {name: $entity_name})-[*1..$max_hops]->(end:Entity)
        WHERE start <> end
        RETURN path,
               [node IN nodes(path) | node.name] AS entity_names,
               [rel IN relationships(path) | type(rel)] AS relation_types,
               length(path) AS hop_count
        ORDER BY hop_count
        LIMIT 50
        """
        result = tx.run(query, entity_name=entity_name, max_hops=max_hops)

        paths = []
        for record in result:
            paths.append({
                "entities": record["entity_names"],
                "relations": record["relation_types"],
                "hop_count": record["hop_count"],
            })

        return paths

    def query_related_entities(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """
        查询直接关联的实体

        Args:
            entity_name: 实体名称
            limit: 返回数量

        Returns:
            list: 关联实体列表
        """
        with self.driver.session() as session:
            result = session.execute_read(
                self._query_related_tx, entity_name, limit
            )
            return result

    @staticmethod
    def _query_related_tx(tx, entity_name: str, limit: int) -> List[Dict]:
        query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(related:Entity)
        RETURN related.name AS name,
               related.type AS type,
               type(r) AS relation,
               r.description AS description
        LIMIT $limit
        """
        result = tx.run(query, entity_name=entity_name, limit=limit)

        entities = []
        for record in result:
            entities.append({
                "name": record["name"],
                "type": record["type"],
                "relation": record["relation"],
                "description": record["description"],
            })

        return entities

    def get_graph_stats(self) -> Dict:
        """获取图谱统计信息"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WITH count(n) AS node_count
                MATCH ()-[r]->()
                RETURN node_count, count(r) AS relation_count
            """)
            record = result.single()
            return {
                "node_count": record["node_count"],
                "relation_count": record["relation_count"],
            }
```

### 2. 使用示例

```python
"""
main.py - 实体关系抽取 + Neo4j 存储 完整流程
"""
from batch_extractor import batch_extract
from neo4j_store import Neo4jKnowledgeGraph


# 初始化 Neo4j
kg = Neo4jKnowledgeGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)

# 模拟文档分块
chunks = [
    {
        "content": "张三是字节跳动的高级工程师，负责开发 AI 知识库项目。",
        "chunk_index": 0,
    },
    {
        "content": "该项目使用了 LangChain、FastAPI 和 PostgreSQL 技术栈。",
        "chunk_index": 1,
    },
    {
        "content": "字节跳动投资了智谱AI，双方在大模型领域有合作。",
        "chunk_index": 2,
    },
]

# 1. 批量抽取
result = batch_extract(chunks, doc_id="doc_001")

print(f"\n抽取结果:")
print(f"  实体: {len(result['entities'])} 个")
for entity in result["entities"]:
    print(f"    - {entity['name']} ({entity['type']})")

print(f"  关系: {len(result['relations'])} 个")
for relation in result["relations"]:
    print(f"    - {relation['head']} --{relation['relation']}--> {relation['tail']}")

# 2. 存储到 Neo4j
kg.batch_create(result["entities"], result["relations"])

# 3. 查询图谱统计
stats = kg.get_graph_stats()
print(f"\n图谱统计: {stats['node_count']} 节点, {stats['relation_count']} 关系")

# 4. 多跳推理查询
paths = kg.query_multi_hop("张三", max_hops=2)
print(f"\n从 '张三' 出发的推理路径:")
for path in paths[:5]:
    path_str = " → ".join(
        f"{e}({r})" if i < len(path["relations"]) else e
        for i, (e, r) in enumerate(zip(path["entities"], path["relations"] + [""]))
    )
    print(f"  [{path['hop_count']}跳] {path_str}")

# 5. 查询关联实体
related = kg.query_related_entities("字节跳动")
print(f"\n与 '字节跳动' 关联的实体:")
for entity in related:
    print(f"  - {entity['name']} ({entity['type']}) --{entity['relation']}-->")

kg.close()
```

---

## GraphRAG：知识图谱 + RAG

### 1. 问题实体抽取

```python
"""
graph_rag.py - GraphRAG 实现
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from neo4j_store import Neo4jKnowledgeGraph


class QueryEntities(BaseModel):
    entities: List[str] = Field(description="问题中的实体列表")
    intent: str = Field(description="问题意图，如：关系查询、多跳推理、事实查询")


llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = JsonOutputParser(pydantic_object=QueryEntities)

QUERY_ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个问题分析专家。从用户问题中抽取关键实体，并判断问题意图。

意图类型：
- 关系查询：询问两个实体之间的关系
- 多跳推理：需要跨多个实体推理
- 事实查询：查询实体的属性或事实
- 普通查询：不需要知识图谱的普通问题

{format_instructions}"""),
    ("human", "用户问题：{question}"),
])


def extract_query_entities(question: str) -> dict:
    """从问题中抽取实体和意图"""
    chain = QUERY_ENTITY_PROMPT | llm | parser
    return chain.invoke({
        "question": question,
        "format_instructions": parser.get_format_instructions(),
    })
```

### 2. 图谱检索 + RAG 融合

```python
def graph_rag_query(question: str, vector_results: List[Dict], kg: Neo4jKnowledgeGraph) -> dict:
    """
    GraphRAG：知识图谱检索 + 向量检索结果融合

    Args:
        question: 用户问题
        vector_results: 向量检索结果
        kg: Neo4j 知识图谱实例

    Returns:
        dict: 增强后的上下文
    """
    # 1. 抽取问题实体
    query_info = extract_query_entities(question)
    entities = query_info["entities"]
    intent = query_info["intent"]

    print(f"🔍 问题实体: {entities}")
    print(f"🎯 问题意图: {intent}")

    # 2. 图谱检索
    graph_context = []

    if intent in ["关系查询", "多跳推理"]:
        for entity in entities:
            # 多跳推理
            paths = kg.query_multi_hop(entity, max_hops=2)
            for path in paths:
                path_str = " → ".join(
                    f"{e}" if i == 0 else f"[{path['relations'][i-1]}]→ {e}"
                    for i, e in enumerate(path["entities"])
                )
                graph_context.append(f"[图谱推理-{path['hop_count']}跳] {path_str}")

            # 直接关联实体
            related = kg.query_related_entities(entity, limit=5)
            for rel in related:
                graph_context.append(
                    f"[图谱关联] {entity} --[{rel['relation']}]--> {rel['name']} ({rel['type']})"
                )

    # 3. 融合上下文
    enhanced_context = {
        "vector_results": vector_results,
        "graph_results": graph_context,
        "query_entities": entities,
        "query_intent": intent,
    }

    return enhanced_context


def build_graph_rag_prompt(question: str, enhanced_context: dict) -> str:
    """构建 GraphRAG Prompt"""
    # 向量检索结果
    vector_text = ""
    for i, result in enumerate(enhanced_context["vector_results"], 1):
        vector_text += f"[向量检索{i}] {result['content']}\n\n"

    # 图谱检索结果
    graph_text = ""
    if enhanced_context["graph_results"]:
        graph_text = "知识图谱推理结果：\n"
        for result in enhanced_context["graph_results"]:
            graph_text += f"{result}\n"

    prompt = f"""你是一个专业的知识库问答助手。根据以下参考资料回答用户问题。

【向量检索结果】
{vector_text}

【知识图谱推理结果】
{graph_text}

【问题分析】
- 关键实体：{', '.join(enhanced_context['query_entities'])}
- 问题意图：{enhanced_context['query_intent']}

用户问题：{question}

回答要求：
1. 综合向量检索和知识图谱推理结果回答
2. 如果是关系查询或多跳推理，优先使用图谱推理结果
3. 引用来源，如 [向量检索1] [图谱推理-2跳]
4. 如果参考资料中没有相关信息，回答"知识库中未找到相关信息"
5. 用中文回答，简洁准确"""

    return prompt
```

---

## 关键优化点

### 1. 实体消歧

| 方法 | 说明 | 效果 |
|------|------|------|
| **规则消歧** | 基于实体类型、上下文规则 | 简单场景有效 |
| **LLM 消歧** | 用 LLM 判断两个实体是否同一 | 准确率高，成本高 |
| **实体链接** | 链接到知识库（如 Wikidata） | 专业领域效果好 |

### 2. 关系抽取优化

- **Few-shot 示例**：在 Prompt 中提供抽取示例，提升准确率
- **分类型抽取**：先抽实体，再抽关系，分步进行
- **批量抽取**：多个分块合并抽取，减少 LLM 调用
- **增量更新**：只处理新增/修改的分块，避免重复抽取

### 3. 图谱质量控制

- **实体覆盖率**：统计文档中实体被抽取的比例
- **关系准确率**：人工抽样验证关系正确性
- **图谱密度**：节点数/关系数比例，避免过稀或过密
- **社区发现**：用 Louvain 算法发现实体社区

---

## 学习要点

1. **知识图谱是企业级知识库的亮点**：实体关系抽取 + 多跳推理，是简历加分项
2. **LLM 是实体关系抽取的核心**：用 Prompt 工程 + JSON 输出解析，从文本中抽取实体和关系
3. **实体归一化很重要**：去重、消歧、合并，避免图谱中出现大量重复节点
4. **Neo4j 用 MERGE 实现幂等写入**：重复运行不会创建重复节点和关系
5. **多跳推理是知识图谱的核心能力**：从起始实体出发，沿关系路径推理，回答复杂问题
6. **GraphRAG 融合向量检索和图谱推理**：关系查询和多跳推理用图谱，事实查询用向量
7. **问题实体抽取是 GraphRAG 的第一步**：从用户问题中抽取实体，决定是否需要图谱推理
8. **图谱质量决定回答质量**：实体覆盖率、关系准确率直接影响推理结果
9. **批量抽取和增量更新是性能关键**：减少 LLM 调用，降低成本
10. **图谱可视化是展示亮点**：用 Neo4j Browser 或 ECharts 可视化知识图谱

## 扩展方向

- 学习知识图谱构建（实体抽取、关系抽取、实体链接、知识融合）
- 探索 GraphRAG 高级技术（LightRAG、Hierarchical GraphRAG、社区摘要）
- 学习图神经网络（GNN、GraphSAGE、GAT、知识图谱嵌入）
- 探索图算法（PageRank、社区发现、最短路径、中心性分析）
- 学习实体链接（Entity Linking、Wikidata、知识图谱对齐）
- 探索关系抽取（远程监督、弱监督、主动学习）
- 学习图谱嵌入（TransE、TransH、RotatE、ComplEx）
- 探索多模态知识图谱（图像实体、视频实体、跨模态关系）
- 学习图谱问答（KBQA、语义解析、SPARQL生成）
- 探索知识图谱推理（规则推理、嵌入推理、LLM推理）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/05-resume-interview/53-kb-neo4j-entity

包含本文的完整实体关系抽取代码（LLM抽取、批量处理、实体归一化）、Neo4j存储（幂等写入、多跳查询、关联查询）、GraphRAG实现（问题实体抽取、图谱检索、Prompt融合）。

---

**上一篇**：[全文检索链路](./52_企业级知识库项目-全文检索链路.md) | **下一篇**：[文档审核机制](./54_企业级知识库项目-文档审核机制.md)
