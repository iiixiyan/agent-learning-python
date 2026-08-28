# Neo4j 知识图谱和 Graph RAG

> **Python 版** | 基于 Neo4j + Python neo4j-driver + LangChain 技术栈
> 前置知识：[RAG 基础](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)、[混合检索 RAG](./29_混合检索RAG.md)

---

## 为什么需要知识图谱？

传统 RAG（向量检索）擅长语义匹配，但有局限性：

| 维度 | 向量检索 RAG | 知识图谱 Graph RAG |
|------|-------------|-------------------|
| **检索方式** | 语义相似度 | 实体关系推理 |
| **擅长场景** | 模糊查询、同义表达 | 精确关系、多跳推理 |
| **可解释性** | 低（黑盒） | 高（关系路径可见） |
| **结构化数据** | 不擅长 | 天然支持 |
| **多跳推理** | 困难 | 原生支持 |
| **构建成本** | 低（自动切分嵌入） | 高（需要实体抽取） |

### 实际案例

| 用户问题 | 向量检索 | 知识图谱 |
|----------|----------|----------|
| "张三的公司是做什么的？" | ⚠️ 可能匹配到张三的介绍 | ✅ 张三 → 任职于 → 公司 → 业务 |
| "A公司的CEO的毕业院校是？" | ❌ 需要多跳，向量检索困难 | ✅ A公司 → CEO → 张三 → 毕业于 → 院校 |
| "和Python相关的技术有哪些？" | ✅ 语义匹配好 | ✅ Python → 关联 → FastAPI/Django/Flask |

**结论**：知识图谱能实现推理式的检索，用到这种技术后的 RAG 叫做 **Graph RAG**，是可以写到简历上的亮点技术。

---

## 知识图谱基础概念

### 什么是知识图谱？

知识图谱是一种用图结构表示知识的方式，由**节点（Node）**和**关系（Relationship）**组成：

```
节点（Node）：实体，如"张三"、"公司A"、"Python"
关系（Relationship）：实体之间的联系，如"任职于"、"开发了"、"关联"

示例：
  [张三] --任职于--> [公司A]
  [张三] --开发了--> [产品X]
  [产品X] --使用--> [Python]
  [Python] --关联--> [FastAPI]
```

### 核心概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **节点（Node）** | 实体，表示事物 | 人、公司、技术、产品 |
| **关系（Relationship）** | 实体之间的联系 | 任职于、开发了、使用、关联 |
| **属性（Property）** | 节点或关系的特征 | 姓名、年龄、创建时间 |
| **标签（Label）** | 节点的类型 | :Person、:Company、:Technology |
| **路径（Path）** | 节点和关系组成的序列 | 张三 → 任职于 → 公司A |

---

## Neo4j 入门

### 什么是 Neo4j？

Neo4j 是最流行的图数据库，专为存储和查询知识图谱设计：

| 特性 | 说明 |
|------|------|
| **原生图存储** | 节点和关系直接存储，无需 JOIN |
| **Cypher 查询语言** | 声明式图查询语言，语法直观 |
| **ACID 事务** | 支持完整的事务特性 |
| **高可用集群** | 支持主从复制和集群部署 |
| **可视化浏览器** | Neo4j Browser 自带图可视化 |

### Docker 启动 Neo4j

```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15.0
    container_name: neo4j-dev
    environment:
      - NEO4J_AUTH=neo4j/password123  # 用户名/密码
      - NEO4J_PLUGINS=["apoc"]          # 安装 APOC 插件
    ports:
      - "7474:7474"   # HTTP 端口（浏览器访问）
      - "7687:7687"   # Bolt 端口（Python 驱动连接）
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: always

volumes:
  neo4j_data:
  neo4j_logs:
```

```bash
# 启动 Neo4j
docker compose up -d

# 访问 Neo4j Browser
# 浏览器打开 http://localhost:7474
# 用户名: neo4j, 密码: password123
```

---

## Cypher 查询语言

Cypher 是 Neo4j 的查询语言，用 ASCII 艺术表示图模式：

| 符号 | 含义 | 示例 |
|------|------|------|
| `(n)` | 节点 | `(p:Person)` |
| `-[r]->` | 关系 | `-[r:WORKS_AT]->` |
| `{key: value}` | 属性 | `{name: "张三"}` |
| `:Label` | 标签 | `:Person` |

### 常用查询示例

```cypher
-- 1. 创建节点
CREATE (p:Person {name: "张三", age: 30})
CREATE (c:Company {name: "字节跳动", industry: "互联网"})

-- 2. 创建关系
MATCH (p:Person {name: "张三"}), (c:Company {name: "字节跳动"})
CREATE (p)-[r:WORKS_AT {position: "工程师", since: 2020}]->(c)

-- 3. 查询节点
MATCH (p:Person) RETURN p

-- 4. 查询关系（1跳）
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
RETURN p.name, r.position, c.name

-- 5. 多跳查询（2跳）
MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:USES]->(t:Technology)
RETURN p.name, c.name, t.name

-- 6. 条件查询
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
WHERE r.since >= 2020 AND c.industry = "互联网"
RETURN p.name, c.name, r.position

-- 7. 更新节点
MATCH (p:Person {name: "张三"})
SET p.age = 31, p.city = "北京"

-- 8. 删除节点和关系
MATCH (p:Person {name: "张三"})
DETACH DELETE p  -- DETACH 会先删除关系再删除节点

-- 9. 统计查询
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, count(p) as employee_count
ORDER BY employee_count DESC

-- 10. 最短路径
MATCH path = shortestPath(
  (p1:Person {name: "张三"})-[*]-(p2:Person {name: "李四"})
)
RETURN path
```

---

## Python 操作 Neo4j

### 安装依赖

```bash
pip install neo4j==5.16.0 python-dotenv
```

### 完整示例

创建 `neo4j_demo.py`：

```python
"""
neo4j_demo.py - Python 操作 Neo4j 完整示例
包含：连接、创建节点、创建关系、查询、多跳推理、统计
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jDemo:
    """Neo4j 操作示例类"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化 Neo4j 驱动

        Args:
            uri: Neo4j Bolt 地址，如 bolt://localhost:7687
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("✅ Neo4j 连接成功")

    def close(self):
        """关闭连接"""
        self.driver.close()
        print("✅ Neo4j 连接已关闭")

    # ========== 1. 创建节点 ==========

    def create_person(self, name: str, age: int, city: str = ""):
        """创建人物节点"""
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (p:Person {name: $name, age: $age, city: $city})
                RETURN p
                """,
                name=name, age=age, city=city
            )
            node = result.single()["p"]
            print(f"✅ 创建人物: {node['name']} (年龄: {node['age']})")
            return node

    def create_company(self, name: str, industry: str):
        """创建公司节点"""
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (c:Company {name: $name, industry: $industry})
                RETURN c
                """,
                name=name, industry=industry
            )
            node = result.single()["c"]
            print(f"✅ 创建公司: {node['name']} (行业: {node['industry']})")
            return node

    def create_technology(self, name: str, category: str):
        """创建技术节点"""
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (t:Technology {name: $name, category: $category})
                RETURN t
                """,
                name=name, category=category
            )
            node = result.single()["t"]
            print(f"✅ 创建技术: {node['name']} (分类: {node['category']})")
            return node

    # ========== 2. 创建关系 ==========

    def create_works_at(self, person_name: str, company_name: str, position: str, since: int):
        """创建任职关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})
                MATCH (c:Company {name: $company_name})
                CREATE (p)-[r:WORKS_AT {position: $position, since: $since}]->(c)
                RETURN p.name, r.position, c.name
                """,
                person_name=person_name, company_name=company_name,
                position=position, since=since
            )
            record = result.single()
            print(f"✅ 创建关系: {record['p.name']} --{record['r.position']}--> {record['c.name']}")

    def create_uses_technology(self, company_name: str, tech_name: str):
        """创建公司使用技术的关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company {name: $company_name})
                MATCH (t:Technology {name: $tech_name})
                CREATE (c)-[r:USES]->(t)
                RETURN c.name, t.name
                """,
                company_name=company_name, tech_name=tech_name
            )
            record = result.single()
            print(f"✅ 创建关系: {record['c.name']} --使用--> {record['t.name']}")

    def create_related_tech(self, tech1: str, tech2: str):
        """创建技术之间的关联关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t1:Technology {name: $tech1})
                MATCH (t2:Technology {name: $tech2})
                CREATE (t1)-[r:RELATED_TO]->(t2)
                RETURN t1.name, t2.name
                """,
                tech1=tech1, tech2=tech2
            )
            record = result.single()
            print(f"✅ 创建关系: {record['t1.name']} --关联--> {record['t2.name']}")

    # ========== 3. 查询 ==========

    def query_all_persons(self):
        """查询所有人物"""
        with self.driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN p ORDER BY p.name")
            print("\n📋 所有人物:")
            for record in result:
                p = record["p"]
                print(f"  - {p['name']} (年龄: {p['age']}, 城市: {p.get('city', '未知')})")

    def query_persons_by_company(self, company_name: str):
        """查询某公司的所有员工（1跳查询）"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person)-[r:WORKS_AT]->(c:Company {name: $company_name})
                RETURN p.name, r.position, r.since
                ORDER BY r.since
                """,
                company_name=company_name
            )
            print(f"\n🏢 {company_name} 的员工:")
            for record in result:
                print(f"  - {record['p.name']} ({record['r.position']}, 入职: {record['r.since']})")

    def query_tech_by_person(self, person_name: str):
        """查询某人所在公司使用的技术（2跳查询）"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})-[:WORKS_AT]->(c:Company)-[:USES]->(t:Technology)
                RETURN DISTINCT t.name, t.category
                ORDER BY t.category, t.name
                """,
                person_name=person_name
            )
            print(f"\n🔧 {person_name} 所在公司使用的技术:")
            for record in result:
                print(f"  - {record['t.name']} ({record['t.category']})")

    def query_related_tech(self, tech_name: str):
        """查询与某技术相关的其他技术（多跳查询）"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Technology {name: $tech_name})-[r:RELATED_TO*1..2]-(related:Technology)
                WHERE related.name <> $tech_name
                RETURN DISTINCT related.name, related.category
                LIMIT 10
                """,
                tech_name=tech_name
            )
            print(f"\n🔗 与 {tech_name} 相关的技术:")
            for record in result:
                print(f"  - {record['related.name']} ({record['related.category']})")

    def query_company_stats(self):
        """统计各公司员工数量"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person)-[:WORKS_AT]->(c:Company)
                RETURN c.name, count(p) as employee_count
                ORDER BY employee_count DESC
                """
            )
            print("\n📊 公司员工统计:")
            for record in result:
                print(f"  - {record['c.name']}: {record['employee_count']} 人")

    # ========== 4. 清空数据 ==========

    def clear_all(self):
        """清空所有数据（谨慎使用）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  所有数据已清空")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 连接 Neo4j
    demo = Neo4jDemo(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password123"),
    )

    try:
        # 清空旧数据
        demo.clear_all()

        # 1. 创建节点
        print("\n" + "="*60)
        print("创建节点")
        print("="*60)
        demo.create_person("张三", 30, "北京")
        demo.create_person("李四", 28, "上海")
        demo.create_person("王五", 32, "深圳")
        demo.create_company("字节跳动", "互联网")
        demo.create_company("阿里巴巴", "电商")
        demo.create_technology("Python", "编程语言")
        demo.create_technology("FastAPI", "Web框架")
        demo.create_technology("LangChain", "AI框架")
        demo.create_technology("Neo4j", "数据库")

        # 2. 创建关系
        print("\n" + "="*60)
        print("创建关系")
        print("="*60)
        demo.create_works_at("张三", "字节跳动", "高级工程师", 2020)
        demo.create_works_at("李四", "字节跳动", "工程师", 2021)
        demo.create_works_at("王五", "阿里巴巴", "架构师", 2019)
        demo.create_uses_technology("字节跳动", "Python")
        demo.create_uses_technology("字节跳动", "FastAPI")
        demo.create_uses_technology("字节跳动", "LangChain")
        demo.create_uses_technology("阿里巴巴", "Python")
        demo.create_uses_technology("阿里巴巴", "Neo4j")
        demo.create_related_tech("Python", "FastAPI")
        demo.create_related_tech("Python", "LangChain")
        demo.create_related_tech("FastAPI", "LangChain")

        # 3. 查询
        print("\n" + "="*60)
        print("查询演示")
        print("="*60)
        demo.query_all_persons()
        demo.query_persons_by_company("字节跳动")
        demo.query_tech_by_person("张三")
        demo.query_related_tech("Python")
        demo.query_company_stats()

    finally:
        demo.close()
```

### 运行示例

```bash
# 1. 确保 Neo4j 已启动
docker compose up -d

# 2. 创建 .env 文件
echo "NEO4J_URI=bolt://localhost:7687" > .env
echo "NEO4J_USER=neo4j" >> .env
echo "NEO4J_PASSWORD=password123" >> .env

# 3. 运行示例
python neo4j_demo.py
```

---

## Graph RAG 实现

### Graph RAG 架构

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────┐
│              实体抽取 (LLM)                       │
│  从问题中提取实体和关系，如"张三的公司用什么技术"  │
│  → 实体: 张三                                    │
│  → 关系: 任职于、使用                             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           图查询 (Cypher)                         │
│  根据实体和关系生成 Cypher 查询，从知识图谱检索    │
│  MATCH (p:Person {name:"张三"})                  │
│  -[:WORKS_AT]->(c:Company)-[:USES]->(t:Tech)   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           结果融合 + 上下文构建                    │
│  将图查询结果和向量检索结果融合，构建上下文        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           大模型生成 (LLM)                        │
│  基于上下文生成回答，引用知识图谱中的关系路径      │
└─────────────────────────────────────────────────┘
```

### 完整 Graph RAG 实现

创建 `graph_rag.py`：

```python
"""
graph_rag.py - Graph RAG 完整实现
包含：实体抽取、图查询、结果融合、大模型生成
"""
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class GraphRAG:
    """Graph RAG 实现类"""

    def __init__(self):
        # Neo4j 连接
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "password123"))
        )

        # 大模型
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "qwen-plus"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0,
        )

        print("✅ Graph RAG 初始化完成")

    def close(self):
        """关闭连接"""
        self.driver.close()

    # ========== 1. 实体抽取 ==========

    def extract_entities(self, query: str) -> Dict:
        """
        从用户问题中抽取实体和关系

        Args:
            query: 用户问题

        Returns:
            Dict: 包含 entities 和 relations 的字典
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个实体抽取专家。从用户问题中抽取实体和关系。

知识图谱中的节点类型：
- Person: 人物，属性有 name, age, city
- Company: 公司，属性有 name, industry
- Technology: 技术，属性有 name, category

关系类型：
- WORKS_AT: 人物任职于公司，属性有 position, since
- USES: 公司使用技术
- RELATED_TO: 技术之间关联

请以 JSON 格式输出，格式如下：
{
  "entities": [{"type": "Person", "name": "张三"}],
  "relations": ["WORKS_AT", "USES"],
  "query_type": "multi_hop"  // single_hop, multi_hop, stats
}"""),
            ("human", "用户问题：{query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": query})

        # 解析 JSON（简化处理，实际应该用 JSON 解析器）
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"entities": [], "relations": [], "query_type": "single_hop"}

        print(f"  实体抽取: {result}")
        return result

    # ========== 2. 图查询 ==========

    def query_graph(self, entities: List[Dict], relations: List[str], query_type: str) -> List[Dict]:
        """
        根据实体和关系执行图查询

        Args:
            entities: 实体列表
            relations: 关系列表
            query_type: 查询类型

        Returns:
            List[Dict]: 查询结果
        """
        results = []

        with self.driver.session() as session:
            # 根据实体类型生成不同的查询
            for entity in entities:
                if entity["type"] == "Person":
                    # 查询人物相关信息
                    if "USES" in relations or query_type == "multi_hop":
                        # 多跳：人物 → 公司 → 技术
                        result = session.run(
                            """
                            MATCH (p:Person {name: $name})-[:WORKS_AT]->(c:Company)
                            OPTIONAL MATCH (c)-[:USES]->(t:Technology)
                            RETURN p.name as person, c.name as company,
                                   collect(DISTINCT t.name) as technologies
                            """,
                            name=entity["name"]
                        )
                    else:
                        # 单跳：人物 → 公司
                        result = session.run(
                            """
                            MATCH (p:Person {name: $name})-[r:WORKS_AT]->(c:Company)
                            RETURN p.name as person, r.position as position,
                                   c.name as company, c.industry as industry
                            """,
                            name=entity["name"]
                        )

                    for record in result:
                        results.append(dict(record))

                elif entity["type"] == "Company":
                    # 查询公司相关信息
                    result = session.run(
                        """
                        MATCH (c:Company {name: $name})
                        OPTIONAL MATCH (p:Person)-[:WORKS_AT]->(c)
                        OPTIONAL MATCH (c)-[:USES]->(t:Technology)
                        RETURN c.name as company, c.industry as industry,
                               collect(DISTINCT p.name) as employees,
                               collect(DISTINCT t.name) as technologies
                        """,
                        name=entity["name"]
                    )
                    for record in result:
                        results.append(dict(record))

                elif entity["type"] == "Technology":
                    # 查询技术相关信息
                    result = session.run(
                        """
                        MATCH (t:Technology {name: $name})
                        OPTIONAL MATCH (c:Company)-[:USES]->(t)
                        OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Technology)
                        RETURN t.name as tech, t.category as category,
                               collect(DISTINCT c.name) as companies,
                               collect(DISTINCT related.name) as related_tech
                        """,
                        name=entity["name"]
                    )
                    for record in result:
                        results.append(dict(record))

        print(f"  图查询结果: {len(results)} 条")
        return results

    # ========== 3. 构建上下文 ==========

    def build_context(self, graph_results: List[Dict]) -> str:
        """
        将图查询结果转换为文本上下文

        Args:
            graph_results: 图查询结果

        Returns:
            str: 文本上下文
        """
        context_parts = []

        for i, result in enumerate(graph_results, 1):
            parts = []
            for key, value in result.items():
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value if v)
                if value:
                    parts.append(f"{key}: {value}")
            context_parts.append(f"[{i}] " + "; ".join(parts))

        return "\n\n".join(context_parts)

    # ========== 4. 大模型生成 ==========

    def generate_answer(self, query: str, context: str) -> str:
        """
        基于图查询上下文生成回答

        Args:
            query: 用户问题
            context: 图查询上下文

        Returns:
            str: 回答
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个知识图谱问答助手。根据知识图谱查询结果回答用户问题。
如果查询结果中没有相关信息，请如实说明。
回答要简洁明了，引用查询结果中的信息。"""),
            ("human", """知识图谱查询结果：
{context}

用户问题：{query}

回答：""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        return response.content

    # ========== 5. 完整问答流程 ==========

    def answer(self, query: str) -> str:
        """
        Graph RAG 完整问答流程

        Args:
            query: 用户问题

        Returns:
            str: 回答
        """
        print(f"\n{'='*60}")
        print(f"问题: {query}")
        print(f"{'='*60}")

        # Step 1: 实体抽取
        print("\n[Step 1] 实体抽取...")
        extracted = self.extract_entities(query)

        # Step 2: 图查询
        print("\n[Step 2] 图查询...")
        graph_results = self.query_graph(
            extracted["entities"],
            extracted["relations"],
            extracted["query_type"]
        )

        # Step 3: 构建上下文
        print("\n[Step 3] 构建上下文...")
        context = self.build_context(graph_results)
        print(f"  上下文:\n{context}")

        # Step 4: 生成回答
        print("\n[Step 4] 生成回答...")
        answer = self.generate_answer(query, context)

        print(f"\n回答:\n{answer}")
        print(f"\n{'='*60}\n")

        return answer


# ========== 使用示例 ==========

if __name__ == "__main__":
    graph_rag = GraphRAG()

    try:
        questions = [
            "张三在哪个公司工作？",
            "张三所在的公司使用什么技术？",
            "字节跳动有哪些员工？",
            "和Python相关的技术有哪些？",
        ]

        for q in questions:
            graph_rag.answer(q)

    finally:
        graph_rag.close()
```

---

## 知识图谱构建流程

### 从文本构建知识图谱

```python
"""
build_knowledge_graph.py - 从文本自动构建知识图谱
使用大模型抽取实体和关系，写入 Neo4j
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def extract_entities_relations(text: str, llm: ChatOpenAI) -> Dict:
    """
    从文本中抽取实体和关系

    Args:
        text: 输入文本
        llm: 大模型

    Returns:
        Dict: 包含 nodes 和 edges 的字典
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个知识图谱构建专家。从文本中抽取实体和关系。

节点类型：Person, Company, Technology, Product, Location
关系类型：WORKS_AT, USES, DEVELOPED, RELATED_TO, LOCATED_IN

请以 JSON 格式输出：
{
  "nodes": [{"type": "Person", "name": "张三", "properties": {"age": 30}}],
  "edges": [{"source": "张三", "source_type": "Person",
             "target": "字节跳动", "target_type": "Company",
             "relation": "WORKS_AT", "properties": {"position": "工程师"}}]
}"""),
        ("human", "文本：{text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"text": text})

    import json
    import re
    json_match = re.search(r'\{[\s\S]*\}', response.content)
    if json_match:
        return json.loads(json_match.group())
    return {"nodes": [], "edges": []}


def write_to_neo4j(data: Dict, driver: GraphDatabase):
    """
    将抽取的实体和关系写入 Neo4j

    Args:
        data: 包含 nodes 和 edges 的字典
        driver: Neo4j 驱动
    """
    with driver.session() as session:
        # 创建节点
        for node in data["nodes"]:
            props = ", ".join([f"{k}: ${k}" for k in node.get("properties", {})])
            props_str = f", {props}" if props else ""
            session.run(
                f"MERGE (n:{node['type']} {{name: $name}}) SET n += {{properties}}",
                name=node["name"],
                properties=node.get("properties", {})
            )
            print(f"  节点: {node['type']} - {node['name']}")

        # 创建关系
        for edge in data["edges"]:
            session.run(
                f"""
                MATCH (s:{edge['source_type']} {{name: $source_name}})
                MATCH (t:{edge['target_type']} {{name: $target_name}})
                MERGE (s)-[r:{edge['relation']}]->(t)
                SET r += $properties
                """,
                source_name=edge["source"],
                target_name=edge["target"],
                properties=edge.get("properties", {})
            )
            print(f"  关系: {edge['source']} --{edge['relation']}--> {edge['target']}")


# 使用示例
if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"),
              os.getenv("NEO4J_PASSWORD", "password123"))
    )

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    text = """
    张三是字节跳动的高级工程师，专注于 Python 和 FastAPI 开发。
    字节跳动使用 LangChain 框架构建 AI 应用，LangChain 和 Python 密切相关。
    李四也在字节跳动工作，是一名前端工程师。
    """

    print("从文本抽取实体和关系...")
    data = extract_entities_relations(text, llm)

    print("\n写入 Neo4j...")
    write_to_neo4j(data, driver)

    driver.close()
    print("\n✅ 知识图谱构建完成！")
```

---

## 学习要点

1. **知识图谱**由节点（实体）和关系（联系）组成，适合表示结构化知识
2. **Neo4j** 是最流行的图数据库，用 Cypher 查询语言操作图
3. **Cypher 语法**用 ASCII 艺术表示图模式：`(n)-[r]->(m)`
4. **多跳推理**是知识图谱的核心优势：人物 → 公司 → 技术，2跳查询
5. **Graph RAG** = 实体抽取 + 图查询 + 结果融合 + 大模型生成
6. **实体抽取**用大模型从问题中提取实体和关系类型
7. **图查询**根据实体类型和关系生成 Cypher 查询
8. **知识图谱构建**可以用大模型从文本自动抽取实体和关系
9. **APOC 插件**提供丰富的图算法和工具函数，建议安装
10. **Graph RAG 和向量 RAG 结合**效果最好：向量负责语义召回，图谱负责关系推理

## 扩展方向

- 学习更多 Cypher 高级语法（路径查询、图算法、模式匹配）
- 探索 Neo4j APOC 插件的图算法（社区发现、中心性分析、路径搜索）
- 学习知识图谱嵌入（Knowledge Graph Embedding）：TransE、ComplEx、RotatE
- 探索 GraphRAG 框架（微软 GraphRAG、LlamaIndex GraphRAG）
- 学习实体链接（Entity Linking）和实体消歧
- 探索时序知识图谱（Temporal Knowledge Graph）
- 学习知识图谱质量评估和清洗
- 探索多模态知识图谱（文本 + 图片 + 视频）
- 学习 Neo4j 集群部署和性能优化
- 探索知识图谱和向量数据库的混合检索方案

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/30-neo4j-graphrag

包含本文的完整可运行代码示例（Neo4j 操作 + Cypher 查询 + Graph RAG 实现 + 知识图谱自动构建）。

---

**上一篇**：[混合检索 RAG](./29_混合检索RAG.md) | **下一篇**：[第三阶段总结](../03-第三阶段-检索增强与知识图谱/31_第三阶段总结.md)
