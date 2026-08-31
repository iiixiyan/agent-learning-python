# 向量数据库 Milvus：AI Agent 开发必备技术

> **Python 版** | 基于 FastAPI + LangChain Python + pymilvus 技术栈

---

## 为什么需要向量数据库？

前面我们实现了 RAG：

```
文档向量化 → 向量数据库 → 查询向量化 → 相似度匹配 → 相关文档 → 增强 Prompt → 大模型回答
```

文档向量化放到向量数据库，每次查询根据向量化的 Query 去数据库做相似度匹配，查出相关文档放到 Prompt 里给大模型，大模型来生成回答。

但之前向量数据库是放在内存里的（InMemoryVectorStore），而实际上 AI Agent 产品都会用 Milvus 这种专业的向量数据库。

就像 Web 应用会把数据存在 MySQL 里，基于对数据的增删改查实现各种业务功能。而 AI Agent 应用会把知识、记忆放在 Milvus 数据库中，基于对知识的检索、增删改实现各种功能。

## MySQL vs Milvus

有同学可能会问，把数据存在 MySQL 里，和现在存在 Milvus 里有什么不同么？

| 对比项 | MySQL | Milvus |
|--------|-------|--------|
| 查询方式 | 根据 ID、关键词匹配 | 根据语义匹配，用自然语言检索 |
| 数据类型 | 结构化数据 | 向量数据 + 元数据 |
| 典型场景 | 精确查询、关联查询 | 语义搜索、RAG、相似度推荐 |
| 是否需要嵌入模型 | 不需要 | 检索、新增、修改需要；删除不需要 |

比如你做了一个 AI 日记本：
- 查询日记列表可以从 MySQL 来查，不走 AI
- 查询"我哪几天的日记心情比较好"，就要去 Milvus 做向量相似度检索，然后交给 AI 生成回答

所以一般会做 **MySQL 和 Milvus 的双写**，也就是同时对两个数据库做增删改，保持数据同步。

```
用户操作 → 同时写入 MySQL + Milvus
查询操作 → 精确查询走 MySQL，语义检索走 Milvus
```

这节我们先学下 Milvus，做下增删改查，跑通基于 Milvus 的 RAG 流程。

## 安装和启动 Milvus

### 安装 Docker

本地跑 Milvus 需要安装 Docker：
https://www.docker.com/

下载后安装，会有桌面端和命令行工具。如果 `docker` 命令可用了，就代表装好了。

### 下载 Milvus 配置文件

创建一个目录用来放 Milvus 的 Docker 配置文件和数据：

```bash
mkdir milvus-test
cd milvus-test
```

从这里下载 Milvus 的 Docker Compose 配置文件：
https://github.com/milvus-io/milvus/releases

下载 `milvus-standalone-docker-compose.yml` 文件。

### 启动 Milvus

```bash
docker compose -f ./milvus-standalone-docker-compose.yml up -d
```

用到的镜像根据配置文件自动下载。

Milvus 数据库跑在 **19530** 这个端口。

访问这个 URL 可以做健康度检查：
http://localhost:9091/healthz

### Attu：Milvus GUI 工具

Attu 是 Milvus 生态最好的 GUI 工具：
https://github.com/zilliztech/attu

下载安装后，用默认配置连接（localhost:19530），就可以看到所有的集合、集合下所有的 Entity。

**Milvus 安装与 Attu GUI 工具参考截图：**

![Milvus安装1](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/0_公众号_Yi昭.png)

![Milvus安装2](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/1_公众号_Yi昭.png)

![Milvus安装3](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/2_公众号_Yi昭.png)

![Milvus安装4](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/3_公众号_Yi昭.png)

![Milvus安装5](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/4_公众号_Yi昭.png)

![Milvus安装6](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/5_公众号_Yi昭.png)

![Milvus安装7](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/6_公众号_Yi昭.png)

![Milvus安装8](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/7_公众号_Yi昭.png)

![Milvus安装9](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/8_公众号_Yi昭.png)

![Milvus安装10](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/9_公众号_Yi昭.png)

## Milvus 数据结构

在 Milvus 里是这样存储数据的：

```
Database（数据库）
└── Collection（集合，类似 MySQL 的表）
    └── Entity（实体，类似 MySQL 的行）
        ├── 主键字段（id）
        ├── 向量字段（vector，FloatVector 类型）
        └── 其他元数据字段（content、date、mood、tags 等）
```

可以分为多个 Database，每个 Database 下有多个 Collection。每个 Collection 下是符合 Schema 的 Entity，也就是数据。

所以我们插入数据，就定义一个 Schema，然后插入 Entity 就好了。同时要建立一个向量字段的索引，用来快速查询。

## 实战：用 Python 操作 Milvus

### 安装依赖

```bash
pip install pymilvus langchain langchain-openai python-dotenv
```

### 配置文件 .env

```env
# OpenAI API 配置
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
EMBEDDINGS_MODEL_NAME=text-embedding-v3

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 第一步：创建集合并插入数据

创建 `src/insert_data.py`：

```python
"""
Milvus 插入数据示例：创建集合、建立索引、插入日记数据
"""
import os
from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 配置
COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024  # 向量维度，需要和嵌入模型一致

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=VECTOR_DIM,
)


def get_embedding(text: str) -> list:
    """获取文本的向量嵌入"""
    return embeddings.embed_query(text)


def main():
    # 1. 连接 Milvus
    print("连接到 Milvus...")
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print("✓ 已连接\n")

    # 2. 如果集合已存在，先删除（方便重复测试）
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合: {COLLECTION_NAME}")

    # 3. 定义字段（Schema）
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="mood", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=50),
    ]

    # 4. 创建集合
    print(f"\n创建集合: {COLLECTION_NAME}")
    schema = CollectionSchema(fields=fields, description="AI 日记集合")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("✓ 集合已创建")

    # 5. 创建向量索引
    print("\n创建向量索引...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print("✓ 索引已创建")

    # 6. 加载集合到内存
    print("\n加载集合...")
    collection.load()
    print("✓ 集合已加载")

    # 7. 准备日记数据
    diary_contents = [
        {
            "id": "diary_001",
            "content": "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
            "date": "2026-01-10",
            "mood": "happy",
            "tags": ["生活", "散步"],
        },
        {
            "id": "diary_002",
            "content": "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
            "date": "2026-01-11",
            "mood": "excited",
            "tags": ["工作", "成就"],
        },
        {
            "id": "diary_003",
            "content": "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
            "date": "2026-01-12",
            "mood": "relaxed",
            "tags": ["户外", "朋友"],
        },
        {
            "id": "diary_004",
            "content": "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
            "date": "2026-01-12",
            "mood": "curious",
            "tags": ["学习", "技术"],
        },
        {
            "id": "diary_005",
            "content": "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
            "date": "2026-01-13",
            "mood": "proud",
            "tags": ["美食", "家庭"],
        },
    ]

    # 8. 生成向量并插入数据
    print("\n生成向量并插入数据...")
    for diary in diary_contents:
        vector = get_embedding(diary["content"])
        data = [
            [diary["id"]],
            [vector],
            [diary["content"]],
            [diary["date"]],
            [diary["mood"]],
            [diary["tags"]],
        ]
        collection.insert(data)
        print(f"  ✓ 插入: {diary['id']}")

    # 9. 刷新集合，确保数据可查询
    collection.flush()
    print(f"\n✓ 共插入 {collection.num_entities} 条记录")

    # 10. 断开连接
    connections.disconnect("default")
    print("\n完成！")


if __name__ == "__main__":
    main()
```

运行：

```bash
python src/insert_data.py
```

### Schema 详解

这就是 Schema，创建 Collection 集合的时候需要指定：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | VARCHAR(50) | 主键，日记 ID |
| vector | FLOAT_VECTOR(1024) | 向量字段，1024 维 |
| content | VARCHAR(5000) | 日记内容 |
| date | VARCHAR(50) | 日期 |
| mood | VARCHAR(50) | 心情 |
| tags | ARRAY(VARCHAR) | 标签数组 |

其实和 MySQL 的表差不多，唯一的区别是 vector 这个字段，我们设置了 FloatVector 类型，也就是向量，指定维度是 1024 维。这样我们后面插入数据，也要把嵌入模型指定为 1024 的维度。

### 向量索引

向量字段需要建立索引：

| 参数 | 值 | 说明 |
|------|-----|------|
| index_type | IVF_FLAT | 索引类型，倒排文件扁平索引 |
| metric_type | COSINE | 距离度量，余弦相似度 |
| params.nlist | 1024 | 聚类中心数量 |

metric_type 指定用余弦相似度作为距离度量。余弦相似度的原理前面讲过，值越接近 1 表示越相似。

---

### 第二步：向量搜索查询

创建 `src/query_data.py`：

```python
"""
Milvus 向量搜索示例：根据自然语言查询相似日记
"""
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=VECTOR_DIM,
)


def get_embedding(text: str) -> list:
    """获取文本的向量嵌入"""
    return embeddings.embed_query(text)


def main():
    # 1. 连接 Milvus
    print("连接到 Milvus...")
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print("✓ 已连接\n")

    # 2. 获取集合
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 3. 向量搜索
    query = "我想看看关于户外活动的日记"
    print(f"查询: \"{query}\"\n")

    query_vector = get_embedding(query)

    # 搜索参数
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    # 执行搜索
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=2,
        output_fields=["id", "content", "date", "mood", "tags"],
    )

    # 4. 打印结果
    print(f"找到 {len(results[0])} 条结果:\n")
    for i, hit in enumerate(results[0]):
        print(f"{i + 1}. [相似度: {hit.score:.4f}]")
        print(f"   ID: {hit.entity.get('id')}")
        print(f"   日期: {hit.entity.get('date')}")
        print(f"   心情: {hit.entity.get('mood')}")
        print(f"   标签: {', '.join(hit.entity.get('tags'))}")
        print(f"   内容: {hit.entity.get('content')}\n")

    # 5. 断开连接
    connections.disconnect("default")


if __name__ == "__main__":
    main()
```

运行：

```bash
python src/query_data.py
```

是把 Query 向量化，做余弦相似度的检索。可以看到，检索出了两条户外活动的日记。

改一下 Query，查询做饭、学习的日记，也能搜出来。你用 MySQL 做关键词搜索可以做到么？很明显不能，这就是为啥用向量数据库！

---

### 第三步：完整 RAG 流程

创建 `src/rag_with_milvus.py`：

```python
"""
基于 Milvus 的完整 RAG 流程：AI 日记本
"""
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024

# 初始化大语言模型（温度调高，让 AI 可以发挥创造性回答）
model = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=VECTOR_DIM,
)


def get_embedding(text: str) -> list:
    """获取文本的向量嵌入"""
    return embeddings.embed_query(text)


def retrieve_relevant_diaries(question: str, k: int = 2) -> list:
    """从 Milvus 中检索相关的日记条目"""
    collection = Collection(COLLECTION_NAME)
    collection.load()

    query_vector = get_embedding(question)

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=k,
        output_fields=["id", "content", "date", "mood", "tags"],
    )

    return results[0]


def answer_diary_question(question: str, k: int = 2) -> str:
    """使用 RAG 回答关于日记的问题"""
    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    # 1. 检索相关日记
    print("\n【检索相关日记】")
    retrieved_diaries = retrieve_relevant_diaries(question, k)

    if len(retrieved_diaries) == 0:
        print("未找到相关日记")
        return "抱歉，我没有找到相关的日记内容。"

    # 2. 打印检索到的日记及相似度
    for i, diary in enumerate(retrieved_diaries):
        print(f"\n[日记 {i + 1}] 相似度: {diary.score:.4f}")
        print(f"日期: {diary.entity.get('date')}")
        print(f"心情: {diary.entity.get('mood')}")
        print(f"标签: {', '.join(diary.entity.get('tags'))}")
        print(f"内容: {diary.entity.get('content')}")

    # 3. 构建上下文
    context = "\n\n━━━━━\n\n".join(
        f"""[日记 {i + 1}]
日期: {diary.entity.get('date')}
心情: {diary.entity.get('mood')}
标签: {', '.join(diary.entity.get('tags'))}
内容: {diary.entity.get('content')}"""
        for i, diary in enumerate(retrieved_diaries)
    )

    # 4. 构建 Prompt
    prompt = f"""你是一个温暖贴心的 AI 日记助手。基于用户的日记内容回答问题，用亲切自然的语言。

请根据以下日记内容回答问题：

{context}

用户问题: {question}

回答要求：
1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
2. 可以总结多篇日记的内容，找出共同点或趋势
3. 如果日记中没有相关信息，请温和地告知用户
4. 用第一人称"你"来称呼日记的作者
5. 回答要有同理心，让用户感到被理解和关心

AI 助手的回答:"""

    # 5. 调用 LLM 生成回答
    print("\n【AI 回答】")
    response = model.invoke(prompt)
    print(response.content)
    print("\n")

    return response.content


def main():
    # 连接 Milvus
    print("连接到 Milvus...")
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print("✓ 已连接\n")

    # 运行 RAG 查询
    answer_diary_question("我最近做了什么让我感到快乐的事情？", 2)

    # 断开连接
    connections.disconnect("default")


if __name__ == "__main__":
    main()
```

运行：

```bash
python src/rag_with_milvus.py
```

我们先把 Query 向量化，去 Milvus 里查出相关数据，然后把这些加到 Prompt 里让大模型回答。

可以看到，大模型基于我们的问题，查询了相关的日记，然后做了回答。完全是根据语义检索的！实际的 AI Agent 里就是这样来做 RAG 的。

---

### 第四步：更新数据（Upsert）

创建 `src/update_data.py`：

```python
"""
Milvus 更新数据示例：通过 upsert 更新日记
"""
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    dimensions=VECTOR_DIM,
)


def get_embedding(text: str) -> list:
    return embeddings.embed_query(text)


def main():
    print("连接到 Milvus...")
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print("✓ 已连接\n")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 更新数据（Milvus 通过 upsert 实现更新）
    print("更新日记条目...")
    update_id = "diary_001"
    updated_content = {
        "id": update_id,
        "content": "今天下了一整天的雨，心情很糟糕。工作上遇到了很多困难，感觉压力很大。一个人在家，感觉特别孤独。",
        "date": "2026-01-10",
        "mood": "sad",
        "tags": ["生活", "压力", "孤独"],
    }

    print("生成新的向量...")
    vector = get_embedding(updated_content["content"])

    # upsert 数据
    data = [
        [updated_content["id"]],
        [vector],
        [updated_content["content"]],
        [updated_content["date"]],
        [updated_content["mood"]],
        [updated_content["tags"]],
    ]
    collection.upsert(data)
    collection.flush()

    print(f"✓ 已更新日记: {update_id}")
    print(f"  新内容: {updated_content['content']}")
    print(f"  新心情: {updated_content['mood']}")
    print(f"  新标签: {', '.join(updated_content['tags'])}")

    connections.disconnect("default")
    print("\n完成！")


if __name__ == "__main__":
    main()
```

运行：

```bash
python src/update_data.py
```

因为要向量化，所以也要嵌入模型。调用 upsert 方法，数据里带上 id 即可。这样，更新就完成了。

---

### 第五步：删除数据

创建 `src/delete_data.py`：

```python
"""
Milvus 删除数据示例：单条删除、批量删除、条件删除
"""
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection

load_dotenv()

COLLECTION_NAME = "ai_diary"


def main():
    print("连接到 Milvus...")
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print("✓ 已连接\n")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 1. 删除单条数据
    print("删除单条日记...")
    delete_id = "diary_005"
    result = collection.delete(expr=f'id == "{delete_id}"')
    print(f"✓ 已删除 {result.delete_count} 条记录")
    print(f"  ID: {delete_id}\n")

    # 2. 批量删除
    print("批量删除日记...")
    delete_ids = ["diary_002", "diary_003"]
    ids_str = ", ".join([f'"{id}"' for id in delete_ids])
    batch_result = collection.delete(expr=f'id in [{ids_str}]')
    print(f"✓ 批量删除 {batch_result.delete_count} 条记录")
    print(f"  IDs: {', '.join(delete_ids)}\n")

    # 3. 条件删除
    print("按条件删除...")
    condition_result = collection.delete(expr='mood == "sad"')
    print(f"✓ 删除 {condition_result.delete_count} 条 mood='sad' 的记录\n")

    # 刷新集合
    collection.flush()
    print(f"集合剩余记录数: {collection.num_entities}")

    connections.disconnect("default")
    print("\n完成！")


if __name__ == "__main__":
    main()
```

运行：

```bash
python src/delete_data.py
```

这个不用向量化数据，也就不用嵌入模型。这里用了 expr（表达式），根据条件来删除，或者 `id in ["id1", "id2"]` 这样来批量删除。

可以看到，数据都被正确删除了。这样我们就完成了对 Milvus 数据的增删改查。

**Milvus Python 操作实战参考截图：**

![Milvus实战1](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/10_公众号_Yi昭.png)

![Milvus实战2](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/11_公众号_Yi昭.png)

![Milvus实战3](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/12_公众号_Yi昭.png)

![Milvus实战4](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/13_公众号_Yi昭.png)

![Milvus实战5](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/14_公众号_Yi昭.png)

![Milvus实战6](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/15_公众号_Yi昭.png)

![Milvus实战7](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/16_公众号_Yi昭.png)

![Milvus实战8](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/17_公众号_Yi昭.png)

![Milvus实战9](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/18_公众号_Yi昭.png)

![Milvus实战10](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/19_公众号_Yi昭.png)

![Milvus实战11](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/20_公众号_Yi昭.png)

![Milvus实战12](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/21_公众号_Yi昭.png)

![Milvus实战13](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/22_公众号_Yi昭.png)

![Milvus实战14](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/23_公众号_Yi昭.png)

![Milvus实战15](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/24_公众号_Yi昭.png)

![Milvus实战16](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/25_公众号_Yi昭.png)

![Milvus实战17](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/26_公众号_Yi昭.png)

![Milvus实战18](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/27_公众号_Yi昭.png)

![Milvus实战19](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/28_公众号_Yi昭.png)

![Milvus实战20](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/29_公众号_Yi昭.png)

![Milvus实战21](../IMG/2026-01-13_向量数据库Milvus：做AIAgent开发必备技术/30_公众号_Yi昭.png)

## Milvus 操作总结

| 操作 | 方法 | 是否需要嵌入模型 | 说明 |
|------|------|------------------|------|
| 创建集合 | Collection + create_index | 否 | 定义 Schema，建立向量索引 |
| 插入数据 | collection.insert() | 是 | 数据需要先向量化 |
| 查询数据 | collection.search() | 是 | Query 需要先向量化 |
| 更新数据 | collection.upsert() | 是 | 新数据需要先向量化 |
| 删除数据 | collection.delete() | 否 | 根据条件/ID 删除 |

## 学习要点

1. **MySQL** 只能根据 ID、关键词去检索，涉及到语义检索的，我们都会存到 **Milvus** 里
2. 用 Docker Compose 跑 Milvus 数据库，然后在 Attu（GUI 工具）和 Python 代码里连上，并做增删改查
3. Milvus 分为 **Database、Collection、Entity** 这三级，Collection 要指定数据结构也就是 Schema
4. **vector 向量字段**需要做索引，用来快速检索，常用 IVF_FLAT + COSINE
5. 把 Milvus 接入 RAG 流程，实现了 AI 日记本的功能，可以根据自然语言去做语义检索，查出最相关的日记
6. MySQL 和 Milvus 分别用于不同的场景：一个是做精确查询，可以关联查出很多表的数据；一个是做语义检索，可以用自然语言来查询
7. 实际上一般会做**双写**，同时对两者做增删改查
8. 做 AI Agent 项目，Milvus 向量数据库是必备技术，可以写到简历上，围绕这个聊很多功能的实现，比如知识、记忆等，需要重点掌握

## 扩展方向

- 学习更多索引类型（HNSW、DISKANN、AUTOINDEX）
- 探索标量过滤 + 向量检索的混合查询
- 学习 Milvus 的分区（Partition）功能
- 探索 Zilliz Cloud（Milvus 托管服务）
- 下节会学习更多 Loader 的详细用法

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/11-milvus-vector-db

包含本文的完整可运行代码示例（Milvus 增删改查 + 完整 RAG 流程 + Docker Compose 配置）。

---

**上一篇**：[LangChain 全部 Splitter 详解](./10_LangChain全部Splitter.md) | **下一篇**：[LangChain 全部 Loader 详解](./12_LangChain全部Loader.md)
