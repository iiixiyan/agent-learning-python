# Milvus + RAG 实战：电子书语义检索助手

> **Python 版** | 基于 LangChain + Milvus + FastAPI 技术栈
> 阶段：第一阶段 | 前置知识：Milvus 基础、RAG 原理、LangChain

---

## 项目目标

做一个电子书语义检索助手：上传一本电子书（PDF/TXT），系统自动解析、切分、向量化存入 Milvus，用户可以用自然语言提问，系统检索相关段落并让大模型给出答案。

## 完整流程

```
电子书 → 文档加载 → 文本切分 → 向量化 → 存入 Milvus
                                          ↓
用户提问 → 问题向量化 → Milvus 检索 Top-K → 拼接 Prompt → 大模型生成答案
```

## 环境准备

### 1. 启动 Milvus（Docker）

```bash
# 下载 Milvus Docker Compose 配置文件
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动 Milvus
docker compose up -d
```

Milvus 启动后，服务运行在 `localhost:19530`。

### 2. 安装 Python 依赖

```bash
pip install langchain langchain-openai langchain-community pymilvus pypdf unstructured python-dotenv
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
# OpenAI API 配置（以通义千问为例）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
EMBEDDINGS_MODEL_NAME=text-embedding-v3

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## 第一步：文档加载与切分

创建 `document_processor.py`：

```python
"""
文档加载与切分模块
支持 PDF 和 TXT 格式，使用 RecursiveCharacterTextSplitter 进行语义切分
"""
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    """
    加载文档，支持 PDF 和 TXT 格式

    Args:
        file_path: 文档文件路径

    Returns:
        List[Document]: 文档列表，每个 Document 包含 page_content 和 metadata

    Raises:
        ValueError: 不支持的文件格式
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        # PDF 加载器，自动按页分割
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        # 纯文本加载器
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的格式: {ext}，仅支持 .pdf 和 .txt")

    return loader.load()


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    切分文档为小块

    Args:
        documents: 文档列表
        chunk_size: 每个分块的字符数，默认 500
        chunk_overlap: 分块之间的重叠字符数，默认 50（防止语义断裂）

    Returns:
        List[Document]: 切分后的文档块列表
    """
    # 递归字符分割器：按段落→换行→句子→词→字符逐级递归分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )

    return splitter.split_documents(documents)


# 使用示例
if __name__ == "__main__":
    # 加载文档
    docs = load_document("example_book.pdf")
    print(f"加载了 {len(docs)} 页")

    # 切分文档
    chunks = split_documents(docs)
    print(f"切分为 {len(chunks)} 块")

    # 查看第一个分块
    if chunks:
        print(f"\n第一个分块（前100字符）:")
        print(chunks[0].page_content[:100])
        print(f"元数据: {chunks[0].metadata}")
```

### 切分策略说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| chunk_size | 500 | 每个分块的字符数，太小会丢失上下文，太大会降低检索精度 |
| chunk_overlap | 50 | 分块间的重叠字符数，通常为 chunk_size 的 10%，防止语义断裂 |
| separators | ["\n\n", "\n", "。", "！", "？", ".", " ", ""] | 递归分割符，优先按段落，最后按字符强制分割 |

---

## 第二步：存入 Milvus

创建 `milvus_store.py`：

```python
"""
Milvus 向量存储模块
使用 LangChain 的 Milvus 集成，简化向量数据库操作
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# Milvus 连接配置
connection_args = {
    "host": os.getenv("MILVUS_HOST", "localhost"),
    "port": os.getenv("MILVUS_PORT", "19530"),
}


def create_vector_store(chunks, collection_name="ebook_rag"):
    """
    创建向量库并存入数据

    Args:
        chunks: 切分后的文档块列表
        collection_name: Milvus 集合名称

    Returns:
        Milvus: 向量存储对象
    """
    print(f"正在创建集合 {collection_name} 并向量化 {len(chunks)} 个文档块...")

    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=connection_args,
        drop_old=True,  # 如果集合已存在，先删除再创建
    )

    print("向量库创建完成！")
    return vector_store


def get_vector_store(collection_name="ebook_rag"):
    """
    获取已存在的向量库（不需要重新导入数据）

    Args:
        collection_name: Milvus 集合名称

    Returns:
        Milvus: 向量存储对象
    """
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args=connection_args,
    )


# 使用示例
if __name__ == "__main__":
    from document_processor import load_document, split_documents

    # 加载并切分文档
    docs = load_document("example_book.pdf")
    chunks = split_documents(docs)

    # 创建并存入向量库
    vector_store = create_vector_store(chunks)
    print("向量库创建完成")
```

### Milvus 集合说明

LangChain 的 Milvus 集成会自动创建集合，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| pk | Int64 | 主键，自动生成 |
| vector | FloatVector | 向量字段，维度由嵌入模型决定 |
| text | VarChar | 文档内容 |
| 其他 metadata | 动态字段 | 如 page、source 等 |

---

## 第三步：语义检索

创建 `retriever.py`：

```python
"""
语义检索模块
支持相似度搜索和带分数的相似度搜索
"""


def search_documents(query: str, vector_store, top_k=5):
    """
    检索相关文档

    Args:
        query: 用户查询
        vector_store: 向量存储对象
        top_k: 返回最相似的前 K 个文档

    Returns:
        List[Document]: 检索到的文档列表
    """
    # 方式一：相似度搜索（只返回文档）
    results = vector_store.similarity_search(query, k=top_k)

    # 方式二：带分数的相似度搜索（返回文档和距离分数）
    results_with_score = vector_store.similarity_search_with_score(query, k=top_k)

    print(f"查询: {query}")
    print("=" * 60)

    for i, (doc, score) in enumerate(results_with_score, 1):
        # Milvus 使用 L2 距离，分数越小越相似
        # 转换为相似度：1 - score（近似）
        similarity = 1 - score if score < 1 else max(0, 1 / (1 + score))

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"来源: 第{doc.metadata.get('page', '?')}页")
        print(f"内容: {doc.page_content[:200]}...")

    return results


# 使用示例
if __name__ == "__main__":
    from milvus_store import get_vector_store

    # 获取已存在的向量库
    vector_store = get_vector_store("ebook_rag")

    # 测试检索
    search_documents("什么是 RAG 检索增强生成？", vector_store)
```

### 相似度分数说明

Milvus 默认使用 L2（欧几里得）距离：

| 距离度量 | 说明 | 分数含义 |
|----------|------|----------|
| L2 | 欧几里得距离 | 分数越小越相似 |
| IP | 内积 | 分数越大越相似 |
| COSINE | 余弦相似度 | 分数越大越相似 |

LangChain 的 Milvus 集成默认使用 L2，所以 `similarity_search_with_score` 返回的分数越小表示越相似。

---

## 第四步：RAG 问答

创建 `rag_qa.py`：

```python
"""
RAG 问答模块
使用 LCEL（LangChain Expression Language）构建 RAG Chain
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,  # 温度设为 0，减少幻觉，保证回答准确
)

# RAG Prompt 模板
RAG_PROMPT = ChatPromptTemplate.from_template("""你是一个专业的电子书问答助手。根据以下参考资料回答用户问题。

参考资料：
{context}

用户问题：{question}

回答要求：
1. 只根据参考资料回答，不要编造
2. 引用来源页码，如（第3页）
3. 如果参考资料中没有相关信息，回答"该电子书未涉及此内容"
4. 用中文回答，简洁准确
""")


def format_docs(docs):
    """
    格式化文档为上下文字符串

    Args:
        docs: 文档列表

    Returns:
        str: 格式化后的上下文字符串
    """
    return "\n\n".join([
        f"[第{doc.metadata.get('page', '?')}页] {doc.page_content}"
        for doc in docs
    ])


def build_rag_chain(vector_store):
    """
    构建 RAG Chain

    Args:
        vector_store: 向量存储对象

    Returns:
        Chain: RAG 问答链
    """
    # 创建检索器，返回 Top 5 相关文档
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 使用 LCEL 构建 Chain
    # 1. 检索文档并格式化为上下文
    # 2. 填充 Prompt 模板
    # 3. 调用大模型
    # 4. 解析输出为字符串
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# 使用示例
if __name__ == "__main__":
    from milvus_store import get_vector_store

    # 获取向量库
    vector_store = get_vector_store("ebook_rag")

    # 构建 RAG Chain
    rag_chain = build_rag_chain(vector_store)

    # 提问
    answer = rag_chain.invoke("这本书的主要内容是什么？")
    print(f"回答: {answer}")
```

### RAG Prompt 设计要点

1. **明确角色**：告诉大模型它是一个"专业的电子书问答助手"
2. **提供上下文**：把检索到的文档作为参考资料传入
3. **约束回答**：明确要求"只根据参考资料回答，不要编造"，减少幻觉
4. **引用来源**：要求引用页码，让答案更可信，也方便用户查证
5. **兜底处理**：如果参考资料中没有相关信息，明确回答"未涉及"

---

## 第五步：完整入口

创建 `main.py`：

```python
"""
电子书语义检索助手 - 完整入口
整合文档加载、切分、向量化、检索、问答全流程
"""
from document_processor import load_document, split_documents
from milvus_store import create_vector_store, get_vector_store
from rag_qa import build_rag_chain


class EbookRAG:
    """电子书 RAG 问答类"""

    def __init__(self, collection_name="ebook_rag"):
        """
        初始化

        Args:
            collection_name: Milvus 集合名称
        """
        self.collection_name = collection_name
        self.vector_store = None
        self.chain = None

    def ingest(self, file_path: str):
        """
        导入电子书：加载→切分→向量化→存入 Milvus

        Args:
            file_path: 电子书文件路径
        """
        print(f"加载文档: {file_path}")
        docs = load_document(file_path)
        print(f"加载了 {len(docs)} 页")

        print("切分文档...")
        chunks = split_documents(docs)
        print(f"共切分为 {len(chunks)} 块")

        print("正在向量化并存入 Milvus...")
        self.vector_store = create_vector_store(chunks, self.collection_name)

        self.chain = build_rag_chain(self.vector_store)
        print("导入完成！可以开始提问了。")

    def ask(self, question: str) -> str:
        """
        提问：检索相关文档→构建 Prompt→大模型生成答案

        Args:
            question: 用户问题

        Returns:
            str: 大模型生成的答案
        """
        # 如果还没有初始化 Chain，先获取已存在的向量库
        if self.chain is None:
            self.vector_store = get_vector_store(self.collection_name)
            self.chain = build_rag_chain(self.vector_store)

        return self.chain.invoke(question)


# 使用示例
if __name__ == "__main__":
    # 创建 RAG 实例
    rag = EbookRAG(collection_name="ebook_rag")

    # 首次使用：导入电子书
    # rag.ingest("example_book.pdf")

    # 如果已经导入过，直接提问即可
    print("电子书语义检索助手已启动")
    print("输入问题开始提问，输入 q 退出\n")

    while True:
        q = input("请提问: ")
        if q.lower() == "q":
            print("再见！")
            break

        print("\n正在检索并生成答案...")
        answer = rag.ask(q)
        print(f"\n回答: {answer}\n")
        print("-" * 60)
```

### 项目文件结构

```
ebook_rag/
├── .env                    # 环境变量配置
├── docker-compose.yml      # Milvus Docker 配置
├── main.py                 # 完整入口
├── document_processor.py   # 文档加载与切分
├── milvus_store.py         # Milvus 向量存储
├── retriever.py            # 语义检索
├── rag_qa.py               # RAG 问答
└── example_book.pdf        # 示例电子书（自行准备）
```

---

## 学习要点

1. **RAG 四步走**：加载→切分→向量化→检索生成，每一步都影响最终效果
2. **切分策略很重要**：chunk_size 500、overlap 50 是常用配置，按语义分隔符切分效果更好
3. **Milvus 距离度量**：默认用 L2 距离，分数越小越相似，展示时可以转换为相似度
4. **RAG Prompt 设计**：要明确要求"只根据参考资料回答"，减少幻觉；引用来源页码让答案更可信
5. **LCEL 构建 Chain**：使用 LangChain Expression Language 可以优雅地构建 RAG 流水线
6. **生产环境优化**：要加缓存（Redis）和检索结果重排序（Reranker）提升效果

## 扩展方向

- 支持更多文档格式（Word、Excel、PPT、HTML）
- 添加文档解析预览功能
- 实现多轮对话（带历史记录的 RAG）
- 添加检索结果高亮展示
- 集成 Reranker 模型提升检索精度
- 实现文档增量更新（不用每次全量重建）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/12-milvus-rag-practice

包含本文的完整可运行代码示例（电子书语义检索助手全流程）。

---

**上一篇**：[向量数据库 Milvus](./11_向量数据库Milvus.md) | **下一篇**：[Memory 管理三大策略](./13_Memory管理的三大策略.md)
