# 知识库的 Loader 和 Splitter：从各种来源加载文档并分割成小块

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

## 为什么需要 Loader 和 Splitter？

上节我们学了 RAG，它可以解决大模型的幻觉问题。

幻觉就是大模型对于它不知道的知识，会以为自己知道，然后胡乱回答。

解决方案 RAG 就是根据用户的 Prompt，去知识库查询相关文档，加到 Prompt 里给到大模型作为背景知识来回答。

这种相关文档的检索，要根据 Prompt 的语义来搜，所以一般要结合向量来实现：

基于嵌入模型把文档向量化，存入向量数据库，查询的时候把 Prompt 向量化，根据余弦相似度，来检索最相近的向量，然后把相关文档放到 Prompt 里。

```
用户 Prompt → 嵌入模型向量化 → 向量数据库相似度检索 → 相关文档 → 增强 Prompt → 大模型回答
```

上节我们跑通了这个流程，会查询出几个相似度最高的文档放到 Prompt 里，大模型基于这些来回答。

但上节我们是直接创建的 Document 对象，然后用嵌入模型存入了向量数据库。

实际上知识的来源可能有很多：一个 Word 文档、一个 PDF 文件、一个 YouTube 视频、一个 URL、一个 X 的推文等。

这种显然就不是直接创建 Document 对象了，而是要用各种 **Loader** 来转换。

## Loader：从各种来源加载文档

经过对应的 Loader 处理后，变成 Document，之后再由嵌入模型向量化后存入知识库。

```
各种知识来源 → Loader → Document 对象 → 嵌入模型向量化 → 向量数据库
```

知识有各种来源，所以对应的各种 Loader 也很多。

现在 LangChain 文档里有 180+ Loader：
https://python.langchain.com/docs/integrations/document_loaders/

常见的 Loader 包括：

| Loader 类型 | 用途 |
|-------------|------|
| TextLoader | 加载纯文本文件 |
| PyPDFLoader | 加载 PDF 文件 |
| Docx2txtLoader | 加载 Word 文档 |
| WebBaseLoader | 加载网页内容 |
| YoutubeLoader | 加载 YouTube 视频字幕 |
| UnstructuredLoader | 加载各种格式文件 |
| CSVLoader | 加载 CSV 文件 |
| SitemapLoader | 加载网站站点地图 |

你可以把各种知识来源通过 Loader 转化为文档存入知识库。

## Splitter：把大文档分割成小块

当然，有的文档可能会很大，比如一个 PDF 文件可能是一本书的大小。

这种很明显不能直接把转化后的 Document 向量化，需要先拆分文档。也就是需要 **Splitter**。

```
大文档 → TextSplitter → 多个小文档（Chunk）→ 嵌入模型向量化 → 向量数据库
```

大的文档经过 TextSplitter 分割后，变成一个个小文档，再给到嵌入模型做向量化。

### 分割策略

分割最简单的就是按照字符，比如换行符 `\n`。

但并不是每一行一个 Document，而是要设置一个 **chunk size**，按照换行符分割好的内容加入到这个 Chunk，当达到 chunk size 后，再继续生成下个 Chunk。

### 重叠（Overlap）

为了避免分割点处的语义断裂，通常会设置一个 **chunk overlap**，让相邻的 Chunk 之间有部分重叠。

```
Chunk 1: [内容A 内容B 内容C]
Chunk 2:         [内容C 内容D 内容E]
                    ↑ 重叠部分
```

这个 Chunk 也是 Document 对象，只是文档内容是分割好的一个个大小合适的块。

### 常见的 Splitter

| Splitter 类型 | 用途 |
|---------------|------|
| RecursiveCharacterTextSplitter | 递归字符分割（最常用） |
| CharacterTextSplitter | 按指定字符分割 |
| TokenTextSplitter | 按 Token 数分割 |
| MarkdownHeaderTextSplitter | 按 Markdown 标题分割 |
| PythonCodeTextSplitter | Python 代码分割 |
| LatexTextSplitter | LaTeX 文档分割 |

## 实战：用 Python 实现 Loader + Splitter + RAG

在上节的 rag-test 项目里继续写。

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters python-dotenv beautifulsoup4
```

### 第一步：用 Loader 加载网页

创建 `src/loader_demo.py`：

```python
"""
Loader 示例：从网页加载文档
"""
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

# 使用 WebBaseLoader 加载网页
loader = WebBaseLoader(
    web_paths=("https://juejin.cn/post/7233327509919547452",),
)

# 加载文档
documents = loader.load()

print(f"加载到 {len(documents)} 个文档")
print(f"文档内容长度: {len(documents[0].page_content)} 字符")
print(f"文档元数据: {documents[0].metadata}")
print(f"\n文档前 200 字符:\n{documents[0].page_content[:200]}")
```

运行：

```bash
python src/loader_demo.py
```

可以看到，网页内容被取出来了，放入了 Document 对象。

### 第二步：用 Splitter 分割文档

创建 `src/splitter_demo.py`：

```python
"""
Splitter 示例：分割大文档
"""
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. 加载网页
loader = WebBaseLoader(
    web_paths=("https://juejin.cn/post/7233327509919547452",),
)
documents = loader.load()
print(f"原始文档长度: {len(documents[0].page_content)} 字符")

# 2. 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个分块的字符数
    chunk_overlap=50,       # 分块之间的重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 分割符，优先使用段落分隔
)

# 3. 分割文档
split_documents = text_splitter.split_documents(documents)

print(f"\n文档分割完成，共 {len(split_documents)} 个分块\n")

# 4. 打印每个分块的信息
for i, doc in enumerate(split_documents):
    print(f"[分块 {i + 1}] 长度: {len(doc.page_content)} 字符")
    print(f"内容: {doc.page_content[:80]}...")
    print()
```

运行：

```bash
python src/splitter_demo.py
```

可以看到，文档被分成了多个小的文档。每个文档都是 500 字符左右，前后重复了 50 个字符。

这样分割好的文档用来做 RAG 性能显然会更好，不需要加载整个大文档。

### 第三步：完整的 RAG 流程

创建 `src/loader_splitter_rag.py`：

```python
"""
完整 RAG 流程：Loader + Splitter + 向量数据库 + 检索 + 大模型回答
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. 初始化大语言模型
model = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 2. 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v3"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 3. 用 Loader 加载网页
print("正在加载网页...")
loader = WebBaseLoader(
    web_paths=("https://juejin.cn/post/7233327509919547452",),
)
documents = loader.load()
print(f"加载完成，原始文档长度: {len(documents[0].page_content)} 字符")

# 4. 用 Splitter 分割文档
print("\n正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
split_documents = text_splitter.split_documents(documents)
print(f"分割完成，共 {len(split_documents)} 个分块")

# 5. 创建向量数据库
print("\n正在创建向量存储...")
vector_store = InMemoryVectorStore.from_documents(
    documents=split_documents,
    embedding=embeddings,
)
print("向量存储创建完成")

# 6. 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 7. 运行 RAG 查询
def run_rag_query(question: str):
    """运行 RAG 查询"""
    print("\n" + "=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    # 检索相关文档
    retrieved_docs = retriever.invoke(question)

    # 获取相似度评分
    scored_results = vector_store.similarity_search_with_score(question, k=2)

    # 打印检索结果
    print("\n【检索到的文档及相似度评分】")
    for i, doc in enumerate(retrieved_docs):
        scored_result = next(
            ([sd, score] for sd, score in scored_results if sd.page_content == doc.page_content),
            None
        )
        score = scored_result[1] if scored_result else None
        similarity = f"{1 - score:.4f}" if score is not None else "N/A"

        print(f"\n[文档 {i + 1}] 相似度: {similarity}")
        print(f"内容: {doc.page_content[:100]}...")

    # 构建增强后的 Prompt
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"""你是一个文章辅助阅读助手，根据文章内容来解答。

文章内容:
{context}

问题: {question}

你的回答:"""

    # 调用大模型
    print("\n【AI 回答】")
    response = model.invoke(prompt)
    print(response.content)


# 8. 测试
if __name__ == "__main__":
    questions = [
        "父亲的去世对作者的人生态度产生了怎样的根本性逆转？",
    ]

    for q in questions:
        run_rag_query(q)
```

运行：

```bash
python src/loader_splitter_rag.py
```

可以看到，Loader 加载了文档，用 Splitter 分成了多个分块（Chunk）。回答的时候检索了相似度最高的 2 个文档块，基于这个做了回答。

## 核心概念总结

| 概念 | 作用 | 所在包 |
|------|------|--------|
| Loader | 从各种来源加载内容为 Document | langchain_community.document_loaders |
| Splitter | 把大文档分割成小文档块 | langchain_text_splitters |
| Document | 文档对象，包含内容和元数据 | langchain_core.documents |
| Chunk | 分割后的文档块，也是 Document 对象 | - |

## 学习要点

1. **Loader** 可以从各种地方加载内容作为 Document，比如 Word、PDF、网页、YouTube、X 的推文等
2. **180+ Loader**，社区维护，所以在 `langchain_community` 这个包
3. **Splitter** 把大文档分割成一个个小的文档块，提升 RAG 检索精度
4. **chunk_size** 控制每个分块的大小，**chunk_overlap** 控制分块间的重叠，避免语义断裂
5. **RecursiveCharacterTextSplitter** 是最常用的分割器，按段落、换行、句子逐级递归分割
6. 分割后的文档块（Chunk）也是 Document 对象，可以直接向量化存入数据库

## 扩展方向

- 探索更多 Loader（PDF、Word、CSV、YouTube 等）
- 学习更多 Splitter 策略（按 Token、按 Markdown 标题、语义分割）
- 优化 chunk_size 和 chunk_overlap 参数
- 结合元数据过滤检索结果
- 下节会详细过一遍各种 Loader 和 Splitter

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/09-loader-splitter

包含本文的完整可运行代码示例（Loader + Splitter + 完整 RAG 流程）。

---

**上一篇**：[RAG：把文档向量化，基于向量实现真正的语义搜索](./08_RAG-把文档向量化.md) | **下一篇**：[LangChain 全部 Splitter 详解](./10_LangChain全部Splitter.md)
