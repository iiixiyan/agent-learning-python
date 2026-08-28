# RAG：把文档向量化，基于向量实现真正的语义搜索

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

## 为什么需要 RAG？

大模型所知道的知识，取决于在训练的时候给它的数据集。

如果你问它最近发生的事情，或者你企业内部私有文档的一些事情，它是不知道的。

但它很可能不会说自己不知道，而是会胡乱回答，也就是所谓的**幻觉**（以为自己知道）。

如何解决大模型的幻觉呢？

其实也很容易想到：用户要查询的内容，我们先去内部知识库里查一下，把它放到 Prompt 里再给大模型。

这样大模型通过这些文档知道了背景知识，就可以回答相应的问题了。

这就是 RAG：

- **R**etrieval 检索
- **A**ugmented 增强
- **G**eneration 生成

去知识库里**检索**用户问的知识的相关文档片段，作为背景知识加到 Prompt 里**增强**它，让大模型根据这些来**生成**回答。

```
用户提问 → 检索知识库 → 相关文档片段 → 增强 Prompt → 大模型生成回答
```

这个是很容易想到的思路，也是很贴切的名字。

## 语义搜索需要向量

但有个问题：用户问了一个问题，你怎么把相关的文档片段查出来呢？

比如用户查水果的信息，你要把苹果、香蕉、草莓的相关文档查出来。

想想怎么做？关键词搜索可以么？很明显不行。

这种语义搜索就需要**向量（Vector）**了。

### 二维向量示例

比如如果按照两个维度存储信息，分为可食用性、硬度：

- 维度 1：食用性（0 = 无，1 = 高）
- 维度 2：硬度（0 = 软/液体，1 = 硬）

那这几个概念大概是这样的向量：

| 概念 | 食用性 | 硬度 | 向量 |
|------|--------|------|------|
| 水果 | 0.9 | 0.3 | [0.9, 0.3] |
| 苹果 | 0.9 | 0.5 | [0.9, 0.5] |
| 香蕉 | 0.9 | 0.1 | [0.9, 0.1] |
| 石头 | 0.1 | 0.9 | [0.1, 0.9] |

明显可以看出来，苹果、水果、香蕉，这三个概念相关性很大，而水果和石头相关性就不大。

### 余弦相似度

计算的话，可以通过夹角判断相似度，夹角越小相似度越高：

也就是**余弦相似度**（两个向量夹角的余弦值）。

```
相似度 = cos(θ) = (A · B) / (|A| × |B|)
```

当然，具体的向量数据肯定不会只有二维，可能会是几百维。虽然高纬度没法可视化，但是原理是一样的。

我们都是通过两个概念对应的向量的余弦相似度来判断相关性。

也就是说**通过向量计算实现语义检索！**

这就是为啥 RAG 一般都结合向量化来做，虽然基于关键词来做也是 RAG，但是那种没法语义搜索，意义不大。

## 嵌入模型（Embedding Model）

有的同学可能会问，那给你一个概念，怎么计算它的向量值呢？

这个需要用到专门的模型，叫**嵌入模型（Embedding Model）**。

它和大语言模型（LLM）是不一样的，它的功能就只有把知识转成向量。

这个知识可以是文本、图片、语音等，向量化之后，就都可以实现语义搜索了！

| 对比项 | 大语言模型 (LLM) | 嵌入模型 (Embedding) |
|--------|-------------------|----------------------|
| 功能 | 生成文本、对话、推理 | 把内容转成向量 |
| 输入 | 文本 Prompt | 文本/图片/语音 |
| 输出 | 生成的文本 | 多维向量 |
| 价格 | 较贵 | 便宜很多 |

我们写代码会用专门的嵌入模型，收费比大模型便宜很多很多。

## RAG 完整流程

加上向量化之后的 RAG 流程是什么样的呢？

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户 Prompt │────→│  嵌入模型     │────→│  向量数据库   │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                   │ 相似度检索
                                                   ↓
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  大模型生成  │←────│ 增强后的 Prompt│←────│ 相关文档片段  │
└─────────────┘     └──────────────┘     └──────────────┘
```

用户的 Prompt 会通过嵌入模型转成向量，然后 Retriever 基于这个向量去向量数据库中检索，找到相似的向量，把对应的文档块返回，加到 Prompt 里作为背景知识，给大模型。

**存的不是向量么？怎么记录向量关联的文档？**

文档在向量化的时候，会在向量的元信息里记录来源文档。

综上，我们可以**在原始 Prompt 给到大模型之前，查询下知识库，把相关的文档作为背景知识加入到 Prompt 里，再让大模型回答，这就是 RAG。**

RAG 要实现语义查询，需要基于向量来做，把文档向量化存储到向量数据库，查询的时候也把 Prompt 向量化，去数据库中做相似度检索，这样就可以找到语义相近的文档块。

## 实战：用 Python 实现 RAG

知道了什么是 RAG，我们来写代码试一下。

### 项目初始化

```bash
mkdir rag-test
cd rag-test
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### 环境配置

创建 `.env` 文件：

```env
# OpenAI 兼容 API 配置（以通义千问为例）
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
MODEL_NAME=qwen-plus
EMBEDDINGS_MODEL_NAME=text-embedding-v3
```

### 完整代码

创建 `src/hello_rag.py`：

```python
"""
RAG 入门示例：基于内存向量数据库的语义检索
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore

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

# 3. 准备文档（模拟知识库）
documents = [
    Document(
        page_content="""小明是一个活泼开朗的小男孩，他有一双明亮的大眼睛，总是带着灿烂的笑容。
小明最喜欢的事情就是和朋友们一起玩耍，他特别擅长踢足球，每次在球场上奔跑时，就像一道阳光一样充满活力。""",
        metadata={"chapter": 1, "character": "小明", "type": "角色介绍", "mood": "活泼"}
    ),
    Document(
        page_content="""小华是小明最好的朋友，他是一个安静而聪明的男孩。
小华喜欢读书和画画，他的画总是充满了想象力。
虽然性格不同，但小华和小明从幼儿园就认识了，他们一起度过了无数个快乐的时光。""",
        metadata={"chapter": 2, "character": "小华", "type": "角色介绍", "mood": "温馨"}
    ),
    Document(
        page_content="""有一天，学校要举办一场足球比赛，小明非常兴奋，他邀请小华一起参加。
但是小华从来没有踢过足球，他担心自己会拖累小明。
小明看出了小华的担忧，他拍着小华的肩膀说："没关系，我们一起练习，我相信你一定能行的！" """,
        metadata={"chapter": 3, "character": "小明和小华", "type": "友情情节", "mood": "鼓励"}
    ),
    Document(
        page_content="""接下来的日子里，小明每天放学后都会教小华踢足球。
小明耐心地教小华如何控球、传球和射门，而小华虽然一开始总是踢不好，但他从不放弃。
小华也用自己的方式回报小明，他画了一幅画送给小明，画上是两个小男孩在球场上一起踢球的场景。""",
        metadata={"chapter": 4, "character": "小明和小华", "type": "友情情节", "mood": "互助"}
    ),
    Document(
        page_content="""比赛那天终于到了，小明和小华一起站在球场上。
虽然小华的技术还不够熟练，但他非常努力，而且他用自己的观察力帮助小明找到了对手的弱点。
在关键时刻，小华传出了一个漂亮的球，小明接球后射门得分！
他们赢得了比赛，更重要的是，他们的友谊变得更加深厚了。""",
        metadata={"chapter": 5, "character": "小明和小华", "type": "高潮转折", "mood": "激动"}
    ),
    Document(
        page_content="""从那以后，小明和小华成为了学校里最要好的朋友。
小明教小华运动，小华教小明画画，他们互相学习，共同成长。
每当有人问起他们的友谊，他们总是笑着说："真正的朋友就是互相帮助，一起变得更好的人！" """,
        metadata={"chapter": 6, "character": "小明和小华", "type": "结局", "mood": "欢乐"}
    ),
    Document(
        page_content="""多年后，小明成为了一名职业足球运动员，而小华成为了一名优秀的插画师。
虽然他们走上了不同的道路，但他们的友谊从未改变。
小华为小明设计了球衣上的图案，小明在每场比赛后都会给小华打电话分享喜悦。
他们证明了，真正的友情可以跨越时间和距离，永远闪闪发光。""",
        metadata={"chapter": 7, "character": "小明和小华", "type": "尾声", "mood": "温馨"}
    ),
]

# 4. 创建向量数据库（内存版）
vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
)

# 5. 创建检索器，返回相似度最高的 3 个文档
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. 运行 RAG 查询
def run_rag_query(question: str):
    """运行 RAG 查询"""
    print("=" * 80)
    print(f"问题: {question}")
    print("=" * 80)

    # 使用检索器获取文档
    retrieved_docs = retriever.invoke(question)

    # 使用 similarity_search_with_score 获取相似度评分
    scored_results = vector_store.similarity_search_with_score(question, k=3)

    # 打印检索到的文档和相似度评分
    print("\n【检索到的文档及相似度评分】")
    for i, doc in enumerate(retrieved_docs):
        # 找到对应的评分
        scored_result = next(
            ([sd, score] for sd, score in scored_results if sd.page_content == doc.page_content),
            None
        )
        score = scored_result[1] if scored_result else None
        similarity = f"{1 - score:.4f}" if score is not None else "N/A"

        print(f"\n[文档 {i + 1}] 相似度: {similarity}")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"元数据: 章节={doc.metadata.get('chapter')}, 角色={doc.metadata.get('character')}, 类型={doc.metadata.get('type')}, 心情={doc.metadata.get('mood')}")

    # 构建增强后的 Prompt
    context = "\n\n━━━━━\n\n".join(
        f"[片段 {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"""你是一个讲友情故事的老师。基于以下故事片段回答问题，用温暖生动的语言。
如果故事中没有提到，就说"这个故事里还没有提到这个细节"。

故事片段:
{context}

问题: {question}

老师的回答:"""

    # 调用大模型生成回答
    print("\n【AI 回答】")
    response = model.invoke(prompt)
    print(response.content)
    print("\n")


# 7. 测试
if __name__ == "__main__":
    questions = [
        "小华和小明是怎么成为朋友的？",
        "足球比赛的结果是什么？",
        "多年后他们各自做了什么工作？",
    ]

    for q in questions:
        run_rag_query(q)
```

### 运行测试

```bash
python src/hello_rag.py
```

可以看到，根据你的问题，查询到了 3 个文档，然后大模型基于这些做了回答。

这样我们就跑通了 RAG 的流程！

## 核心 API 解析

| API | 作用 |
|-----|------|
| `InMemoryVectorStore.from_documents()` | 基于嵌入模型把文档向量化存入数据库 |
| `as_retriever(search_kwargs={"k": 3})` | 创建检索器，指定查询相似度最大的几个文档 |
| `similarity_search_with_score()` | 相似度搜索并返回评分 |
| `retriever.invoke(question)` | 查询文档 |

只要你理解了 RAG 的流程，这些 API 自然也就会用了。

## 学习要点

1. **RAG = 检索 + 增强 + 生成**，解决大模型幻觉和知识过时问题
2. **语义搜索需要向量**，关键词搜索做不到语义匹配
3. **余弦相似度**判断两个向量的相关性，夹角越小越相似
4. **嵌入模型**专门负责把内容转成向量，和大语言模型不同
5. **文档元数据**记录来源信息，方便追溯和过滤
6. **内存向量数据库**适合学习和测试，生产环境用 Milvus/Chroma 等

## 扩展方向

- 使用专业向量数据库（Milvus、Chroma、Pinecone）
- 文档分块策略（按段落、按字符数、语义分块）
- 混合检索（向量检索 + 全文检索 + 重排）
- 多模态 RAG（图片、语音、视频向量化）
- 企业级知识库项目（后续章节会详细讲解）

想一下，如果你要做公司内部文档的智能助手，是不是就可以用 RAG 来实现呢？

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/08-rag-intro

包含本文的完整可运行代码示例（RAG 入门 + 内存向量数据库）。

---

**上一篇**：[高德 MCP + 浏览器 MCP：LangChain 复用别人的 MCP Server](./07_高德MCP+浏览器MCP.md) | **下一篇**：[知识库的 Loader 和 Splitter：文档解析与分块](./09_知识库的loader和splitter.md)
