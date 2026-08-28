# Runnable：把写逻辑变成组装 Chain

> **Python 版** | 基于 LangChain Python + LCEL 技术栈
> 阶段：第一阶段 | 核心概念：LCEL（LangChain Expression Language）

---

## 为什么需要 Runnable？

### 传统写法：一步步调用

新手常见的写法是一步步调用，代码冗长且难以复用：

```python
"""
传统写法：一步步调用
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 创建 Prompt
prompt = ChatPromptTemplate.from_template("翻译：{text}")

# 2. 创建大模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. 格式化 Prompt
messages = prompt.format_messages(text="hello")

# 4. 调用大模型
result = llm.invoke(messages)

# 5. 输出结果
print(result.content)
```

### 问题

| 问题 | 说明 |
|------|------|
| **代码冗长** | 每一步都要手动调用，代码量大 |
| **难以复用** | 逻辑散落在各处，无法整体复用 |
| **难以测试** | 无法单独测试某个环节 |
| **难以扩展** | 添加新环节需要修改多处代码 |
| **不支持流式/批量** | 需要手动实现流式和批量逻辑 |

### LCEL 写法：用管道符组装

LangChain 提供了 **LCEL（LangChain Expression Language）**，用管道符 `|` 把各个组件组装成 Chain，简洁优雅：

```python
"""
LCEL 写法：用管道符组装 Chain
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 创建组件
prompt = ChatPromptTemplate.from_template("翻译：{text}")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 用管道符组装 Chain
chain = prompt | llm

# 调用 Chain
result = chain.invoke({"text": "hello"})
print(result.content)
```

**对比**：

| 传统写法 | LCEL 写法 |
|----------|-----------|
| 5 步手动调用 | 1 行组装 + 1 行调用 |
| 代码冗长 | 简洁优雅 |
| 难以复用 | Chain 可整体复用 |
| 不支持流式/批量 | 原生支持 invoke/stream/batch |

---

## Runnable 核心接口

所有 Runnable 组件（Prompt、LLM、Parser、自定义函数等）都实现了三个核心方法：

| 方法 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `invoke()` | 同步调用，输入一个输出一个 | 单个输入 | 单个输出 |
| `stream()` | 流式输出，逐块返回 | 单个输入 | 异步迭代器 |
| `batch()` | 批量调用，输入列表输出列表 | 输入列表 | 输出列表 |

```python
"""
Runnable 三种调用方式
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("翻译：{text}")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm

# 1. 同步调用：输入一个，输出一个
print("=== 同步调用 ===")
result = chain.invoke({"text": "hello"})
print(result.content)

# 2. 流式输出：逐块返回，适合打字机效果
print("\n=== 流式输出 ===")
for chunk in chain.stream({"text": "写一首短诗"}):
    print(chunk.content, end="", flush=True)
print()

# 3. 批量调用：输入列表，输出列表
print("\n=== 批量调用 ===")
results = chain.batch([
    {"text": "hello"},
    {"text": "world"},
    {"text": "python"},
])
for i, result in enumerate(results):
    print(f"{i + 1}. {result.content}")
```

### 三种方式的适用场景

| 方式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| `invoke()` | 简单查询、同步处理 | 简单直接 | 等待时间长 |
| `stream()` | 对话、长文本生成 | 用户体验好 | 代码稍复杂 |
| `batch()` | 批量处理、数据导入 | 效率高 | 内存占用大 |

---

## 常用 Runnable 组件

### 1. PromptTemplate / ChatPromptTemplate

Prompt 模板，负责把用户输入格式化成大模型需要的消息格式。

```python
"""
PromptTemplate / ChatPromptTemplate
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ChatPromptTemplate：对话模板，支持多角色
chat_prompt = ChatPromptTemplate.from_template("用中文解释：{topic}")

# PromptTemplate：纯文本模板
text_prompt = PromptTemplate.from_template("翻译：{text}")
```

### 2. ChatModel（大模型）

大语言模型，负责生成回答。

```python
"""
ChatModel：大语言模型
"""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### 3. OutputParser（输出解析）

输出解析器，负责把大模型的输出解析成需要的格式（字符串、JSON、Pydantic 对象等）。

```python
"""
OutputParser：输出解析
"""
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# StrOutputParser：把 Message 对象转成纯字符串
parser = StrOutputParser()

# 完整 chain：prompt → llm → parser
chain = prompt | llm | parser

# 调用后直接得到字符串，不是 Message 对象
result = chain.invoke({"topic": "递归"})
print(result)  # 直接是字符串
```

### 4. RunnableLambda（自定义函数）

把任意 Python 函数包装成 Runnable，可以插入到 Chain 中。

```python
"""
RunnableLambda：自定义函数
"""
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 自定义函数：给结果加前缀
def add_prefix(text):
    return f"【翻译结果】{text}"

# 组装 Chain：prompt → llm → 字符串解析 → 自定义函数
chain = prompt | llm | StrOutputParser() | RunnableLambda(add_prefix)

# 调用
result = chain.invoke({"text": "hello"})
print(result)
# 输出: 【翻译结果】你好
```

### 5. RunnableParallel（并行执行）

同时执行多个 Chain，返回字典结果。适合需要同时做多个任务的场景（如翻译+摘要）。

```python
"""
RunnableParallel：并行执行多个 Chain
"""
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 同时做翻译和摘要
chain = RunnableParallel({
    "translation": ChatPromptTemplate.from_template("翻译成中文：{text}") | llm | StrOutputParser(),
    "summary": ChatPromptTemplate.from_template("用一句话总结：{text}") | llm | StrOutputParser(),
})

# 调用
result = chain.invoke({"text": "Python is a programming language."})
print("翻译:", result["translation"])
print("摘要:", result["summary"])
```

### 6. RunnablePassthrough（透传输入）

把输入原样透传，常用于 RAG 场景中把用户问题同时传给检索器和 Prompt。

```python
"""
RunnablePassthrough：透传输入
"""
from langchain_core.runnables import RunnablePassthrough

# 透传输入，不做任何处理
passthrough = RunnablePassthrough()
result = passthrough.invoke({"question": "你好"})
print(result)  # 输出: {'question': '你好'}
```

---

## 实战：RAG Chain

RAG（检索增强生成）是 LCEL 最经典的应用场景，展示了如何用 Runnable 组装复杂的处理流程。

```python
"""
RAG Chain 实战：用 LCEL 组装检索增强生成
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. 准备向量库（模拟知识库）
print("正在创建向量库...")
vectorstore = FAISS.from_texts(
    [
        "Python由Guido van Rossum发明于1991年",
        "Python是一种解释型、面向对象的编程语言",
        "Python以简洁易读的语法著称",
        "Python广泛应用于Web开发、数据分析、人工智能等领域",
    ],
    embedding=OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("向量库创建完成\n")

# 2. RAG Prompt
rag_prompt = ChatPromptTemplate.from_template("""根据以下资料回答问题。
要求：只根据资料回答，不要编造。

资料：{context}

问题：{question}

回答：""")

# 3. 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 4. 格式化文档的函数
def format_docs(docs):
    """把检索到的文档列表格式化成字符串"""
    return "\n".join([f"- {doc.page_content}" for doc in docs])

# 5. 组装 RAG Chain
# 数据流：
#   用户问题 → 同时传给两个分支
#     分支1：retriever 检索 → format_docs 格式化 → context
#     分支2：RunnablePassthrough 透传 → question
#   → 合并成字典 {context, question}
#   → rag_prompt 填充 Prompt
#   → llm 生成回答
#   → StrOutputParser 解析成字符串
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 6. 使用
print("=== RAG 查询 ===")
questions = [
    "Python是谁发明的？",
    "Python有什么特点？",
    "Python用于哪些领域？",
]

for q in questions:
    print(f"\n问题: {q}")
    answer = rag_chain.invoke(q)
    print(f"回答: {answer}")
```

### RAG Chain 数据流图

```
用户问题: "Python是谁发明的？"
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
retriever（检索）              RunnablePassthrough（透传）
    │                                  │
    ▼                                  │
format_docs（格式化）                  │
    │                                  │
    ▼                                  ▼
context: "- Python由Guido...    question: "Python是谁发明的？"
           - Python是解释型..."
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
          rag_prompt（填充 Prompt）
                   │
                   ▼
          llm（大模型生成回答）
                   │
                   ▼
          StrOutputParser（解析成字符串）
                   │
                   ▼
          最终回答: "Python由Guido van Rossum发明于1991年"
```

---

## 学习要点

1. **`|` 管道符**是 LCEL 的核心，把各个组件串成 Chain，数据流从左到右
2. **常用组合**：`prompt | llm | output_parser`，这是最基础的 Chain 模式
3. **`RunnableParallel`** 可以并行执行多个 Chain，返回字典结果，适合多任务场景
4. **`RunnablePassthrough`** 透传输入，常用于 RAG 中把用户问题同时传给检索器和 Prompt
5. **`RunnableLambda`** 可以把任意 Python 函数包装成 Runnable，插入到 Chain 中
6. **所有 Runnable** 都支持 `invoke`/`stream`/`batch` 三种调用方式，无需额外代码
7. **LCEL 的优势**：代码简洁、易于复用、易于测试、原生支持流式和批量
8. **RAG 是 LCEL 最经典的应用**，展示了如何用 Runnable 组装复杂的处理流程

## 扩展方向

- 学习更多 Runnable 组件（RunnableBranch、RunnableWithFallbacks 等）
- 探索 LCEL 的高级用法（动态路由、错误处理、回调函数）
- 学习 LangGraph，用图的方式组装更复杂的 Agent 流程
- 结合 LangSmith 调试和监控 Chain 的执行
- 探索 LCEL 的异步调用（ainvoke、astream、abatch）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/17-runnable-lcel

包含本文的完整可运行代码示例（Runnable 三种调用方式 + 6种常用组件 + RAG Chain 实战）。

---

**上一篇**：[Prompt Template](./16_Prompt-Template.md) | **下一篇**：[实战练习 LCEL 组装 Chain](./18_实战练习LCEL组装chain.md)
