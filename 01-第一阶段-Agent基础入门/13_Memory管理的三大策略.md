# Memory 管理的三大策略：截断、总结、检索

> **Python 版** | 基于 LangChain Python 技术栈
> 阶段：第一阶段 | 解决问题：大模型上下文窗口有限，如何管理对话历史

---

## 为什么需要 Memory 管理？

大模型的上下文窗口有限（如 GPT-4o 是 128K token），对话越来越长时会遇到：

| 问题 | 说明 |
|------|------|
| **超出窗口** | 历史消息被截断，模型丢失上下文 |
| **成本增加** | 每次请求都带全部历史，token 消耗大 |
| **响应变慢** | 输入越长，生成越慢 |
| **注意力分散** | 太多无关历史干扰模型判断 |

**三大策略**：截断（Window）、总结（Summary）、检索（Retrieval），实际项目中通常组合使用。

---

## 策略一：截断（Window Buffer Memory）

### 原理

只保留最近 N 轮对话，更早的直接丢弃。最简单粗暴。

```
完整对话: [轮1] [轮2] [轮3] [轮4] [轮5] [轮6]
保留最近3轮:          [轮4] [轮5] [轮6]
```

### 实现

```python
"""
截断型记忆示例：只保留最近 N 轮对话
"""
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 只保留最近 5 轮对话
# k=5 表示保留 5 轮（用户+AI 各算一条，共 10 条消息）
memory = ConversationBufferWindowMemory(k=5)

# 创建对话链
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# 多轮对话
conversation.predict(input="我叫小明")
conversation.predict(input="我喜欢Python")
conversation.predict(input="我在学Agent开发")
conversation.predict(input="我前面说了什么？")
# 模型能记住最近5轮的内容

# 查看内存中的历史
print("当前记忆中的历史:")
print(memory.load_memory_variables({}))
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| k | 保留的对话轮数 | 5-10（短对话），3-5（长对话） |
| return_messages | 是否返回消息对象列表 | True（配合 Chat 模型） |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单，零额外成本 | 早期信息完全丢失 |
| 响应快，token 可控 | 不适合需要长期记忆的场景 |
| 适合短对话、闲聊 | 重要信息可能被挤掉 |

---

## 策略二：总结（Summary Memory）

### 原理

用大模型把历史对话总结成一段摘要，新对话继续追加，超出阈值时重新总结。

```
对话历史 → 大模型总结 → "用户叫小明，喜欢Python，在学Agent开发..."
新对话 → 追加到摘要 → 定期重新总结
```

### 实现

```python
"""
总结型记忆示例：用大模型把历史对话总结成摘要
"""
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 初始化大语言模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 总结型记忆，每次对话后自动更新摘要
# 需要传入 llm，因为要用大模型来生成总结
memory = ConversationSummaryMemory(llm=llm)

# 创建对话链
conversation = ConversationChain(llm=llm, memory=memory)

# 多轮对话
conversation.predict(input="我叫小明，是一名后端开发")
conversation.predict(input="我想转型做Agent开发")
conversation.predict(input="你有什么学习路线建议吗？")

# 查看总结内容
print("当前对话摘要:")
print(memory.buffer)
# 输出类似: "用户介绍自己叫小明，是一名后端开发，想转型做Agent开发，询问学习路线建议..."
```

### 进阶：总结 + 缓冲（SummaryBufferMemory）

这是最实用的组合策略：最近 N 轮保留原文，更早的总结成摘要。

```python
"""
总结+缓冲型记忆：最近 N 轮保留原文，更早的总结
"""
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 总结+缓冲记忆
# - 最近的对话保留原文（精确）
# - 超过 max_token_limit 后，最早的对话被总结成摘要
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,  # 超过 500 token 就开始总结
    return_messages=True,
)

conversation = ConversationChain(llm=llm, memory=memory)

# 多轮对话测试
for i in range(10):
    conversation.predict(input=f"这是第{i+1}轮对话，内容是关于话题{i+1}")

# 查看记忆状态
print("移动摘要（更早对话的总结）:")
print(memory.moving_summary_buffer)
print("\n缓冲中的最近对话（原文）:")
print(memory.chat_memory.messages)
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| llm | 用于生成总结的大模型 | 同主模型或更便宜的模型 |
| max_token_limit | 触发总结的 token 阈值 | 500-2000 |
| return_messages | 是否返回消息对象列表 | True |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 保留长期信息的概要 | 细节会丢失，总结可能有偏差 |
| token 消耗相对稳定 | 每次总结需要额外调用大模型（有成本） |
| 适合中长对话 | 总结质量依赖大模型能力 |

---

## 策略三：检索（Retrieval Memory）

### 原理

把所有历史消息向量化存入向量库，需要时检索与当前问题最相关的历史片段。类似 RAG，但检索的是对话历史。

```
所有对话历史 → 向量化 → 存入向量库
当前问题 → 向量化 → 检索相关历史 → 拼入 Prompt
```

### 实现

```python
"""
检索型记忆示例：把对话历史存入向量库，按需检索
"""
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationChain

# 1. 创建向量库（用 FAISS 轻量级向量库，生产环境用 Milvus）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_texts([""], embeddings)

# 2. 创建检索器，返回 Top 3 相关历史
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. 创建检索型记忆
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 4. 保存一些历史信息（模拟之前的对话）
memory.save_context(
    {"input": "我叫小明，在字节跳动工作"},
    {"output": "你好小明，很高兴认识你！"}
)
memory.save_context(
    {"input": "我主要用Python做后端开发"},
    {"output": "Python是很好的语言，适合Agent开发"}
)
memory.save_context(
    {"input": "我最近在学LangChain和LangGraph"},
    {"output": "这两个是Agent开发的主流框架"}
)

# 5. 创建对话链
llm = ChatOpenAI(model="gpt-4o", temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

# 6. 问一个需要历史信息的问题
response = conversation.predict(input="我是做什么工作的？用什么语言？")
print("AI 回答:", response)
# 模型会检索到相关历史："在字节跳动工作"、"用Python做后端开发"
```

### 工作流程

```
用户提问: "我是做什么工作的？"
    ↓
问题向量化: [0.12, -0.34, 0.56, ...]
    ↓
向量库检索 Top 3:
  1. "我主要用Python做后端开发" (相似度 0.92)
  2. "我叫小明，在字节跳动工作" (相似度 0.85)
  3. "我最近在学LangChain和LangGraph" (相似度 0.71)
    ↓
拼入 Prompt:
  系统: 你是一个助手...
  相关历史:
    - 用户: 我主要用Python做后端开发
    - 用户: 我叫小明，在字节跳动工作
  用户: 我是做什么工作的？
    ↓
大模型生成回答: "你是一名后端开发，主要使用Python语言，在字节跳动工作。"
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 可以保留无限长的历史 | 需要额外维护向量库 |
| 只检索相关信息，不浪费 token | 检索可能漏掉重要但不相关的信息 |
| 适合需要长期记忆的场景 | 实现复杂度较高 |

---

## 三大策略对比

| 维度 | 截断 | 总结 | 检索 |
|------|------|------|------|
| 实现难度 | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 复杂 |
| 额外成本 | 无 | 每次总结调用 LLM | 向量化 + 向量库 |
| 信息保留 | 只保留最近 N 轮 | 保留概要，丢细节 | 保留全部，按需检索 |
| 适用场景 | 短对话、闲聊 | 中长对话、客服 | 长期记忆、个人助手 |
| Token 控制 | 最好 | 中等 | 取决于检索数量 |
| 信息精度 | 高（原文） | 中（摘要） | 高（原文片段） |

---

## 最佳实践：组合使用

生产环境通常组合多种策略。

### 推荐方案：总结 + 缓冲

```python
"""
推荐方案：总结+缓冲组合
- 最近 1000 token 保留原文（精确）
- 更早的对话总结成摘要（保留长期信息）
- 重要信息可以额外存入向量库做检索
"""
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # 超过 1000 token 开始总结
    return_messages=True,
)
```

### 企业级 Agent 的 Memory 架构

```
┌─────────────────────────────────────────────────┐
│              当前对话上下文                        │
│  ┌───────────────────────────────────────────┐  │
│  │  系统 Prompt + 最近 5 轮原文（Window）     │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  历史对话摘要（Summary）                     │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  检索到的相关历史/知识库（Retrieval）        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 不同场景的 Memory 选型

| 场景 | 推荐策略 | 说明 |
|------|----------|------|
| 闲聊机器人 | 截断（Window） | 简单高效，不需要长期记忆 |
| 客服对话 | 总结 + 缓冲 | 保留对话概要，控制 token |
| 个人助手 | 检索 + 总结 | 需要长期记忆用户偏好 |
| 企业知识库问答 | 检索（RAG） | 从知识库检索，不是对话历史 |
| 多轮任务执行 | 截断 + 摘要 | 保留当前任务上下文 |

---

## 学习要点

1. **三大策略**：截断（简单但丢信息）、总结（保概要但丢细节）、检索（保全部但复杂）
2. **实际项目中组合使用**：最近对话保留原文 + 早期对话总结 + 重要信息检索
3. **`ConversationSummaryBufferMemory`** 是 LangChain 中最实用的记忆类型，兼顾精度和长期记忆
4. **检索型记忆类似 RAG**，把对话历史当知识库检索
5. **记忆管理的核心**是在"信息完整性"和"token 成本"之间找平衡
6. **生产环境还要考虑**：多用户隔离、记忆持久化（Redis/Milvus）、重要信息手动标注

## 扩展方向

- 学习 Mem0 长期记忆框架（分层记忆 + 三路召回）
- 探索基于 Redis 的短期记忆方案
- 学习 DeepAgents 的上下文压缩 middleware
- 实现多用户记忆隔离
- 探索记忆的重要性评分和自动清理机制

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/13-memory-strategies

包含本文的完整可运行代码示例（三种 Memory 策略的 Python 实现 + 对比测试）。

---

**上一篇**：[Milvus + RAG 实战](./12_Milvus+RAG实战.md) | **下一篇**：[结构化大模型输出](./14_结构化大模型输出.md)
