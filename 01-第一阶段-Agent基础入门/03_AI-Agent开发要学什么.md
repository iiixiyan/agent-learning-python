# AI Agent 开发要学什么？

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

我们每天都在用各种 AI Agent，比如用 Cursor 写代码，用 Manus 做各种自动化的事情。

那你有没有想过自己开发一个 AI Agent 产品呢？

## 大模型的局限性

有同学说，直接调大模型不就行了？

你可以想一下，大模型是不是有这些问题：

- 你上周和它聊过的消息，它是不是记不住？
- 你让它帮你访问一个网页，做一些事情，它是不是只能告诉你思路让你自己做？
- 你想让它基于你公司内部的私密文档来做一些解答，它是不是不知道那是什么？
- 你问它刚发生的一个新闻，它是不是不知道？因为它训练的时候没这些数据

这些问题的解决就需要分别给大模型加上 **Memory 记忆能力**、**Tool 工具调用能力**、**RAG 文档查询能力**等。

不管大模型怎么发展，它都没法解决这些：

- 它的记忆总是有限的，需要开发者做 Memory 管理
- 它可以调的特定领域的工具，需要开发者开发好 Tool 交给它
- 它没办法对不知道的私密文档做解答，需要扩展 RAG 知识库查询能力

这些就是我们要学的 AI 技术，也就是 AI Agent 的开发能力。

## Agent 是什么？

其实就是给大模型扩展了 Tool 和 Memory。

它本来就可以思考、规划，你给它用 Tool 扩展了能力，它就可以自动做事情了；用 Memory 管理了记忆，它就可以记住你想让它记住的东西；还可以用 RAG 查询内部知识库来获取知识。

这样一个**知道内部知识、能思考规划、有记忆，能够帮你做事情的扩展后的大模型**，就是一个 Agent。

## 实际案例

### Cursor 等 AI IDE

你每天用的 Cursor 等 AI IDE，它是怎么读写文件、执行命令的？

- 改代码需要读写文件 → 扩展了 Tool
- 帮你运行代码需要执行命令 → 扩展了 Tool

### Manus 自动化助手

帮你做各种事情的 Manus，是怎么唤起浏览器来自动访问一些网页，怎么执行命令，怎么帮你操作各种软件的？

- 打开浏览器、访问网页、点击元素 → 扩展了 Tool
- 总结成文档，写入 md 文件 → 扩展了 Tool

### 支付宝理财助手

支付宝里的理财助手，可以帮你分析基金，推荐基金，它是怎么知道这些基金的数据的？

肯定不是大模型本身就知道，而是基于 **RAG** 来访问了内部的知识库。

类似的各种需要给大模型扩展能力的场景太多了。

**光会和大模型聊天不行，你得能给它扩展各种能力，这样才能满足各种 AI 需求。**

## 用什么框架？

那 RAG、Memory、Tool、Agent 等用什么框架呢？

最常用的就是 **LangChain**，它对这些能力封装好了各种 API，可以直接用。

它有 Python 和 Node.js 版，本教程学的是 Python 版。

> 其实语言是次要的，等你学会了 AI 技术，换语言来写也是很简单的事情，逻辑、概念都是一样的。

### LangChain vs LangGraph

- **LangChain** 是用来开发单个 Agent 的，每个 Agent 做一件事情
- 但涉及到多个 Agent 协作，用 LangChain 自己维护多 Agent 的交互就比较麻烦了，这时候就要学习 **LangGraph**
- LangGraph 是基于 LangChain 封装的，用于多 Agent 交互的框架

## 为什么要结合后端一起学？

学 AI 不要只学用 AI 做一些工具，最好是结合后端技术一起学，也就是 **AI 全栈**。

因为 AI 代码是在后端跑的，这时候你需要用 Redis 存记忆、用 MySQL 存知识等，这种就需要后端的知识了。

所以我们是后端 + AI 一起学，做 AI 全栈产品，Python 后端框架用 **FastAPI**。

## 新手快速上手代码

```python
# 最简单的 Agent 示例：让大模型调用工具
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city}今天晴，温度25°C"

# 2. 初始化大模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. 创建 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    ("human", "{input}"),
    ("agent_scratchpad", "{agent_scratchpad}"),
])

tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. 运行
result = executor.invoke({"input": "北京今天天气怎么样？"})
print(result["output"])
```

## 总结

给大模型扩展 Tool、Memory、RAG 等能力，让它可以去做一些具体的事情，这就是一个 Agent 了。

平时我们用的 Cursor、Manus 等工具都是 AI Agent。

我们会学习：
- 开发 Agent 用的 **LangChain** 框架
- 开发 Multi Agent 用的 **LangGraph** 框架
- 基于 **Python + FastAPI** 来学，学会了后面换语言也很简单，API、逻辑都一样
- 结合后端技术一起学，开发 AI 全栈产品，这样才能让 AI 技术落地到产品里

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/03-what-is-agent

包含本文的完整可运行代码示例。

---

**下一篇**：[从 Tool 开始：让大模型自动调工具读文件](./04_从Tool开始-让大模型自动调工具读文件.md)
