# LangChain 整体总结：AI Agent 第一阶段学习完成

> **Python 版** | 基于 LangChain Python 技术栈
> 阶段：第一阶段总结 | 恭喜你完成了 Agent 基础入门！

---

## 第一阶段知识地图

```
Agent 基础入门
├── 核心概念
│   ├── LLM 大模型调用
│   ├── Prompt 提示词工程
│   └── Output Parser 结构化输出
├── Tool 工具调用
│   ├── @tool 装饰器定义工具
│   ├── Function Calling 原理
│   └── ReAct 循环（思考+行动）
├── MCP 协议
│   ├── Model Context Protocol
│   ├── MCP Server/Client
│   └── 跨进程跨语言复用工具
├── RAG 检索增强
│   ├── Document Loader 文档加载
│   ├── Text Splitter 文本切分
│   ├── Embedding 向量化
│   ├── Vector Store 向量数据库（Milvus/FAISS）
│   └── Retriever 检索器
├── Memory 记忆管理
│   ├── 截断（WindowMemory）
│   ├── 总结（SummaryMemory）
│   └── 检索（VectorStoreMemory）
└── LCEL 链式调用
    ├── Runnable 接口
    ├── | 管道符组装
    ├── RunnableParallel 并行
    └── RunnablePassthrough 透传
```

---

## 核心能力清单

学完第一阶段，你应该能：

| 能力 | 对应技术 | 难度 |
|------|----------|------|
| 调用大模型对话 | ChatOpenAI / invoke/stream | ⭐ 简单 |
| 让大模型调用工具 | @tool / bind_tools / ReAct | ⭐⭐ 中等 |
| 做一个知识库问答 | RAG（Loader→Splitter→Embedding→VectorStore→Retriever） | ⭐⭐⭐ 较难 |
| 让 Agent 记住对话 | Memory（截断/总结/检索） | ⭐⭐ 中等 |
| 结构化输出数据 | PydanticOutputParser / Function Calling | ⭐⭐ 中等 |
| 组装复杂工作流 | LCEL（prompt\|llm\|parser） | ⭐⭐ 中等 |
| 跨进程复用工具 | MCP 协议 | ⭐⭐⭐ 较难 |

---

## 最小可用 Agent 代码

这是一个完整的 Agent 示例，整合了第一阶段的所有知识：

```python
"""
最小可用 Agent：整合第一阶段所有知识
- Tool 工具调用
- Memory 记忆管理
- ReAct 循环（思考+行动）
- 结构化输出
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.memory import ConversationSummaryBufferMemory

load_dotenv()


# ========== 1. 定义工具 ==========

@tool
def search_web(query: str) -> str:
    """
    搜索网络信息

    Args:
        query: 搜索关键词

    Returns:
        str: 搜索结果
    """
    # 实际项目中接入真实搜索 API
    return f"关于'{query}'的搜索结果：这是一个模拟的搜索结果，包含相关信息。"


@tool
def calculate(expression: str) -> str:
    """
    数学计算

    Args:
        expression: 数学表达式，如 "123*456"

    Returns:
        str: 计算结果
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# ========== 2. 初始化模型和工具 ==========

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 工具列表
tools = [search_web, calculate]

# 绑定工具到模型
llm_with_tools = llm.bind_tools(tools)

# 工具映射（用于快速查找）
tool_map = {
    "search_web": search_web,
    "calculate": calculate,
}


# ========== 3. 初始化记忆 ==========

# 总结+缓冲记忆：最近对话保留原文，更早的总结成摘要
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,
    return_messages=True,
)


# ========== 4. Agent 主逻辑 ==========

def agent_chat(user_input: str, max_iterations: int = 5) -> str:
    """
    Agent 对话主函数

    Args:
        user_input: 用户输入
        max_iterations: 最大工具调用轮数

    Returns:
        str: AI 最终回答
    """
    # 构建消息列表
    messages = [
        SystemMessage(content="你是一个智能助手，可以搜索网络和进行数学计算。用中文回答，简洁专业。")
    ]

    # 加载历史记忆
    history = memory.load_memory_variables({}).get("history", [])
    messages.extend(history)

    # 添加用户输入
    messages.append(HumanMessage(content=user_input))

    # ReAct 循环：思考 → 行动 → 观察 → 思考...
    final_response = None
    for iteration in range(max_iterations):
        print(f"\n--- 第 {iteration + 1} 轮思考 ---")

        # 调用大模型（思考）
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 如果没有工具调用，说明任务完成
        if not response.tool_calls:
            final_response = response.content
            print(f"AI 回答: {final_response}")
            break

        # 有工具调用，执行工具（行动）
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"调用工具: {tool_name}，参数: {tool_args}")

            # 查找并执行工具
            tool = tool_map.get(tool_name)
            if tool:
                result = tool.invoke(tool_args)
                print(f"工具结果: {result}")

                # 添加工具结果到消息（观察）
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                ))
            else:
                messages.append(ToolMessage(
                    content=f"错误：未找到工具 {tool_name}",
                    tool_call_id=tool_id,
                ))

    # 保存到记忆
    if final_response:
        memory.save_context(
            {"input": user_input},
            {"output": final_response},
        )

    return final_response or "抱歉，我无法完成这个任务。"


# ========== 5. 使用示例 ==========

if __name__ == "__main__":
    print("=== 最小可用 Agent 演示 ===\n")

    # 测试1：普通对话
    print("【测试1：普通对话】")
    print(agent_chat("你好，我叫小明"))

    # 测试2：记忆测试
    print("\n【测试2：记忆测试】")
    print(agent_chat("我叫什么名字？"))  # 能记住

    # 测试3：工具调用测试
    print("\n【测试3：工具调用测试】")
    print(agent_chat("123*456等于多少？"))  # 能调用工具计算

    # 测试4：搜索测试
    print("\n【测试4：搜索测试】")
    print(agent_chat("搜索一下 LangChain 是什么"))
```

### 代码说明

| 组件 | 作用 | 对应知识点 |
|------|------|------------|
| `@tool` | 定义工具函数 | Tool 工具调用 |
| `bind_tools` | 绑定工具到模型 | Function Calling |
| `ConversationSummaryBufferMemory` | 总结+缓冲记忆 | Memory 记忆管理 |
| `for` 循环 + `tool_calls` 判断 | ReAct 循环 | ReAct 思考+行动 |
| `ToolMessage` | 工具结果消息 | Tool 工具调用 |
| `memory.save_context` | 保存对话记忆 | Memory 记忆管理 |

### ReAct 循环流程

```
用户输入
    ↓
构建消息（系统提示 + 历史记忆 + 用户输入）
    ↓
┌─────────────────────────────────────┐
│  循环（最多 max_iterations 次）       │
│                                     │
│  1. 思考：调用大模型                  │
│     ↓                               │
│  2. 判断：是否有 tool_calls？         │
│     ├── 否 → 任务完成，返回回答       │
│     └── 是 → 继续下一步               │
│     ↓                               │
│  3. 行动：执行工具                    │
│     ↓                               │
│  4. 观察：把工具结果加入消息           │
│     ↓                               │
│  5. 回到第1步，继续思考               │
└─────────────────────────────────────┘
    ↓
保存到记忆
    ↓
返回最终回答
```

---

## 第二阶段预告

第二阶段我们将学习企业级后端与进阶框架：

| 主题 | 内容 | 难度 |
|------|------|------|
| **FastAPI + SSE** | 把 Agent 封装成流式 API 接口 | ⭐⭐ 中等 |
| **定时任务** | 让 Agent 定时自动执行 | ⭐⭐ 中等 |
| **语音交互** | ASR 语音识别 + TTS 语音合成 | ⭐⭐⭐ 较难 |
| **LangGraph** | 图编排引擎，多 Agent 协作 | ⭐⭐⭐⭐ 难 |
| **Agentic RAG** | 自主决策的高级 RAG | ⭐⭐⭐⭐ 难 |
| **Docker Compose** | 一键部署开发环境 | ⭐⭐ 中等 |

---

## 学习建议

### 1. 多动手

每个知识点都写代码跑一遍，不要只看。理论和实践结合才能真正掌握。

### 2. 做项目

用第一阶段知识做一个完整的 Agent 应用，比如：
- 智能客服（Tool + Memory + RAG）
- 知识库问答（RAG + Milvus）
- 个人助手（Tool + Memory + 结构化输出）
- 代码助手（Tool + LCEL）

### 3. 查文档

LangChain 官方文档很详细，遇到问题先查文档：
- Python 版文档：https://python.langchain.com/
- API 参考：https://api.python.langchain.com/

### 4. 关注更新

LangChain 更新很快，关注：
- 官方博客：https://blog.langchain.dev/
- GitHub Release：https://github.com/langchain-ai/langchain/releases

### 5. 理解原理，不要死记 API

框架会变，但原理不变。理解了 ReAct、RAG、Memory 的原理，换任何框架都能快速上手。

---

## 常见问题

### Q1：LangChain 版本更新太快，代码经常报错怎么办？

A：建议固定版本号（`pip install langchain==0.3.0`），遇到报错先查官方文档的迁移指南。核心概念（Tool、RAG、Memory、LCEL）是稳定的，变化的主要是 API 细节。

### Q2：第一阶段学完能找到工作吗？

A：第一阶段是基础，能做简单的 Agent 应用。但企业级项目还需要第二阶段的知识（FastAPI、LangGraph、部署等）。建议学完两阶段后再找工作。

### Q3：需要学 JavaScript 版本吗？

A：不需要。Python 是 AI 开发的主流语言，生态更完善。掌握 Python 版就够了。

### Q4：大模型 API 费用高吗？

A：学习阶段费用很低（几块钱就能跑很多例子）。可以用通义千问、DeepSeek 等国产模型，价格更便宜。

---

## 恭喜！

你已经完成了 AI Agent 开发的第一阶段，掌握了核心基础知识。

接下来的第二阶段会更有挑战性，也更接近真实企业项目。

记住：**Agent 开发的核心不是框架，而是解决问题的思维方式**。框架只是工具，理解原理才能灵活运用。

祝你在第二阶段学习顺利！

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/19-langchain-summary

包含本文的完整可运行代码示例（最小可用 Agent + 第一阶段知识整合）。

---

**上一篇**：[实战练习 LCEL](./18_实战练习LCEL组装chain.md) | **下一篇**：[FastAPI + LangChain SSE 流式接口](../02-第二阶段-企业级后端与进阶框架/20_FastAPI+LangChain实现SSE流式接口.md)
