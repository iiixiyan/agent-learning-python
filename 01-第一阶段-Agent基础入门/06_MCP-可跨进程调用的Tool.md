# MCP：可跨进程调用的 Tool

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

## 为什么需要 MCP？

我们已经写了一些 Tool 了：读写文件和目录、执行命令。

只要声明 Tool 的名字、描述、参数格式，模型会在发现需要用 Tool 的时候自动解析出参数传入来调用，然后把执行结果封装成 ToolMessage 传入对话。

比如上节我们实现了简易的 Cursor，就是声明了读写文件和目录、执行命令的 Tool，这样你让大模型创建 React + Vite 项目，它就会自动判断什么时候调用哪个 Tool，自动实现目录、文件的创建，以及 `pip install` 和 `python run` 的执行。

我们只是告诉它要创建的项目，然后安装依赖跑起来。这些 Tool 怎么调用、参数是什么都是大模型自己决定的。

**Tool 给大模型扩展了做事情的能力**：本来它只能思考，不能做事情，但是现在可以自己调用 Tool 来帮你做事情了。

## Tool 的局限性

但你有没有发现 Tool 有个问题：

Python 写的 AI Agent 代码，你的 Tool 也得是 Python 写的。

如果你之前有一些工具是 Java、Rust、Go 写的呢？你想封装成 Tool 怎么办呢？

### 方案一：通过子进程调用

有的同学说：现在不是可以执行命令么，通过单独进程把这些其他语言写的代码跑一下就行啊。

确实，也就是这样：

```
AI Agent (Python)
    ↓ stdio (标准输入输出)
子进程 (Java/Rust/Go 写的工具)
```

这里的 stdio 就是标准输入输出流，也就是键盘输入、控制台输出。当你进程跑一个子进程，就可以用这种方式通信。

### 方案二：通过 HTTP 调用

还有的同学说：简单，用 HTTP 啊！本地跑个服务就好了。

```
AI Agent (Python)
    ↓ HTTP 请求
本地/远程服务 (任意语言写的工具)
```

现在是解决了跨语言调用工具的问题。

## 统一协议的必要性

那如果每个人都这样搞，它们提供的服务都不一样，我想接入别的 Tool，是不是要了解每个服务都是怎么定义的呢？

**能不能定义一个统一的通信协议，我们都按照这个格式来沟通，这样所有的跨进程工具调用就都可以接入了。**

```
AI Agent (MCP Client)
    ↓ MCP 协议 (stdio / http)
各种 MCP Server (任意语言)
```

想跨进程调用某个工具，通过这个协议通信就行。不管是本地工具，直接跑那个进程，然后 stdio 通信；还是远程工具，通过 HTTP 连接远程服务进程。

**这个协议就叫 MCP（Model Context Protocol）**——给 Model 扩展 Context 上下文，让它能做的更多、知道的更多的协议。

## MCP 核心概念

MCP 最大的特点就是可以**跨进程调用工具**：

- 跨本地的进程调用，就是用 **stdio**
- 跨远程的进程调用，就是用 **HTTP**

### MCP 架构图

```
┌─────────────────────────────────────────┐
│           AI Agent (MCP Client)          │
│  (Cursor / LangChain / Claude Desktop)  │
└──────────────┬──────────────────────────┘
               │ MCP 协议
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌───────┐ ┌───────┐ ┌───────┐
│MCP    │ │MCP    │ │MCP    │
│Server1│ │Server2│ │Server3│
│(Python)│ │(Node) │ │(Rust) │
└───────┘ └───────┘ └───────┘
```

你的 AI Agent 就是 MCP 客户端，可以通过 MCP 协议调用各种 MCP Server，实现跨进程的工具调用。

当然，在 LangChain 里，它也是 Tool，只不过是 Tool 的一种而已。

> MCP 由 AI 巨头 Anthropic 公司发起并开发，2025 年 12 月交给了 Linux 基金会维护。也就是说它现在是完全中立于任何一个模型的行业通用协议。

## 实战：用 Python 写一个 MCP Server

继续在 tool-test 项目里写。

### 安装依赖

```bash
pip install mcp langchain-mcp-adapters
```

### 创建 MCP Server

创建 `src/my_mcp_server.py`：

```python
"""
MCP Server 示例：提供用户查询工具和使用指南资源
"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource
import asyncio

# 模拟数据库
database = {
    "users": {
        "001": {"id": "001", "name": "张三", "email": "zhangsan@example.com", "role": "admin"},
        "002": {"id": "002", "name": "李四", "email": "lisi@example.com", "role": "user"},
        "003": {"id": "003", "name": "王五", "email": "wangwu@example.com", "role": "user"},
    }
}

# 创建 MCP Server 实例
server = Server("my-mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的工具"""
    return [
        Tool(
            name="query_user",
            description="查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "userId": {
                        "type": "string",
                        "description": "用户 ID，例如: 001, 002, 003"
                    }
                },
                "required": ["userId"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """调用工具"""
    if name == "query_user":
        user_id = arguments.get("userId", "")
        user = database["users"].get(user_id)

        if not user:
            return [TextContent(
                type="text",
                text=f"用户 ID {user_id} 不存在。可用的 ID: 001, 002, 003"
            )]

        return [TextContent(
            type="text",
            text=f"用户信息：\n- ID: {user['id']}\n- 姓名: {user['name']}\n- 邮箱: {user['email']}\n- 角色: {user['role']}"
        )]

    return [TextContent(type="text", text=f"未知工具: {name}")]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """列出可用的资源（静态数据）"""
    return [
        Resource(
            uri="docs://guide",
            name="使用指南",
            description="MCP Server 使用文档",
            mimeType="text/plain"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源内容"""
    if uri == "docs://guide":
        return """MCP Server 使用指南

功能：提供用户查询等工具。
使用：在 Cursor 等 MCP Client 中通过自然语言对话，Client 会自动调用相应工具。
"""
    return ""


async def main():
    """启动 MCP Server（stdio 模式）"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

**代码解析**：

- `Server` 创建了 MCP Server 实例
- `@server.list_tools()` 注册工具列表，声明 name、description、inputSchema
- `@server.call_tool()` 处理工具调用
- `@server.list_resources()` / `@server.read_resource()` 注册资源（静态数据）
- `stdio_server()` 提供 stdio 的本地进程通信方式

> Resource 和 Tool 的区别：Resource 一般返回静态数据（read），Tool 来做一些事情（call）。

运行 MCP Server：

```bash
python src/my_mcp_server.py
```

这样，我们的 MCP 服务就创建好了！其实就是 Tool，加上了协议而已。

## 在 LangChain 中调用 MCP Server

创建 `src/langchain_mcp_test.py`：

```python
"""
LangChain 调用 MCP Server 示例
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

load_dotenv()


async def main():
    # 1. 初始化大模型
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    # 2. 创建 MCP Client，连接到 MCP Server
    mcp_client = MultiServerMCPClient({
        "my-mcp-server": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(__file__), "my_mcp_server.py")]
        }
    })

    # 3. 获取 MCP 工具并绑定到模型
    async with mcp_client as client:
        tools = await client.get_tools()
        model_with_tools = model.bind_tools(tools)

        # 4. 读取 Resource（可选，放入 SystemMessage）
        resources = await client.list_resources()
        resource_content = ""
        for server_name, res_list in resources.items():
            for res in res_list:
                content = await client.read_resource(server_name, res.uri)
                resource_content += content[0].text + "\n"

        # 5. 运行 Agent
        messages = [
            SystemMessage(content=resource_content) if resource_content else SystemMessage(content=""),
            HumanMessage(content="查一下用户 002 的信息")
        ]

        max_iterations = 10
        for i in range(max_iterations):
            print(f"\n=== 第 {i + 1} 步 ===")
            response = await model_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                print(f"\n最终回复:\n{response.content}")
                break

            print(f"检测到 {len(response.tool_calls)} 个工具调用")
            for tool_call in response.tool_calls:
                print(f"调用 {tool_call['name']}: {tool_call['args']}")
                found_tool = next((t for t in tools if t.name == tool_call["name"]), None)
                if found_tool:
                    tool_result = await found_tool.ainvoke(tool_call["args"])
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    ))


if __name__ == "__main__":
    asyncio.run(main())
```

运行测试：

```bash
python src/langchain_mcp_test.py
```

可以看到：你让大模型查询用户，它识别到了工具调用，然后调用了 MCP 的工具。

## MCP 的核心价值

**MCP 本质上还是 Tool，和之前的 Tool 的区别只不过是可以跨进程调用。**

跨进程就意味着不限语言，开发好之后，可以被任意 MCP Client 调用，比如 Cursor、LangChain、Claude Desktop 等。

**这就是 MCP 的好处：写好之后可以插拔到任何地方当 Tool 用。**

### 什么时候用 MCP？

- ✅ 需要跨语言调用工具（Python 调 Java/Rust 写的工具）
- ✅ 需要远程调用工具（工具部署在另一台服务器）
- ✅ 需要工具被多个 Client 共享（Cursor、LangChain 都能用）
- ❌ 不需要跨进程时，还是之前那样直接写 Tool 更好，还少了进程通信的成本

## 学习要点

1. **MCP = 可跨进程调用的 Tool**，通过统一协议实现工具复用
2. **两种通信方式**：本地用 stdio，远程用 HTTP
3. **MCP Server 提供两类能力**：Tool（执行操作）和 Resource（提供静态数据）
4. **LangChain 中用 `langchain-mcp-adapters`** 把 MCP Server 封装成 Tools 来用
5. **MCP 是行业通用协议**，由 Linux 基金会维护，中立于任何模型厂商

## 扩展方向

- 使用现成的 MCP Server（文件系统、数据库、浏览器等）
- 开发 HTTP 传输模式的 MCP Server（远程调用）
- 在 Cursor / Claude Desktop 中配置使用自己的 MCP Server
- 组合多个 MCP Server 构建复杂的 Agent 系统

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/06-mcp-protocol

包含本文的完整可运行代码示例（MCP Server + LangChain Client）。

---

**上一篇**：[实现 Mini Cursor：大模型自动调用 Tool 执行命令](./05_实现mini-cursor.md) | **下一篇**：[高德 MCP + 浏览器 MCP：LangChain 复用别人的 MCP Server](./07_高德MCP+浏览器MCP.md)
