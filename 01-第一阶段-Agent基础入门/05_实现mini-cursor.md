# 实现 Mini Cursor：大模型自动调用 Tool 执行命令

> 阶段：第一阶段 - Agent 基础入门 | 前置知识：Tool 调用基础

---

## 核心原理

Cursor 这类 AI 编程助手的核心是 **ReAct 循环**：思考 → 行动 → 观察 → 再思考，直到任务完成。

```
用户提问 → 大模型思考需要做什么 → 调用工具（行动）
    ↓
获取工具结果（观察）→ 大模型基于结果继续思考
    ↓
如果任务完成 → 给出总结；如果未完成 → 继续调用工具
```

## 环境准备

```bash
pip install langchain langchain-openai python-dotenv
```

## 第一步：定义工具

创建 `tools.py`：

```python
import subprocess
import os
from langchain_core.tools import tool


@tool
def run_shell(command: str) -> str:
    """执行 shell 命令，返回输出（超时30秒）"""
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        out = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
        return out[:5000]
    except Exception as e:
        return f"错误: {e}"


@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()[:5000]
    except Exception as e:
        return f"读取失败: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """写入文件内容"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"写入成功: {path}"
    except Exception as e:
        return f"写入失败: {e}"
```

**三个核心工具**：
- `run_shell`：执行 shell 命令（运行代码、安装依赖等）
- `read_file`：读取文件内容
- `write_file`：写入文件内容

## 第二步：Agent 主循环

创建 `mini_cursor.py`：

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tools import run_shell, read_file, write_file

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 绑定工具
tools = [run_shell, read_file, write_file]
llm_with_tools = llm.bind_tools(tools)
tool_map = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
}

# 系统提示词
SYSTEM = """你是编程助手，可执行 shell 命令、读写文件。
每次只调用一个工具，根据结果继续操作，任务完成后给出总结。
危险命令（rm -rf /等）禁止执行。"""


def run_agent(task: str, max_steps: int = 10):
    """运行 Agent，执行任务"""
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=task),
    ]

    for step in range(max_steps):
        print(f"\n=== 第 {step + 1} 步 ===")

        # 调用大模型
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        # 如果没有工具调用，说明任务完成
        if not resp.tool_calls:
            print("任务完成:", resp.content)
            return

        # 执行工具调用
        for tc in resp.tool_calls:
            print(f"调用 {tc['name']}: {tc['args']}")
            result = tool_map[tc["name"]].invoke(tc["args"])
            print(f"结果: {result[:150]}")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    print("达到最大步数，任务未完成")


if __name__ == "__main__":
    run_agent("创建 hello.py，写斐波那契数列前10项，然后运行")
```

## 运行

```bash
python mini_cursor.py
```

Agent 会自动完成：写文件 → 跑命令 → 分析结果 → 给出总结。

**示例输出流程**：
```
=== 第 1 步 ===
调用 write_file: {'path': 'hello.py', 'content': '...'}
结果: 写入成功: hello.py

=== 第 2 步 ===
调用 run_shell: {'command': 'python hello.py'}
结果: 0 1 1 2 3 5 8 13 21 34

=== 第 3 步 ===
任务完成: 已创建 hello.py 并成功运行，输出斐波那契数列前10项...
```

## 安全提示

- **加工作目录限制**：防止越权访问系统文件
- **危险命令加白名单或人工确认**：如 `rm -rf`、`sudo` 等
- **设置 max_steps**：防止死循环消耗 token
- **输出长度截断**：避免工具返回内容过长导致 token 溢出

## 学习要点

1. **ReAct 模式** = 思考 + 行动循环，是 Agent 的核心模式
2. **工具描述要清晰**：直接影响大模型是否选择正确的工具
3. **控制输出长度**：避免 token 溢出，工具返回结果要截断
4. **安全第一**：shell 执行必须有限制，不能让 Agent 随意执行危险命令
5. **max_steps 限制**：防止 Agent 陷入死循环

## 扩展方向

- 添加更多工具：目录浏览、文件搜索、Git 操作等
- 支持多轮对话：用户可以在 Agent 执行过程中追加指令
- 人工确认机制：危险操作前需要用户确认
- 工作区隔离：每个任务在独立目录中执行

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/05-mini-cursor

包含本文的完整可运行代码示例（tools.py + mini_cursor.py）。

---

**上一篇**：[从 Tool 开始：让大模型自动调工具读文件](./04_从Tool开始-让大模型自动调工具读文件.md) | **下一篇**：[MCP 协议：可跨进程调用的 Tool](./06_MCP-可跨进程调用的Tool.md)
