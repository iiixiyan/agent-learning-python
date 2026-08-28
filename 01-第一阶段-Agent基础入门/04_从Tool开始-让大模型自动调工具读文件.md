# 从 Tool 开始：让大模型自动调工具读文件

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

我们和大模型聊天，可以问它一些问题，它告诉你怎么做。

但是大模型没法帮你去**做**。

比如你想创建一个 React + Vite 的 todolist 项目，你直接问大模型，它只能告诉你应该创建哪些文件，代码是什么，但是不能帮你读写文件、执行命令。

但是 Cursor 是可以的——你让它创建一个 todolist 项目，它会直接给你写入文件；你还可以让它安装依赖，把项目跑起来。

**这是怎么实现的呢？开发一些 Tool 交给 Agent 调用就可以了。**

比如读文件、写文件、读取目录、创建目录、执行命令。

这节我们来学习 Tool 的开发和使用。

## 准备工作：选择大模型

这里我们用阿里的通义千问（Qwen），因为每个用户登录都有 100 万免费 token，够学习用了。当然，用别的大模型也一样，都可以。

### 获取 API Key

1. 登录 https://bailian.console.aliyun.com/?tab=api#/api
2. 在 API-KEY 管理页面获取你的 API Key

### 选择模型

搜索 coder 相关的编码模型（用于生成代码）。我们用 `qwen-coder-turbo` 这个就行。

> 每个模型训练的数据集不同，适用于不同目的。编码模型更适合写代码场景。

## 项目初始化

```bash
# 创建项目目录
mkdir tool-test
cd tool-test

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install langchain langchain-openai python-dotenv
```

## 第一步：调用大模型

创建 `src/hello_langchain.py`：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 初始化模型
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
)

# 调用模型
response = model.invoke("介绍下自己")
print(response.content)
```

创建 `.env` 文件：

```env
# OpenAI 兼容 API 配置
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置（可选，默认为 qwen-coder-turbo）
MODEL_NAME=qwen-coder-turbo
```

创建 `.gitignore`：

```
.env
venv/
__pycache__/
*.pyc
```

> **为什么用 .env？** 把 API Key 写死到代码里不好，通过 .env 文件管理，然后用 python-dotenv 读取。.env 文件包含私密信息，不提交到 Git，就像数据库密码一样。

运行测试：

```bash
python src/hello_langchain.py
```

可以看到模型调用成功了。

## 第二步：开发第一个 Tool（读文件）

创建 `src/tool_file_read.py`：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

load_dotenv()

# 1. 初始化模型
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-coder-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
)


# 2. 定义工具的输入参数
class ReadFileInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径，可以是相对路径或绝对路径")


# 3. 创建读文件工具
@tool("read_file", args_schema=ReadFileInput)
def read_file(file_path: str) -> str:
    """用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"  [工具调用] read_file('{file_path}') - 成功读取 {len(content)} 字符")
    return f"文件内容:\n{content}"


# 4. 绑定工具到模型
tools = [read_file]
model_with_tools = model.bind_tools(tools)

# 5. 构建消息
messages = [
    SystemMessage("""你是一个代码助手，可以使用工具读取文件并解释代码。

工作流程：
1. 用户要求读取文件时，立即调用 read_file 工具
2. 等待工具返回文件内容
3. 基于文件内容进行分析和解释

可用工具：
- read_file: 读取文件内容"""),
    HumanMessage("请读取 src/tool_file_read.py 文件内容并解释代码"),
]

# 6. 调用模型
response = model_with_tools.invoke(messages)
print("模型返回:", response)

# 7. 处理工具调用
while response.tool_calls and len(response.tool_calls) > 0:
    print(f"\n[检测到 {len(response.tool_calls)} 个工具调用]")
    messages.append(response)

    # 执行所有工具调用
    for tool_call in response.tool_calls:
        tool = next((t for t in tools if t.name == tool_call["name"]), None)
        if not tool:
            result = f"错误: 找不到工具 {tool_call['name']}"
        else:
            print(f"  [执行工具] {tool_call['name']}({tool_call['args']})")
            try:
                result = tool.invoke(tool_call["args"])
            except Exception as e:
                result = f"错误: {str(e)}"

        # 将工具结果添加到消息历史
        messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"],
        ))

    # 再次调用模型，传入工具结果
    response = model_with_tools.invoke(messages)

print("\n[最终回复]")
print(response.content)
```

安装额外依赖：

```bash
pip install pydantic
```

运行：

```bash
python src/tool_file_read.py
```

可以看到：检测到了 tool_calls 工具调用，用 read_file 工具读取了文件，然后让大模型分析了文件内容，给出了代码解释。

**现在大模型就能读文件了！这就是通过工具给大模型扩展了能力。**

## 核心概念解析

### Tool 是什么？

Tool 就是一个函数，加上名字、描述、参数格式。因为要给大模型用，你需要描述清楚：

- **名字（name）**：工具叫什么
- **描述（description）**：这个工具是干什么的，什么时候用
- **参数格式（args_schema）**：用 Pydantic 声明输入参数的类型和描述

### 四种 Message 类型

| 类型 | 说明 |
|------|------|
| **SystemMessage** | 设置 AI 是谁，可以干什么，有什么能力，以及回答、行为的规范 |
| **HumanMessage** | 用户输入的信息 |
| **AIMessage** | AI 的回复信息（可能包含 tool_calls） |
| **ToolMessage** | 调用工具的结果返回，需要用 tool_call_id 关联 |

### 工具调用流程

```
用户提问 → 模型判断需要调用工具 → 返回 tool_calls
    ↓
执行工具 → 获取结果 → 包装为 ToolMessage（带 tool_call_id）
    ↓
把 ToolMessage 加入消息历史 → 再次调用模型
    ↓
模型基于工具结果给出最终回答
```

> **关键点**：ToolMessage 必须用 tool_call_id 关联到对应的工具调用，告诉大模型"你让我调用的哪个工具，返回的结果是什么"。

## 新手常见问题

**Q: temperature 是什么？**
A: 温度，也就是 AI 的创造性。设置为 0，让它严格按照指令来做事情，不要自己发挥。

**Q: 为什么用 Pydantic 声明参数？**
A: Pydantic 会自动生成 JSON Schema，大模型需要知道工具的参数格式才能正确调用。

**Q: 可以同时调用多个工具吗？**
A: 可以。模型可能返回多个 tool_calls，循环执行即可。

## 总结

这节我们入门了 LangChain，调用了大模型，并且实现了第一个 Tool：

1. **环境准备**：用通义千问的免费模型，获取 API Key，用 .env 管理
2. **调用模型**：用 ChatOpenAI 初始化模型，invoke 调用
3. **创建 Tool**：用 @tool 装饰器 + Pydantic 参数声明，写一个读文件函数
4. **绑定工具**：用 model.bind_tools(tools) 把工具传给大模型
5. **处理调用**：检测 tool_calls → 执行工具 → ToolMessage 返回 → 再次调用模型

实现了第一个 Tool 之后，你可以想一下 Cursor 是怎么实现的——后面我们会实现一个简易版 Cursor！

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/04-tool-file-read

包含本文的完整可运行代码示例和环境配置。

---

**下一篇**：[实现 Mini Cursor：让 Agent 读写文件执行命令](./05_实现mini-cursor.md)
