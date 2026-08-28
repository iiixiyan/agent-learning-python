# Output Parser 实战：智能录入 + 流式版 Mini Cursor

> **Python 版** | 基于 LangChain Python + Pydantic 技术栈
> 阶段：第一阶段 | 前置知识：结构化输出、Tool 调用

---

## 本文目标

上一篇学了 Output Parser 和 Tool 的区别。这篇用两个实战项目巩固：

1. **智能录入**：把自然语言转成结构化数据存入数据库
2. **流式版 Mini Cursor**：支持流式输出的命令执行 Agent

---

## 实战一：智能录入系统

### 场景

用户用自然语言描述一个人，系统自动提取姓名、年龄、职业等字段，存入 JSON/数据库。

```
用户输入: "张三，28岁，北京的软件工程师，会Python和Java，喜欢打篮球"
    ↓
大模型 + Output Parser
    ↓
结构化输出:
{
  "name": "张三",
  "age": 28,
  "occupation": "软件工程师",
  "skills": ["Python", "Java", "打篮球"],
  "city": "北京"
}
```

### 安装依赖

```bash
pip install langchain langchain-openai pydantic python-dotenv
```

### 代码实现

创建 `smart_input.py`：

```python
"""
智能录入系统：把自然语言转成结构化数据
"""
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


# 1. 定义数据结构
class Person(BaseModel):
    """人物信息结构"""
    name: str = Field(description="姓名")
    age: Optional[int] = Field(None, description="年龄，未知则为 null")
    occupation: str = Field(description="职业")
    skills: List[str] = Field(description="技能列表")
    city: Optional[str] = Field(None, description="所在城市")


# 2. 创建 parser
# PydanticOutputParser 会根据 Pydantic 模型自动生成格式说明，并解析结果
parser = PydanticOutputParser(pydantic_object=Person)

# 3. 创建 prompt
# 使用 partial_variables 预填充 format_instructions
prompt = PromptTemplate(
    template="根据以下描述提取人物信息。\n{format_instructions}\n描述：{description}",
    input_variables=["description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 创建 chain
# LCEL 语法：prompt | llm | parser
# 数据流：用户描述 → 填充 Prompt → 大模型生成 → Parser 解析为 Pydantic 对象
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)
chain = prompt | llm | parser

# 5. 运行
if __name__ == "__main__":
    result = chain.invoke({
        "description": "张三，28岁，北京的软件工程师，会Python和Java，喜欢打篮球"
    })

    print("解析结果（Pydantic 对象）:")
    print(result)
    # output: name='张三' age=28 occupation='软件工程师' skills=['Python', 'Java', '打篮球'] city='北京'

    # 6. 转字典存储
    print("\n转字典存储:")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
```

运行：

```bash
python smart_input.py
```

### 代码说明

| 组件 | 作用 |
|------|------|
| `Person(BaseModel)` | 定义数据结构，每个字段用 `Field(description=...)` 描述 |
| `PydanticOutputParser` | 根据 Pydantic 模型自动生成格式说明，并解析结果为 Pydantic 对象 |
| `PromptTemplate` | 定义 Prompt 模板，用 `partial_variables` 预填充格式说明 |
| `chain = prompt \| llm \| parser` | LCEL 语法，数据流从左到右 |

### 批量录入

```python
"""
批量录入示例
"""
descriptions = [
    "李四，35岁，上海的数据科学家，精通Python、SQL、机器学习",
    "王五，刚毕业的大学生，学的是计算机专业",
    "赵六在深圳做产品经理，擅长用户研究和需求分析",
]

print("批量录入结果:\n")
for desc in descriptions:
    person = chain.invoke({"description": desc})
    print(f"{person.name}: {person.occupation} in {person.city}")
    print(f"  技能: {', '.join(person.skills)}")
    print()
```

运行结果：

```
批量录入结果:

李四: 数据科学家 in 上海
  技能: Python, SQL, 机器学习

王五: 学生 in None
  技能: 计算机

赵六: 产品经理 in 深圳
  技能: 用户研究, 需求分析
```

### 扩展：存入数据库

```python
"""
扩展：把结构化数据存入 SQLite 数据库
"""
import sqlite3

# 创建数据库连接
conn = sqlite3.connect("people.db")
cursor = conn.cursor()

# 创建表
cursor.execute("""
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    occupation TEXT,
    skills TEXT,
    city TEXT
)
""")

# 插入数据
for desc in descriptions:
    person = chain.invoke({"description": desc})
    cursor.execute(
        "INSERT INTO people (name, age, occupation, skills, city) VALUES (?, ?, ?, ?, ?)",
        (person.name, person.age, person.occupation, ",".join(person.skills), person.city)
    )

conn.commit()
conn.close()
print("数据已存入数据库")
```

---

## 实战二：流式版 Mini Cursor

### 为什么需要流式？

普通 Agent 等大模型完整输出后才显示，用户体验差。流式输出可以边生成边显示，类似 ChatGPT 的打字机效果。

```
普通 Agent:
  用户提问 → 等待 5 秒 → 完整回答一次性显示

流式 Agent:
  用户提问 → 立即开始显示 → 边生成边显示 → 打字机效果
```

### 代码实现

创建 `streaming_agent.py`：

```python
"""
流式版 Mini Cursor：支持流式输出的命令执行 Agent
"""
import os
import json
import subprocess
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()


@tool
def run_cmd(command: str) -> str:
    """
    执行 shell 命令

    Args:
        command: 要执行的 shell 命令

    Returns:
        str: 命令执行结果（stdout + stderr），最多 3000 字符
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        return (result.stdout + result.stderr)[:3000]
    except Exception as e:
        return f"错误: {e}"


# 初始化大语言模型，开启流式输出
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    streaming=True,
)

# 绑定工具
tools = [run_cmd]
llm_with_tools = llm.bind_tools(tools)

# 系统提示词
SYSTEM_PROMPT = "你是编程助手，可执行 shell 命令。用中文回答。"


def run_streaming_agent(task: str):
    """
    运行流式 Agent

    Args:
        task: 用户任务描述
    """
    # 初始化消息列表
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    while True:
        print("\n思考中...", end="", flush=True)

        # 流式获取大模型输出
        tool_calls = []       # 存储工具调用（可能有多个）
        content_chunks = []   # 存储文本内容的 chunk

        for chunk in llm_with_tools.stream(messages):
            # 处理文本内容
            if chunk.content:
                print(chunk.content, end="", flush=True)
                content_chunks.append(chunk.content)

            # 处理工具调用的增量 chunk
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    if tc.get("index") is not None:
                        idx = tc["index"]
                        # 确保 tool_calls 列表有足够的长度
                        while len(tool_calls) <= idx:
                            tool_calls.append({"name": "", "args": "", "id": ""})

                        # 增量拼接工具调用的各个字段
                        if tc.get("name"):
                            tool_calls[idx]["name"] += tc["name"]
                        if tc.get("args"):
                            tool_calls[idx]["args"] += tc["args"]
                        if tc.get("id"):
                            tool_calls[idx]["id"] += tc["id"]

        # 构建完整文本内容
        full_content = "".join(content_chunks)

        if tool_calls:
            # 有工具调用，执行工具
            for tc in tool_calls:
                print(f"\n\n执行命令: {tc['args']}")

                # 解析工具参数（args 可能是 JSON 字符串）
                args = json.loads(tc["args"]) if isinstance(tc["args"], str) else tc["args"]

                # 执行工具
                result = run_cmd.invoke(args)
                print(f"结果:\n{result}")

                # 把工具结果加入消息列表，继续循环
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # 没有工具调用，任务完成
            print("\n\n任务完成")
            break


if __name__ == "__main__":
    run_streaming_agent("查看当前目录有哪些文件，然后创建一个 test.py 写 hello world")
```

运行：

```bash
python streaming_agent.py
```

### 流式输出的关键技术点

| 技术点 | 说明 |
|--------|------|
| `streaming=True` | 初始化模型时开启流式输出 |
| `llm_with_tools.stream(messages)` | 流式调用，返回异步迭代器 |
| `chunk.content` | 文本内容的增量 chunk，直接打印实现打字机效果 |
| `chunk.tool_call_chunks` | 工具调用的增量 chunk，需要手动拼接 |
| `tc["index"]` | 工具调用的索引，支持同时调用多个工具 |
| 增量拼接 | name、args、id 字段都需要增量拼接，因为它们分散在多个 chunk 中 |

### tool_call_chunks 的数据结构

```python
# 流式输出时，tool_call_chunks 的每个 chunk 可能只包含部分信息
{
    "index": 0,              # 工具调用的索引（第几个工具）
    "name": "run_",          # 工具名称的一部分（增量）
    "args": '{"command": "l', # 工具参数的一部分（增量）
    "id": "call_abc123"      # 工具调用 ID（可能在最后一个 chunk 才完整）
}

# 需要把所有 chunk 拼接起来，才能得到完整的工具调用
{
    "name": "run_cmd",
    "args": '{"command": "ls -la"}',
    "id": "call_abc123"
}
```

### Agent 工作流程

```
用户任务
    ↓
大模型流式输出
    ├── 文本内容 → 边生成边打印（打字机效果）
    └── 工具调用 → 收集完整 tool_calls
                        ↓
                    执行工具
                        ↓
                    工具结果加入消息
                        ↓
                    继续循环（大模型根据工具结果继续回答）
                        ↓
                    没有工具调用 → 任务完成
```

---

## 学习要点

1. **PydanticOutputParser** 是结构化输出的首选，类型安全，能自动生成格式说明
2. **LCEL 语法** `prompt | llm | parser` 可以优雅地构建结构化输出流水线
3. **流式输出**需要处理 `tool_call_chunks` 的增量拼接，这是最容易出错的地方
4. **智能录入**的关键是设计好数据结构和 Field 描述，描述越清晰，提取越准确
5. **流式 Agent** 的体验比普通 Agent 好很多，是生产环境的标配
6. **`Optional` 类型**可以处理字段缺失的情况，避免解析失败

## 扩展方向

- 智能录入：支持更多字段类型（日期、嵌套对象、枚举）
- 智能录入：添加数据校验和错误处理
- 流式 Agent：支持多轮对话记忆
- 流式 Agent：添加工具执行进度显示
- 流式 Agent：支持文件读写工具
- 结合 FastAPI 构建 Web 版智能录入和流式 Agent

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/15-output-parser-practice

包含本文的完整可运行代码示例（智能录入系统 + 流式版 Mini Cursor）。

---

**上一篇**：[结构化大模型输出](./14_结构化大模型输出.md) | **下一篇**：[Prompt Template](./16_Prompt-Template.md)
