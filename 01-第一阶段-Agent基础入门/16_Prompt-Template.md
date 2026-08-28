# Prompt Template：组件化管理 Prompt

> **Python 版** | 基于 LangChain Python 技术栈
> 阶段：第一阶段 | 为什么需要模板：Prompt 写死在代码里难以维护和复用

---

## 为什么需要 Prompt Template？

新手常见的写法：

```python
# 直接用 f-string 拼接 Prompt
prompt = f"你是一个翻译官，把以下文本翻译成{language}：\n{text}"
```

### 存在的问题

| 问题 | 说明 |
|------|------|
| **难以维护** | Prompt 散落在代码各处，修改需要找遍整个项目 |
| **容易出错** | 变量多了字符串拼接容易漏变量、拼错格式 |
| **无法复用** | 相同的 Prompt 逻辑在不同地方重复写 |
| **难以测试** | Prompt 和业务逻辑混在一起，无法单独测试 Prompt 效果 |
| **无法版本管理** | Prompt 的变更没有记录，出问题难以回滚 |

LangChain 的 `PromptTemplate` 解决了这些问题，让 Prompt 变成可复用、可维护、可测试的组件。

---

## 基础用法

### 安装依赖

```bash
pip install langchain langchain-openai python-dotenv
```

### 方式一：from_template（推荐）

```python
"""
PromptTemplate 基础用法：from_template
"""
from langchain_core.prompts import PromptTemplate

# 用 from_template 创建模板，自动识别变量
prompt = PromptTemplate.from_template("你是一个{role}，请回答：{question}")

# 格式化模板，传入变量
formatted = prompt.format(role="Python专家", question="什么是装饰器？")
print(formatted)
# 输出: 你是一个Python专家，请回答：什么是装饰器？

# 查看模板的输入变量
print(f"\n输入变量: {prompt.input_variables}")
# 输出: 输入变量: ['role', 'question']
```

### 方式二：直接构造

```python
"""
PromptTemplate 基础用法：直接构造
"""
from langchain_core.prompts import PromptTemplate

# 直接构造，显式指定输入变量和模板
prompt2 = PromptTemplate(
    input_variables=["language", "text"],
    template="把以下文本翻译成{language}：\n{text}"
)

# 格式化
formatted = prompt2.format(language="英语", text="你好，世界")
print(formatted)
# 输出:
# 把以下文本翻译成英语：
# 你好，世界
```

### 两种方式对比

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| `from_template` | 简单，自动识别变量 | 变量名写错不会报错 | 大多数场景（推荐） |
| 直接构造 | 显式指定变量，更安全 | 需要手动列变量 | 变量多、需要严格校验的场景 |

---

## 常用模板类型

### 1. ChatPromptTemplate（对话模板）

对话场景需要区分 system、human、ai 等角色，用 `ChatPromptTemplate`。

```python
"""
ChatPromptTemplate：对话模板，支持多角色消息
"""
from langchain_core.prompts import ChatPromptTemplate

# 从消息列表创建对话模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用中文回答，简洁专业。"),
    ("human", "{question}"),
])

# 格式化消息
messages = chat_prompt.format_messages(role="Python讲师", question="解释GIL")
print("格式化后的消息列表:")
for msg in messages:
    print(f"  [{msg.type}] {msg.content}")

# 也可以直接 format 成字符串
formatted_str = chat_prompt.format(role="Python讲师", question="解释GIL")
print(f"\n格式化后的字符串:\n{formatted_str}")
```

#### 消息角色说明

| 角色 | 说明 | 用途 |
|------|------|------|
| `system` | 系统消息 | 设置角色、规则、约束 |
| `human` | 用户消息 | 用户的问题、输入 |
| `ai` | AI 消息 | AI 的回答、示例输出 |
| `tool` | 工具消息 | 工具执行结果 |

#### 多种创建方式

```python
# 方式一：元组形式（最常用）
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    ("human", "{question}"),
])

# 方式二：消息对象形式
from langchain_core.messages import SystemMessage, HumanMessage
chat_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="{question}"),
])

# 方式三：混合形式
chat_prompt3 = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个{role}"),
    ("human", "{question}"),
])
```

---

### 2. FewShotPromptTemplate（少样本提示）

给大模型几个示例，引导它按照示例的格式输出。

```python
"""
FewShotPromptTemplate：少样本提示，用示例引导输出格式
"""
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 示例列表
examples = [
    {"input": "开心", "output": "😊"},
    {"input": "难过", "output": "😢"},
    {"input": "生气", "output": "😠"},
]

# 单个示例的模板
example_prompt = PromptTemplate.from_template("输入：{input}\n输出：{output}")

# 创建 FewShot 模板
few_shot = FewShotPromptTemplate(
    examples=examples,                    # 示例列表
    example_prompt=example_prompt,        # 单个示例的模板
    suffix="输入：{input}\n输出：",        # 示例后面的后缀（用户输入）
    input_variables=["input"],             # 输入变量
)

# 格式化
formatted = few_shot.format(input="惊讶")
print(formatted)
# 输出:
# 输入：开心
# 输出：😊
#
# 输入：难过
# 输出：😢
#
# 输入：生气
# 输出：😠
#
# 输入：惊讶
# 输出：
```

#### FewShot 的使用场景

| 场景 | 说明 |
|------|------|
| **格式引导** | 让大模型按照特定格式输出（如 JSON、表格、特定句式） |
| **风格模仿** | 让大模型模仿示例的回答风格 |
| **任务示范** | 复杂任务先给几个示例，大模型更容易理解任务要求 |
| **少样本学习** | 大模型可以从少量示例中学习任务模式 |

#### 示例数量建议

- **2-5 个示例**：大多数场景的最佳选择
- 太少（1个）：效果不稳定，大模型可能不理解模式
- 太多（>10个）：增加 token 消耗，可能导致过拟合

---

### 3. PipelinePromptTemplate（管道组合）

把大 Prompt 拆成多个小组件，然后组合起来。适合复杂的 Prompt 管理。

```python
"""
PipelinePromptTemplate：把大 Prompt 拆成小组件组合
"""
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

# 小组件1：角色设定
introduction = PromptTemplate.from_template("你是{persona}。")

# 小组件2：任务描述
task = PromptTemplate.from_template("你的任务是{task}。")

# 小组件3：示例
example = PromptTemplate.from_template("示例：{example}")

# 最终的完整模板（引用小组件的输出）
full_prompt = PromptTemplate.from_template(
    "{introduction}\n{task}\n{example}\n{input}"
)

# 创建管道模板
pipeline = PipelinePromptTemplate(
    final_prompt=full_prompt,              # 最终模板
    pipeline_prompts=[                      # 管道中的小组件
        ("introduction", introduction),     # (变量名, 模板)
        ("task", task),
        ("example", example),
    ]
)

# 格式化
formatted = pipeline.format(
    persona="Python专家",
    task="解释代码",
    example="代码: print(1) → 输出1",
    input="代码: [x*2 for x in range(5)]"
)
print(formatted)
# 输出:
# 你是Python专家。
# 你的任务是解释代码。
# 示例：代码: print(1) → 输出1
# 代码: [x*2 for x in range(5)]
```

#### Pipeline 的优势

| 优势 | 说明 |
|------|------|
| **模块化** | 每个小组件独立维护，修改不影响其他部分 |
| **复用性** | 小组件可以在不同的大 Prompt 中复用 |
| **可读性** | 大 Prompt 拆成小组件，结构更清晰 |
| **可测试** | 每个小组件可以单独测试效果 |

---

## 实战：RAG 问答模板

RAG（检索增强生成）是最常用的场景之一，需要精心设计 Prompt 模板。

```python
"""
RAG 问答模板实战
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# RAG 标准模板
rag_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的问答助手。
根据以下参考资料回答用户问题。

要求：
1. 只根据参考资料回答，不要编造
2. 如果资料中没有，回答"参考资料中未找到相关信息"
3. 回答要简洁准确
4. 引用资料中的原文时用引号标注

参考资料：
{context}"""),
    ("human", "{question}"),
])

# 初始化大语言模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 创建 Chain：模板 → 大模型
chain = rag_template | llm

# 使用
response = chain.invoke({
    "context": "Python是一种解释型、面向对象的编程语言，由Guido van Rossum于1991年发布。Python以简洁易读的语法著称，广泛应用于Web开发、数据分析、人工智能等领域。",
    "question": "Python是谁发明的？"
})

print("AI 回答:")
print(response.content)
```

### RAG Prompt 设计要点

| 要点 | 说明 |
|------|------|
| **角色设定** | 明确告诉大模型它是"专业的问答助手" |
| **参考资料位置** | 把检索到的资料放在 system 消息中，变量名用 `context` |
| **约束规则** | 明确要求"只根据参考资料回答，不要编造"，减少幻觉 |
| **兜底处理** | 如果资料中没有相关信息，明确回答"未找到"，而不是编造 |
| **引用标注** | 要求引用原文时用引号标注，方便用户查证 |
| **温度设置** | RAG 场景建议 `temperature=0`，保证回答准确稳定 |

---

## 最佳实践

### 1. 变量命名清晰

```python
# ❌ 不好的命名
prompt = PromptTemplate.from_template("你是一个{a}，请回答{b}")

# ✅ 好的命名
prompt = PromptTemplate.from_template("你是一个{role}，请回答{question}")
```

### 2. System Prompt 放角色和规则，Human Prompt 放具体问题

```python
# ✅ 推荐的结构
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个{role}。
规则：
1. 用中文回答
2. 简洁专业
3. 不确定时说"我不知道" """),
    ("human", "{question}"),
])
```

### 3. FewShot 用 2-5 个示例

```python
# ✅ 合适的示例数量（3个）
examples = [
    {"input": "开心", "output": "😊"},
    {"input": "难过", "output": "😢"},
    {"input": "生气", "output": "😠"},
]
```

### 4. 模板单独存放

复杂项目可以把 Prompt 放到单独的文件中管理：

```
project/
├── prompts/
│   ├── __init__.py
│   ├── rag_prompts.py      # RAG 相关模板
│   ├── agent_prompts.py    # Agent 相关模板
│   └── translation_prompts.py  # 翻译相关模板
├── main.py
└── .env
```

```python
# prompts/rag_prompts.py
from langchain_core.prompts import ChatPromptTemplate

RAG_QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的问答助手..."""),
    ("human", "{question}"),
])
```

### 5. 用 partial_variables 预填常量

有些变量是固定的（如格式说明），可以用 `partial_variables` 预填：

```python
"""
partial_variables：预填常量，减少调用时的参数
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# 创建模板时预填 format_instructions
prompt = PromptTemplate(
    template="提取人物信息。\n{format_instructions}\n描述：{description}",
    input_variables=["description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 调用时只需要传 description，不需要传 format_instructions
formatted = prompt.format(description="张三，28岁，软件工程师")
print(formatted)
```

---

## 学习要点

1. **PromptTemplate** 让 Prompt 可复用、可维护、可测试，是工程化的基础
2. **ChatPromptTemplate** 支持多角色消息（system/human/ai/tool），对话场景必用
3. **FewShotPromptTemplate** 用示例引导大模型输出格式，2-5 个示例效果最佳
4. **PipelinePromptTemplate** 把大 Prompt 拆成小组件组合，适合复杂项目
5. **RAG 场景**一定要在 system prompt 里强调"只根据资料回答"，减少幻觉
6. **变量命名要清晰**，用 `context`、`question` 而不是 `a`、`b`
7. **partial_variables** 可以预填常量（如格式说明），减少调用时的参数
8. **复杂项目**把 Prompt 放到单独的文件中管理，便于维护和版本控制

## 扩展方向

- 学习 Prompt 版本管理工具（如 LangSmith、PromptLayer）
- 探索动态 FewShot（根据输入选择最相关的示例）
- 学习 Prompt 优化方法论（如 CoT、ToT、ReAct）
- 结合 LCEL 构建复杂的 Prompt 流水线
- 探索多语言 Prompt 模板的管理

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/16-prompt-template

包含本文的完整可运行代码示例（4种 PromptTemplate 类型 + RAG 实战 + 最佳实践）。

---

**上一篇**：[Output Parser 实战](./15_Output-Parser实战.md) | **下一篇**：[Runnable：把写逻辑变成组装 Chain](./17_Runnable-把写逻辑变成组装chain.md)
