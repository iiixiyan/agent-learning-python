# LangSmith 全链路观测：从 Agent 调试到 RAG 量化评估

> **Python 版** | 基于 LangSmith + LangChain Python 技术栈
> 前置知识：[Agent 基础](../01-第一阶段-Agent基础入门/08_Agent基础.md)、[RAG 基础](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)

---

## 为什么需要 LangSmith？

开发 Agent 和 RAG 应用时，常见痛点：

| 痛点 | 说明 |
|------|------|
| **调试困难** | Agent 调用链复杂，不知道哪一步出了问题 |
| **性能未知** | 不知道每次调用耗时多少、token 消耗多少 |
| **效果无法量化** | RAG 回答好不好，全靠人工感觉，没有量化指标 |
| **优化无方向** | 不知道改 Prompt 还是改检索，没有数据支撑 |
| **线上问题难排查** | 用户反馈回答错误，无法复现当时的调用链 |

**LangSmith** 是 LangChain 官方推出的全链路观测和评估平台，解决以上所有问题。

### LangSmith 核心能力

| 能力 | 说明 |
|------|------|
| **追踪（Tracing）** | 自动记录 Agent 运行过程，包括每一步的输入输出、耗时、token |
| **调试（Debugging）** | 可视化调用链，快速定位问题节点 |
| **数据集（Datasets）** | 创建问题和答案的测试集，用于回归测试 |
| **评估（Evaluation）** | 用评估器自动打分，量化 RAG/Agent 效果 |
| **实验（Experiments）** | 对比不同版本的效果，A/B 测试 |
| **监控（Monitoring）** | 线上应用的性能和质量监控 |

---

## LangSmith 快速开始

### 1. 注册和获取 API Key

1. 访问 [LangSmith 官网](https://smith.langchain.com/)，注册账号
2. 创建 Organization 和 Project
3. 在 Settings → API Keys 中创建 API Key

### 2. 环境配置

创建 `.env` 文件：

```bash
# LangSmith 配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的_api_key
LANGCHAIN_PROJECT=agent-learning-demo

# 大模型配置（以 OpenAI 兼容接口为例）
OPENAI_API_KEY=你的_api_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
```

### 3. 安装依赖

```bash
pip install langchain langchain-openai langsmith python-dotenv pandas
```

---

## 追踪（Tracing）

### 基础追踪示例

创建 `tracing_demo.py`：

```python
"""
tracing_demo.py - LangSmith 追踪基础示例
自动记录 LLM 调用、Chain 执行、Agent 运行过程
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 创建 Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个Python编程专家，用简洁明了的方式回答问题。"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 调用 Chain（自动追踪）
result = chain.invoke({"question": "什么是装饰器？请举一个简单例子。"})
print(f"回答: {result}")

# 多次调用，查看追踪记录
questions = [
    "Python中的列表和元组有什么区别？",
    "如何处理Python中的异常？",
    "什么是生成器？和迭代器有什么区别？",
]

for q in questions:
    result = chain.invoke({"question": q})
    print(f"\nQ: {q}")
    print(f"A: {result[:100]}...")

print("\n✅ 所有调用已自动追踪到 LangSmith！")
print(f"访问: https://smith.langchain.com/ 查看追踪记录")
```

### 运行并查看追踪

```bash
python tracing_demo.py
```

运行后，在 LangSmith 控制台可以看到：

| 信息 | 说明 |
|------|------|
| **调用链可视化** | 每一步的输入输出、耗时、token 消耗 |
| **Latency** | 每次调用的耗时 |
| **Token Usage** | prompt tokens、completion tokens、总 tokens |
| **Status** | 成功/失败状态 |
| **Inputs/Outputs** | 每一步的输入和输出内容 |

---

## Agent 追踪示例

创建 `agent_tracing_demo.py`：

```python
"""
agent_tracing_demo.py - Agent 全链路追踪示例
追踪 Agent 的思考过程、工具调用、最终回答
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


# 定义工具
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        str: 计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def search_python_docs(topic: str) -> str:
    """
    搜索 Python 文档。

    Args:
        topic: 搜索主题

    Returns:
        str: 文档摘要
    """
    docs = {
        "装饰器": "装饰器是一种修改函数行为的方式，使用 @decorator 语法。",
        "生成器": "生成器是一种特殊的迭代器，使用 yield 关键字。",
        "异常处理": "使用 try/except/finally 处理异常。",
    }
    return docs.get(topic, f"未找到关于 {topic} 的文档。")


tools = [calculator, get_current_time, search_python_docs]

# 创建 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
    ("human", "{input}"),
    ("agent_scratchpad", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

# 运行 Agent（自动追踪完整调用链）
questions = [
    "计算 (15 + 25) * 3 的结果是多少？",
    "现在几点了？",
    "Python中的装饰器是什么？",
    "先计算 100 / 4，然后告诉我当前时间",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"问题: {q}")
    print(f"{'='*60}")
    result = agent_executor.invoke({"input": q})
    print(f"回答: {result['output']}")

print("\n✅ Agent 运行过程已完整追踪到 LangSmith！")
print("可以在 LangSmith 中查看每一步的思考、工具调用、耗时、token")
```

### 在 LangSmith 中查看 Agent 追踪

Agent 追踪会显示完整的调用链：

```
AgentExecutor (总耗时: 3.2s, tokens: 1500)
├── LLM Call (1.2s, tokens: 800)
│   ├── Input: 系统提示 + 用户问题
│   └── Output: 决定调用 calculator 工具
├── Tool: calculator (0.01s)
│   ├── Input: expression="(15 + 25) * 3"
│   └── Output: 计算结果: 120
├── LLM Call (1.5s, tokens: 700)
│   ├── Input: 工具结果 + 历史
│   └── Output: 最终回答
└── Final Output: 计算结果是 120
```

---

## 数据集（Datasets）

### 创建数据集

创建 `dataset_demo.py`：

```python
"""
dataset_demo.py - LangSmith 数据集创建和管理
用于 RAG/Agent 的回归测试
"""
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

# 数据集名称
dataset_name = "python-basics-qa"

# 检查数据集是否存在，存在则删除
try:
    existing = client.read_dataset(dataset_name=dataset_name)
    client.delete_dataset(dataset_id=existing.id)
    print(f"🗑️  删除已有数据集: {dataset_name}")
except Exception:
    pass

# 创建数据集
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Python 基础知识问答数据集，用于 RAG 效果评估",
)
print(f"✅ 创建数据集: {dataset_name}")

# 准备数据（问题 + 期望答案）
examples = [
    {
        "input": "Python中的列表和元组有什么区别？",
        "expected_output": "列表是可变的，用[]表示；元组是不可变的，用()表示。列表可以增删改，元组创建后不能修改。",
    },
    {
        "input": "什么是装饰器？请举一个简单例子。",
        "expected_output": "装饰器是一种修改函数行为的方式，使用@decorator语法。例子：@timer可以计算函数执行时间。",
    },
    {
        "input": "如何处理Python中的异常？",
        "expected_output": "使用try/except/finally处理异常。try块放可能出错的代码，except块捕获异常，finally块无论是否异常都执行。",
    },
    {
        "input": "什么是生成器？和迭代器有什么区别？",
        "expected_output": "生成器是一种特殊的迭代器，使用yield关键字。迭代器是实现了__iter__和__next__方法的对象，生成器更简洁。",
    },
    {
        "input": "Python中的GIL是什么？",
        "expected_output": "GIL（全局解释器锁）是Python的机制，同一时刻只允许一个线程执行Python字节码，导致多线程无法真正并行执行CPU密集型任务。",
    },
]

# 添加数据到数据集
for i, example in enumerate(examples, 1):
    client.create_example(
        inputs={"question": example["input"]},
        outputs={"answer": example["expected_output"]},
        dataset_id=dataset.id,
    )
    print(f"  添加示例 {i}: {example['input'][:30]}...")

print(f"\n✅ 数据集创建完成，共 {len(examples)} 条示例")
print(f"访问: https://smith.langchain.com/datasets 查看数据集")
```

### 运行并查看数据集

```bash
python dataset_demo.py
```

---

## 评估（Evaluation）

### RAG 效果评估

创建 `evaluation_demo.py`：

```python
"""
evaluation_demo.py - LangSmith 评估示例
用评估器自动打分，量化 RAG/Agent 效果
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.evaluation import LangChainStringEvaluator

load_dotenv()

client = Client()

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 创建待评估的 Chain（模拟 RAG 系统）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个Python编程专家，用简洁明了的方式回答问题。"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()


# 定义目标函数（待评估的系统）
def predict_answer(inputs: dict) -> dict:
    """
    待评估的预测函数

    Args:
        inputs: 包含 question 的字典

    Returns:
        dict: 包含 output 的字典
    """
    question = inputs["question"]
    answer = chain.invoke({"question": question})
    return {"output": answer}


# 定义评估器
# 1. 正确性评估（LLM 作为评委）
correctness_evaluator = LangChainStringEvaluator(
    "correctness",
    config={
        "llm": llm,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    },
)

# 2. 相关性评估
relevance_evaluator = LangChainStringEvaluator(
    "relevance",
    config={
        "llm": llm,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "input": example.inputs["question"],
    },
)

# 3. 简洁性评估
conciseness_evaluator = LangChainStringEvaluator(
    "conciseness",
    config={
        "llm": llm,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "input": example.inputs["question"],
    },
)

# 运行评估
print("🚀 开始评估...")
print(f"数据集: python-basics-qa")
print(f"评估器: correctness, relevance, conciseness")
print()

experiment_results = evaluate(
    predict_answer,
    data="python-basics-qa",
    evaluators=[
        correctness_evaluator,
        relevance_evaluator,
        conciseness_evaluator,
    ],
    experiment_prefix="python-qa-v1",
    max_concurrency=2,
)

print("\n✅ 评估完成！")
print(f"实验名称: {experiment_results.experiment_name}")
print(f"访问: https://smith.langchain.com/experiments 查看评估结果")
```

### 评估指标说明

| 评估器 | 说明 | 分数范围 |
|--------|------|----------|
| **correctness** | 回答是否正确，与参考答案对比 | 0-1 |
| **relevance** | 回答是否与问题相关 | 0-1 |
| **conciseness** | 回答是否简洁，没有冗余 | 0-1 |
| **helpfulness** | 回答是否有帮助 | 0-1 |
| **depth** | 回答是否有深度 | 0-1 |
| **creativity** | 回答是否有创意 | 0-1 |
| **detail** | 回答是否详细 | 0-1 |

### 自定义评估器

```python
"""
custom_evaluator.py - 自定义评估器示例
根据业务需求定义特定的评估标准
"""
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Example, Run


def check_python_code_format(run: Run, example: Example) -> dict:
    """
    自定义评估器：检查回答中是否包含正确的 Python 代码格式

    Args:
        run: 运行结果
        example: 示例数据

    Returns:
        dict: 评估结果
    """
    prediction = run.outputs.get("output", "")

    # 检查是否包含代码块
    has_code_block = "```python" in prediction or "```" in prediction

    # 检查是否包含示例代码
    has_example = any(keyword in prediction for keyword in ["def ", "import ", "class ", "print("])

    # 计算分数
    score = 0
    if has_code_block:
        score += 0.5
    if has_example:
        score += 0.5

    return {
        "key": "python_code_quality",
        "score": score,
        "comment": f"包含代码块: {has_code_block}, 包含示例: {has_example}",
    }


# 使用自定义评估器
experiment_results = evaluate(
    predict_answer,
    data="python-basics-qa",
    evaluators=[
        correctness_evaluator,
        check_python_code_format,  # 自定义评估器
    ],
    experiment_prefix="python-qa-v2",
)
```

---

## 实验对比（A/B 测试）

### 对比不同 Prompt 版本

```python
"""
ab_test_demo.py - A/B 测试示例
对比不同版本的效果，选择最优方案
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate
from langsmith.evaluation import LangChainStringEvaluator

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 版本1：简单 Prompt
prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个Python编程专家。"),
    ("human", "{question}")
])
chain_v1 = prompt_v1 | llm | StrOutputParser()

# 版本2：详细 Prompt（要求举例、简洁）
prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """你是一个Python编程专家。
回答要求：
1. 先给出核心概念的简洁定义
2. 然后举一个简单易懂的代码示例
3. 最后说明关键注意点
4. 总字数控制在200字以内"""),
    ("human", "{question}")
])
chain_v2 = prompt_v2 | llm | StrOutputParser()


def predict_v1(inputs):
    return {"output": chain_v1.invoke({"question": inputs["question"]})}


def predict_v2(inputs):
    return {"output": chain_v2.invoke({"question": inputs["question"]})}


# 评估器
correctness_evaluator = LangChainStringEvaluator(
    "correctness",
    config={"llm": llm},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    },
)

# 运行版本1
print("🚀 运行版本1（简单Prompt）...")
results_v1 = evaluate(
    predict_v1,
    data="python-basics-qa",
    evaluators=[correctness_evaluator],
    experiment_prefix="v1-simple-prompt",
)

# 运行版本2
print("\n🚀 运行版本2（详细Prompt）...")
results_v2 = evaluate(
    predict_v2,
    data="python-basics-qa",
    evaluators=[correctness_evaluator],
    experiment_prefix="v2-detailed-prompt",
)

print("\n✅ A/B 测试完成！")
print(f"版本1实验: {results_v1.experiment_name}")
print(f"版本2实验: {results_v2.experiment_name}")
print("在 LangSmith 中对比两个实验的评估分数，选择最优版本")
```

---

## 线上监控

### 生产环境追踪配置

```python
"""
production_monitoring.py - 生产环境监控配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 生产环境配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "production-agent"  # 生产环境项目

# 采样率配置（生产环境建议采样，避免成本过高）
# 100% 采样：所有调用都追踪
# 10% 采样：只追踪 10% 的调用
os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "0.1"  # 10% 采样

# 标签配置（用于分类和过滤）
os.environ["LANGCHAIN_TAGS"] = "production,v1.2.0,fastapi"

# 元数据配置（附加自定义信息）
metadata = {
    "environment": "production",
    "version": "1.2.0",
    "deployment": "k8s",
}
```

### 监控指标

在 LangSmith 监控面板可以查看：

| 指标 | 说明 |
|------|------|
| **请求量** | 单位时间内的调用次数 |
| **延迟（Latency）** | P50/P95/P99 延迟 |
| **Token 消耗** | 总 tokens、平均 tokens、成本估算 |
| **错误率** | 失败调用占比 |
| **用户反馈** | 👍/👎 反馈统计 |
| **评估分数** | 线上评估的平均分 |

---

## 学习要点

1. **LangSmith** 是 LangChain 官方的全链路观测和评估平台，解决 Agent/RAG 的调试和量化问题
2. **追踪（Tracing）** 自动记录 LLM 调用、Chain 执行、Agent 运行过程，包括输入输出、耗时、token
3. **数据集（Datasets）** 创建问题和答案的测试集，用于回归测试和效果评估
4. **评估（Evaluation）** 用 LLM 作为评委自动打分，支持 correctness、relevance、conciseness 等多种评估器
5. **实验（Experiments）** 对比不同版本的效果，A/B 测试选择最优方案
6. **自定义评估器** 可以根据业务需求定义特定的评估标准
7. **生产环境监控** 建议使用采样率（如 10%）控制成本，同时配置标签和元数据便于过滤
8. **没有量化就没法优化**，LangSmith 的量化评估是优化 Agent/RAG 的基础
9. **LangSmith 可以作为简历亮点**，也是面试必问的点（如何评估 RAG 效果）
10. **最佳实践**：先建数据集 → 跑基线评估 → 优化后再评估 → 对比实验选择最优

## 扩展方向

- 学习 LangSmith 的 Prompt Hub（提示词管理和版本控制）
- 探索 LangSmith 的 Annotation Queue（人工标注队列）
- 学习在线评估（Online Evaluation）和离线评估（Offline Evaluation）的区别
- 探索 LangSmith 的 Playground（交互式调试环境）
- 学习如何集成 LangSmith 到 CI/CD 流水线（自动回归测试）
- 探索 LangSmith 的成本分析和预算控制
- 学习如何用 LangSmith 进行根因分析（Root Cause Analysis）
- 探索 LangSmith 的团队协作功能（共享数据集、实验、注释）
- 学习 LangSmith 的 API 用法（程序化管理数据集和评估）
- 探索替代方案（Langfuse、Phoenix、Helicone、Braintrust）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/31-langsmith-observability

包含本文的完整可运行代码示例（追踪 + Agent 追踪 + 数据集 + 评估 + 自定义评估器 + A/B 测试 + 生产监控）。

---

**上一篇**：[Neo4j 知识图谱和 Graph RAG](./30_Neo4j知识图谱和Graph-RAG.md) | **下一篇**：[DeepAgents](./32_DeepAgents.md)
