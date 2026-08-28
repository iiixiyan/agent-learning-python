# DeepAgents 实战：多 Agent 架构的深度调研助手

> **Python 版** | 基于 LangGraph + DeepAgents Middleware 技术栈
> 前置知识：[DeepAgents 基础](./32_DeepAgents.md)、[LangGraph 图编排和多 Agent](../02-第二阶段-企业级后端与进阶框架/25_图编排引擎-LangGraph和多Agent架构.md)

---

## 项目目标

做一个**深度调研助手**：用户输入一个主题，Agent 自动规划研究方向、搜索资料、整理报告。

### 多 Agent 协作架构

```
用户输入主题
     │
     ▼
┌─────────────┐
│  Planner    │  规划者：拆解研究任务，制定3-5个研究方向
│  (规划者)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Researcher  │  研究员：按研究方向搜索资料，提取关键信息
│  (研究员)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Writer    │  写作者：整理资料，生成结构化调研报告
│  (写作者)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reviewer   │  审核者：审核报告质量，提出修改意见
│  (审核者)    │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
 通过    不通过 → 返回 Writer 修改（最多2次）
   │
   ▼
 最终报告
```

### 四个 Agent 的职责

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **Planner（规划者）** | 拆解研究任务，制定研究计划 | 主题 | 3-5个研究方向 |
| **Researcher（研究员）** | 搜索资料，提取信息 | 研究计划 | 各方向的研究发现 |
| **Writer（写作者）** | 整理资料，生成报告 | 研究发现 | 结构化调研报告 |
| **Reviewer（审核者）** | 审核质量，提出意见 | 报告 | 通过/修改意见 |

---

## 技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| **LangGraph** | 图编排 | 管理多 Agent 状态流转，支持条件分支和循环 |
| **DeepAgents Middleware** | 高级功能 | 提供 Skill、上下文压缩、长期记忆等中间件 |
| **Tavily Search** | 搜索工具 | 网络搜索（也可以用 SerpAPI、Bing API、DuckDuckGo） |
| **LangSmith** | 观测评估 | 追踪 Agent 运行过程，评估效果 |

---

## 核心实现

### 安装依赖

```bash
pip install langgraph langchain langchain-openai python-dotenv tavily-python
```

### 完整代码

创建 `research_agent.py`：

```python
"""
research_agent.py - 多 Agent 深度调研助手
Planner → Researcher → Writer → Reviewer（审核-修改循环）
"""
import os
import json
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# ============ 1. 状态定义 ============

class ResearchState(TypedDict):
    """调研状态：在多个 Agent 之间传递的数据"""
    topic: str                    # 调研主题
    research_plan: List[str]     # 研究计划（3-5个方向）
    findings: List[str]           # 研究发现（各方向的资料）
    report: str                    # 生成的报告
    review_feedback: str          # 审核反馈
    iteration: int                 # 当前迭代次数（用于限制修改次数）


# ============ 2. 工具定义 ============

@tool
def web_search(query: str) -> str:
    """
    搜索网络信息。

    Args:
        query: 搜索关键词

    Returns:
        str: 搜索结果摘要
    """
    # 实际项目用 Tavily/SerpAPI/Bing API
    # 这里用模拟数据演示
    mock_results = {
        "AI Agent": "AI Agent 是能够自主感知环境、做出决策并执行动作的智能体。2024年 Agent 技术快速发展，出现了 AutoGPT、BabyAGI、Devin 等代表性项目。",
        "发展趋势": "2024年 AI Agent 发展趋势包括：多 Agent 协作、工具使用能力增强、长上下文处理、Agent 安全与对齐、垂直领域 Agent 应用。",
        "应用场景": "AI Agent 应用场景包括：客服助手、代码生成、数据分析、内容创作、流程自动化、个人助理、研究助手等。",
        "技术架构": "主流 Agent 架构包括：ReAct（推理+行动）、Plan-and-Execute（规划执行）、Reflexion（反思）、Multi-Agent（多 Agent 协作）。",
    }

    # 简单匹配关键词
    for key, value in mock_results.items():
        if key in query:
            return f"关于'{query}'的搜索结果：\n{value}"

    return f"关于'{query}'的搜索结果：\n这是一个模拟的搜索结果，实际项目中会调用真实的搜索 API。"


@tool
def read_article(url: str) -> str:
    """
    读取网页内容。

    Args:
        url: 网页 URL

    Returns:
        str: 网页内容摘要
    """
    return f"网页 {url} 的内容：\n这是一个模拟的网页内容，实际项目中会用 requests/BeautifulSoup 抓取。"


tools = [web_search, read_article]


# ============ 3. 初始化大模型 ============

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


# ============ 4. Agent 节点实现 ============

def planner_node(state: ResearchState) -> dict:
    """
    Planner（规划者）：拆解研究任务，制定研究计划

    输入：主题
    输出：研究计划（3-5个研究方向）、清空研究发现
    """
    print(f"\n{'='*60}")
    print(f"[Planner] 正在为主题制定研究计划: {state['topic']}")
    print(f"{'='*60}")

    prompt = f"""你是研究规划专家。请为以下主题制定研究计划，列出3-5个研究方向。
每个研究方向应该是一个简短的关键词或短语，便于后续搜索。

主题：{state['topic']}

请用 JSON 数组格式返回，例如：
["背景介绍", "技术原理", "应用场景", "发展趋势", "挑战与展望"]"""

    response = llm.invoke(prompt)

    # 解析 JSON
    try:
        # 尝试提取 JSON 数组
        content = response.content
        # 找到第一个 [ 和最后一个 ]
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end != -1:
            plan = json.loads(content[start:end])
        else:
            plan = [content]
    except Exception:
        plan = [response.content]

    print(f"[Planner] 研究计划已制定，共 {len(plan)} 个方向:")
    for i, direction in enumerate(plan, 1):
        print(f"  {i}. {direction}")

    return {"research_plan": plan, "findings": []}


def researcher_node(state: ResearchState) -> dict:
    """
    Researcher（研究员）：按研究方向搜索资料

    输入：研究计划
    输出：研究发现（各方向的资料）
    """
    print(f"\n{'='*60}")
    print(f"[Researcher] 正在搜索资料...")
    print(f"{'='*60}")

    findings = []
    for i, direction in enumerate(state["research_plan"], 1):
        query = f"{state['topic']} {direction}"
        print(f"  [{i}/{len(state['research_plan'])}] 搜索: {query}")

        # 调用搜索工具
        result = web_search.invoke({"query": query})
        findings.append(f"【{direction}】\n{result}")

    print(f"[Researcher] 资料搜索完成，共 {len(findings)} 条发现")

    return {"findings": findings}


def writer_node(state: ResearchState) -> dict:
    """
    Writer（写作者）：整理资料，生成报告

    输入：研究发现、审核反馈（如果有）
    输出：调研报告
    """
    print(f"\n{'='*60}")
    print(f"[Writer] 正在生成报告...")
    print(f"{'='*60}")

    # 拼接研究资料
    context = "\n\n".join(state["findings"])

    # 如果有审核反馈，加入修改要求
    feedback_instruction = ""
    if state.get("review_feedback"):
        feedback_instruction = f"\n\n请根据以下审核意见修改报告：\n{state['review_feedback']}"

    prompt = f"""根据以下研究资料，写一份关于'{state['topic']}'的深度调研报告。

要求：
1. 结构清晰，包含：摘要、引言、正文（分章节）、结论、参考文献
2. 正文至少3个章节，每个章节有明确的小标题
3. 字数1000字左右
4. 语言专业、客观、准确
5. 引用研究资料中的信息{feedback_instruction}

研究资料：
{context}"""

    response = llm.invoke(prompt)
    report = response.content

    print(f"[Writer] 报告生成完成，共 {len(report)} 字符")
    print(f"[Writer] 报告预览（前200字）:\n{report[:200]}...")

    return {"report": report}


def reviewer_node(state: ResearchState) -> dict:
    """
    Reviewer（审核者）：审核报告质量

    输入：报告、迭代次数
    输出：通过（空字典）或修改意见
    """
    print(f"\n{'='*60}")
    print(f"[Reviewer] 正在审核报告（第 {state['iteration'] + 1} 次审核）...")
    print(f"{'='*60}")

    # 最多修改2次，第3次直接通过
    if state["iteration"] >= 2:
        print("[Reviewer] 已达到最大修改次数，直接通过")
        return {}

    prompt = f"""请以专业编辑的身份审核以下调研报告，从以下几个维度评估：
1. 结构完整性：是否包含摘要、引言、正文、结论
2. 内容准确性：信息是否准确，有无明显错误
3. 逻辑连贯性：章节之间是否有逻辑联系
4. 语言专业性：语言是否专业、客观
5. 深度与广度：内容是否有足够的深度和广度

如果报告质量合格，请明确回复"通过"。
如果需要修改，请列出具体的修改意见和建议。

报告主题：{state['topic']}

报告内容：
{state['report']}"""

    response = llm.invoke(prompt)
    feedback = response.content

    # 判断是否通过
    if "通过" in feedback and len(feedback) < 200:
        print("[Reviewer] 审核通过！")
        return {}
    else:
        print(f"[Reviewer] 需要修改，反馈:\n{feedback[:300]}...")
        return {
            "review_feedback": feedback,
            "iteration": state["iteration"] + 1,
        }


# ============ 5. 构建图 ============

# 条件边函数：审核通过则结束，否则返回 Writer 修改
def should_rewrite(state: ResearchState) -> str:
    """
    判断是否需要重新写报告

    Returns:
        str: "writer"（需要修改）或 END（通过）
    """
    if state.get("review_feedback") and state["iteration"] < 2:
        print("\n[路由] 审核未通过，返回 Writer 修改")
        return "writer"
    else:
        print("\n[路由] 审核通过，结束流程")
        return END


# 创建图
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

# 设置入口
workflow.set_entry_point("planner")

# 添加边（线性流程）
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

# 添加条件边（审核-修改循环）
workflow.add_conditional_edges("reviewer", should_rewrite)

# 编译图
app = workflow.compile()


# ============ 6. 运行调研助手 ============

def run_research(topic: str) -> str:
    """
    运行深度调研助手

    Args:
        topic: 调研主题

    Returns:
        str: 最终调研报告
    """
    print(f"\n{'#'*60}")
    print(f"# 深度调研助手")
    print(f"# 主题: {topic}")
    print(f"{'#'*60}")

    result = app.invoke({
        "topic": topic,
        "iteration": 0,
    })

    print(f"\n{'#'*60}")
    print(f"# 调研完成！")
    print(f"# 迭代次数: {result['iteration']}")
    print(f"# 报告长度: {len(result['report'])} 字符")
    print(f"{'#'*60}\n")

    return result["report"]


# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例1：AI Agent 行业调研
    report = run_research("2024年AI Agent行业发展趋势")
    print("最终报告:\n")
    print(report)

    # 示例2：其他主题（取消注释运行）
    # report = run_research("Python Web框架对比分析")
    # report = run_research("大模型微调技术综述")
```

### 运行示例

```bash
# 1. 创建 .env 文件
echo "OPENAI_API_KEY=你的_api_key" > .env
echo "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1" >> .env
echo "MODEL_NAME=qwen-plus" >> .env

# 2. 运行调研助手
python research_agent.py
```

---

## DeepAgents Middleware 集成

在基础版本上，可以集成 DeepAgents 的 Middleware 增强功能：

### 1. 上下文压缩 Middleware

```python
"""
context_compression.py - 上下文压缩中间件集成
对话过长时自动摘要历史，控制 token 消耗
"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END


class CompressedState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str  # 历史摘要


def compress_context(state: CompressedState, llm, max_messages: int = 10) -> dict:
    """
    上下文压缩：消息超过阈值时自动摘要

    Args:
        state: 当前状态
        llm: 大模型
        max_messages: 最大消息数，超过则触发摘要

    Returns:
        dict: 更新后的状态
    """
    messages = state["messages"]

    if len(messages) <= max_messages:
        return {}

    # 需要摘要的消息（保留最后5条）
    to_summarize = messages[:-5]
    kept = messages[-5:]

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息，保持简洁。
对话：
{chr(10).join(f"{m.type}: {m.content[:200]}" for m in to_summarize)}
摘要："""
    summary = llm.invoke(summary_prompt).content

    # 构造新消息列表
    summary_msg = SystemMessage(content=f"历史对话摘要：\n{summary}")
    new_messages = [summary_msg] + list(kept)

    # 删除旧消息
    from langchain_core.messages import RemoveMessage
    delete_messages = [RemoveMessage(id=m.id) for m in messages]

    return {
        "messages": delete_messages + new_messages,
        "summary": summary,
    }
```

### 2. Skill Middleware 集成

```python
"""
skill_integration.py - Skill 中间件集成
让 Agent 支持可复用的能力包
"""
import os
from langchain_core.tools import tool


def load_skill(skill_name: str, skills_dir: str = ".agents/skills") -> str:
    """
    加载 Skill 的 SKILL.md 内容

    Args:
        skill_name: Skill 名称
        skills_dir: Skills 目录

    Returns:
        str: SKILL.md 内容
    """
    skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
    if not os.path.exists(skill_path):
        return f"Skill '{skill_name}' 未找到"
    with open(skill_path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def use_skill(skill_name: str, task: str) -> str:
    """
    使用指定的 Skill 完成任务。

    Args:
        skill_name: Skill 名称（如 web_research、data_analysis）
        task: 要完成的任务描述

    Returns:
        str: 任务结果
    """
    skill_content = load_skill(skill_name)
    return f"已加载 Skill: {skill_name}\n\n{skill_content}\n\n任务: {task}"


# 在 Researcher 中使用 Skill
def researcher_with_skill(state: dict, llm) -> dict:
    """
    带 Skill 支持的研究员

    可以使用 web_research Skill 进行更专业的网络调研
    """
    # 先加载 web_research Skill
    skill_content = load_skill("web_research")

    # 将 Skill 说明注入到系统提示中
    system_prompt = f"""你是专业研究员。
请参考以下 Skill 说明进行研究：
{skill_content}

研究主题：{state['topic']}
研究方向：{state['research_plan']}"""

    response = llm.invoke(system_prompt)
    return {"findings": [response.content]}
```

### 3. 长期记忆 Middleware

```python
"""
memory_integration.py - 长期记忆中间件集成
持久化存储用户偏好和项目信息
"""
import os
import json
from typing import Dict


class LongTermMemory:
    """长期记忆：基于文件的持久化存储"""

    def __init__(self, memory_dir: str = ".agents/memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    def save(self, key: str, value: Dict):
        """保存记忆"""
        path = os.path.join(self.memory_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def load(self, key: str) -> Dict:
        """加载记忆"""
        path = os.path.join(self.memory_dir, f"{key}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_keys(self) -> list:
        """列出所有记忆键"""
        return [f.replace(".json", "") for f in os.listdir(self.memory_dir)
                if f.endswith(".json")]


# 在调研助手中使用长期记忆
memory = LongTermMemory()

# 保存调研历史
def save_research_history(topic: str, report: str):
    """保存调研历史到长期记忆"""
    history = memory.load("research_history")
    if "topics" not in history:
        history["topics"] = []
    history["topics"].append({
        "topic": topic,
        "report_length": len(report),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    })
    memory.save("research_history", history)


# 加载用户偏好
def load_user_preferences() -> Dict:
    """加载用户偏好（报告风格、长度等）"""
    return memory.load("user_preferences")
```

---

## 运行流程图解

```
用户输入主题
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ Planner                                              │
│ 输入: topic                                          │
│ 处理: 调用 LLM 生成3-5个研究方向                     │
│ 输出: research_plan, findings=[]                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Researcher                                           │
│ 输入: topic, research_plan                           │
│ 处理: 对每个方向调用 web_search 工具                  │
│ 输出: findings (List[str])                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Writer                                               │
│ 输入: topic, findings, review_feedback(可选)         │
│ 处理: 调用 LLM 基于资料生成报告                       │
│ 输出: report (str)                                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Reviewer                                             │
│ 输入: report, iteration                              │
│ 处理: 调用 LLM 审核报告质量                           │
│ 输出: {} (通过) 或 {review_feedback, iteration+1}   │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
         iteration<2         iteration>=2
         且有反馈            或无反馈
              │                 │
              ▼                 ▼
         → Writer           → END (最终报告)
         (修改报告)
```

---

## 学习要点

1. **多 Agent 架构**适合复杂任务，每个 Agent 职责单一，便于维护和优化
2. **LangGraph** 用图结构管理状态流转，比链式调用更灵活，支持条件分支和循环
3. **状态定义**是多 Agent 协作的核心，明确每个 Agent 的输入输出字段
4. **审核-修改循环**提升输出质量，但要限制迭代次数（如最多2次）防止死循环
5. **Planner-Researcher-Writer-Reviewer** 是经典的内容生产多 Agent 模式
6. **工具调用**在 Researcher 节点使用，每个研究方向独立搜索，保证资料全面性
7. **DeepAgents Middleware** 可以增强基础版本：上下文压缩控制 token、Skill 复用能力、长期记忆持久化
8. **条件边**是 LangGraph 的核心特性，用函数判断下一个节点，实现动态路由
9. **可观测性**很重要，每个节点打印日志，便于调试和追踪 Agent 运行过程
10. **模拟数据**便于开发测试，实际项目中替换为真实的搜索 API（Tavily、SerpAPI）

## 扩展方向

- 集成真实搜索 API（Tavily、SerpAPI、Bing Search）替换模拟数据
- 添加网页抓取和内容提取能力（requests、BeautifulSoup、Playwright）
- 实现并行 Researcher（多个研究方向同时搜索，提升效率）
- 添加引用来源追踪（报告中标注信息来源 URL）
- 集成 LangSmith 进行全链路追踪和效果评估
- 实现报告格式导出（Markdown、PDF、HTML）
- 添加用户反馈收集和报告迭代优化
- 探索更多 Agent 角色（事实核查员、数据分析师、图表生成器）
- 实现多主题批量调研和对比分析
- 集成向量数据库，实现研究资料的语义检索和复用

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/33-deepagents-research-assistant

包含本文的完整可运行代码示例（多 Agent 深度调研助手 + 上下文压缩 + Skill 集成 + 长期记忆）。

---

**上一篇**：[DeepAgents 基础](./32_DeepAgents.md) | **下一篇**：[PostgreSQL](./34_PostgreSQL.md)
