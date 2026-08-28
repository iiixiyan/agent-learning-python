# AGUI 协议：Vercel AI SDK + LangChain 实现流式组件渲染

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈
> 前置知识：[Tool 调用](../01-第一阶段-Agent基础入门/05_Tool调用.md)、[SSE 流式接口](./20_FastAPI+LangChain实现SSE流式接口.md)

---

## 什么是 AGUI？

AGUI（AI Generated UI）是一种让大模型直接生成 UI 组件的技术。

| 方式 | 说明 | 示例 |
|------|------|------|
| **传统方式** | 大模型返回文本，前端固定渲染 | 纯文字回答 |
| **AGUI 方式** | 大模型返回结构化的组件描述，前端动态渲染对应组件 | 卡片、按钮、图表、表单 |

### 应用场景

- **智能客服**：返回卡片、按钮、表单（如订单查询卡片、退款按钮）
- **数据分析**：返回图表组件（如柱状图、折线图、饼图）
- **代码助手**：返回代码块、运行结果、文件树
- **购物助手**：返回商品卡片、规格选择器、购买按钮

## 整体架构

```
┌──────────────┐    SSE 流式数据     ┌──────────────┐
│   前端页面    │ ←────────────────── │  FastAPI 后端 │
│ (动态渲染)    │ ──────────────────→ │  (LangChain)  │
└──────┬───────┘    用户问题          └──────┬───────┘
       │                                      │
       │  解析 SSE 数据                       │  大模型思考
       │  ┌─────────────────────────────┐    │  ↓
       │  │ type: text    → 渲染文字    │    │  Tool Calling
       │  │ type: component → 渲染组件  │    │  ↓
       │  └─────────────────────────────┘    │  执行 Tool
       │                                      │  返回组件数据
       └──────────────────────────────────────┘
```

### SSE 数据协议

| 类型 | 说明 | 数据格式 |
|------|------|----------|
| `text` | 文本内容 | `{"type": "text", "content": "文字"}` |
| `tool_start` | 工具开始调用 | `{"type": "tool_start", "name": "render_card"}` |
| `component` | 组件数据 | `{"type": "component", "component": {...}}` |
| `[DONE]` | 流结束 | `[DONE]` |

---

## 技术方案

| 方案 | 说明 | 适用场景 |
|------|------|----------|
| **Vercel AI SDK** | 前端 SDK，支持流式渲染 UI 组件 | React/Next.js 项目 |
| **LangChain Tool Calling** | 后端用工具调用返回组件数据 | 任意前端框架 |
| **自定义协议** | 约定 JSON 格式，前端解析渲染 | 完全自定义 |

本文采用 **LangChain Tool Calling + 自定义 SSE 协议** 的方案，前后端解耦，适合任意前端框架。

---

## Python 后端实现（FastAPI + LangChain）

### 安装依赖

```bash
pip install fastapi uvicorn langchain langchain-openai python-dotenv pydantic
```

### 完整代码

```python
"""
agui_backend.py - AGUI 流式组件渲染后端
"""
import os
import json
import asyncio
from typing import Literal, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

app = FastAPI(title="AGUI 流式组件渲染服务", version="1.0.0")

# 初始化大模型（开启流式）
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    streaming=True,
)


# ============ 定义 UI 组件工具 ============

@tool
def render_card(
    title: str = Field(description="卡片标题"),
    description: str = Field(description="卡片描述"),
    image_url: str = Field("", description="卡片图片 URL（可选）"),
) -> Dict:
    """
    渲染一个卡片组件。用于展示信息摘要，如商品信息、用户资料、文章摘要等。
    """
    return {
        "type": "card",
        "title": title,
        "description": description,
        "image": image_url,
    }


@tool
def render_button(
    text: str = Field(description="按钮文字"),
    action: str = Field(description="按钮点击动作（JS 函数名或 URL）"),
) -> Dict:
    """
    渲染一个按钮组件。用于触发操作，如查看详情、提交表单、跳转链接等。
    """
    return {
        "type": "button",
        "text": text,
        "action": action,
    }


@tool
def render_chart(
    chart_type: Literal["bar", "line", "pie"] = Field(description="图表类型：bar(柱状图)/line(折线图)/pie(饼图)"),
    data: List[Dict] = Field(description="图表数据，格式如 [{'label': 'A', 'value': 10}]"),
    title: str = Field(description="图表标题"),
) -> Dict:
    """
    渲染一个图表组件。用于数据可视化，如销售统计、趋势分析、占比展示等。
    """
    return {
        "type": "chart",
        "chart_type": chart_type,
        "data": data,
        "title": title,
    }


@tool
def render_form(
    fields: List[Dict] = Field(description="表单字段，格式如 [{'name': 'email', 'label': '邮箱', 'type': 'email'}]"),
    submit_text: str = Field("提交", description="提交按钮文字"),
) -> Dict:
    """
    渲染一个表单组件。用于收集用户输入，如注册表单、反馈表单、搜索表单等。
    """
    return {
        "type": "form",
        "fields": fields,
        "submit_text": submit_text,
    }


# 工具列表和映射
tools = [render_card, render_button, render_chart, render_form]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}


# ============ 系统提示词 ============

SYSTEM_PROMPT = """你是一个智能助手，可以渲染 UI 组件来丰富回答。

## 可用组件
- render_card: 渲染卡片（标题、描述、图片）
- render_button: 渲染按钮（文字、点击动作）
- render_chart: 渲染图表（柱状图/折线图/饼图）
- render_form: 渲染表单（输入字段、提交按钮）

## 使用规则
1. 根据用户需求，选择合适的工具渲染组件
2. 可以同时渲染多个组件
3. 先用文字说明，再渲染组件
4. 用中文回答
5. 组件数据要真实合理，图表数据要有意义"""


# ============ AGUI 流式生成函数 ============

async def agui_stream(user_input: str):
    """
    AGUI 流式生成：先流式输出文本，再执行工具返回组件

    Args:
        user_input: 用户输入

    Yields:
        str: SSE 格式的数据
    """
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(user_input),
    ]

    # 第一轮：流式输出文本（同时收集 tool_calls）
    full_ai_message = None
    async for chunk in llm_with_tools.astream(messages):
        # 流式输出文本内容
        if chunk.content:
            yield f"data: {json.dumps({'type': 'text', 'content': chunk.content}, ensure_ascii=False)}\n\n"

        # 收集完整的 AI Message（用于后续执行工具）
        full_ai_message = full_ai_message + chunk if full_ai_message else chunk

    # 将 AI Message 加入消息历史
    if full_ai_message:
        messages.append(full_ai_message)

    # 第二轮：执行工具并返回组件数据
    if full_ai_message and full_ai_message.tool_calls:
        for tool_call in full_ai_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if tool_name in tool_map:
                # 通知前端工具开始调用
                yield f"data: {json.dumps({'type': 'tool_start', 'name': tool_name}, ensure_ascii=False)}\n\n"

                # 执行工具
                tool = tool_map[tool_name]
                result = tool.invoke(tool_args)

                # 返回组件数据
                yield f"data: {json.dumps({'type': 'component', 'component': result}, ensure_ascii=False)}\n\n"

    # 流结束
    yield "data: [DONE]\n\n"


# ============ API 接口 ============

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(description="用户消息")


@app.post("/agui/chat", summary="AGUI 流式对话接口")
async def chat(body: ChatRequest):
    """
    AGUI 流式对话接口：返回 SSE 流，包含文本和组件数据

    - **message**: 用户消息
    """
    return StreamingResponse(
        agui_stream(body.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/", summary="健康检查")
async def root():
    return {"message": "AGUI 流式组件渲染服务运行中"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 前端渲染示例

### 简单版（原生 JavaScript）

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AGUI 流式组件渲染测试</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
        .input-group { display: flex; gap: 10px; margin: 20px 0; }
        input { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
        button { padding: 10px 20px; border-radius: 8px; border: none; background: #3b82f6; color: white; cursor: pointer; }
        #output { margin: 20px 0; padding: 16px; background: #f9fafb; border-radius: 12px; min-height: 200px; }
        .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 12px 0; background: white; }
        .card h3 { margin: 0 0 8px; }
        .card img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
        .btn { display: inline-block; padding: 8px 16px; background: #3b82f6; color: white; border-radius: 8px; text-decoration: none; margin: 8px 4px; cursor: pointer; border: none; }
        .chart { margin: 12px 0; padding: 16px; background: white; border-radius: 12px; border: 1px solid #e5e7eb; }
        .chart-bar { display: flex; align-items: flex-end; gap: 8px; height: 150px; margin-top: 12px; }
        .chart-bar-item { flex: 1; background: #3b82f6; border-radius: 4px 4px 0 0; position: relative; min-height: 4px; }
        .chart-bar-item span { position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-size: 12px; }
        .form { margin: 12px 0; padding: 16px; background: white; border-radius: 12px; border: 1px solid #e5e7eb; }
        .form-field { margin: 12px 0; }
        .form-field label { display: block; margin-bottom: 4px; font-weight: 500; }
        .form-field input, .form-field select, .form-field textarea { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>🎨 AGUI 流式组件渲染</h1>

    <div class="input-group">
        <input type="text" id="message" placeholder="输入问题，如：展示Python学习路线" value="展示Python学习路线">
        <button onclick="sendMessage()">发送</button>
    </div>

    <div id="output"></div>

    <script>
        const output = document.getElementById('output');

        async function sendMessage() {
            const message = document.getElementById('message').value;
            if (!message) return;

            output.innerHTML = '';

            // 使用 fetch + ReadableStream 处理 SSE（POST 请求）
            const response = await fetch('/agui/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;

                        try {
                            const parsed = JSON.parse(data);
                            handleData(parsed);
                        } catch (e) {
                            console.error('解析失败:', e);
                        }
                    }
                }
            }
        }

        function handleData(data) {
            if (data.type === 'text') {
                // 文本内容：追加到最后一个文本节点
                let textNode = output.querySelector('.text-content:last-child');
                if (!textNode || textNode.nextElementSibling) {
                    textNode = document.createElement('div');
                    textNode.className = 'text-content';
                    output.appendChild(textNode);
                }
                textNode.textContent += data.content;
            } else if (data.type === 'component') {
                // 组件数据：渲染对应组件
                renderComponent(data.component);
            }
        }

        function renderComponent(comp) {
            const container = document.createElement('div');

            if (comp.type === 'card') {
                container.className = 'card';
                container.innerHTML = `
                    <h3>${comp.title}</h3>
                    <p>${comp.description}</p>
                    ${comp.image ? `<img src="${comp.image}" alt="${comp.title}">` : ''}
                `;
            } else if (comp.type === 'button') {
                container.innerHTML = `<button class="btn" onclick="${comp.action}">${comp.text}</button>`;
            } else if (comp.type === 'chart') {
                container.className = 'chart';
                let barsHtml = '';
                const maxValue = Math.max(...comp.data.map(d => d.value));
                for (const item of comp.data) {
                    const height = (item.value / maxValue) * 100;
                    barsHtml += `<div class="chart-bar-item" style="height: ${height}%"><span>${item.label}</span></div>`;
                }
                container.innerHTML = `<h3>${comp.title}</h3><div class="chart-bar">${barsHtml}</div>`;
            } else if (comp.type === 'form') {
                container.className = 'form';
                let fieldsHtml = '';
                for (const field of comp.fields) {
                    const inputType = field.type || 'text';
                    fieldsHtml += `
                        <div class="form-field">
                            <label>${field.label}</label>
                            <input type="${inputType}" name="${field.name}" placeholder="${field.placeholder || ''}">
                        </div>
                    `;
                }
                container.innerHTML = `${fieldsHtml}<button class="btn">${comp.submit_text}</button>`;
            }

            output.appendChild(container);
        }
    </script>
</body>
</html>
```

### 运行

```bash
uvicorn agui_backend:app --reload
```

访问前端页面，输入问题，即可看到流式文本和动态渲染的组件。

---

## 组件协议参考

### Card 卡片

```json
{
  "type": "card",
  "title": "卡片标题",
  "description": "卡片描述",
  "image": "https://example.com/image.jpg"
}
```

### Button 按钮

```json
{
  "type": "button",
  "text": "按钮文字",
  "action": "handleClick()"
}
```

### Chart 图表

```json
{
  "type": "chart",
  "chart_type": "bar",
  "title": "图表标题",
  "data": [
    {"label": "A", "value": 10},
    {"label": "B", "value": 20},
    {"label": "C", "value": 15}
  ]
}
```

### Form 表单

```json
{
  "type": "form",
  "fields": [
    {"name": "email", "label": "邮箱", "type": "email", "placeholder": "请输入邮箱"},
    {"name": "name", "label": "姓名", "type": "text", "placeholder": "请输入姓名"}
  ],
  "submit_text": "提交"
}
```

---

## 学习要点

1. **AGUI 的核心**是大模型返回结构化组件数据，前端动态渲染，而不是固定的文本输出
2. **Tool Calling 实现组件渲染**是最简单的方案：每个组件对应一个 Tool，大模型调用 Tool 返回组件数据
3. **SSE 流式传输**让文本和组件可以逐步展示，提升用户体验
4. **两轮调用模式**：第一轮流式输出文本并收集 tool_calls，第二轮执行工具返回组件数据
5. **组件协议需要前后端约定**：组件类型、字段格式、渲染方式都要统一
6. **FastAPI StreamingResponse** 实现 SSE 流，`media_type="text/event-stream"`
7. **前端用 fetch + ReadableStream** 处理 POST 请求的 SSE 流（EventSource 只支持 GET）
8. **组件渲染函数**根据 `type` 字段分发到不同的渲染逻辑

## 扩展方向

- 集成 Vercel AI SDK，实现更完善的流式组件渲染（React/Next.js）
- 添加更多组件类型：表格、时间线、代码块、文件树、地图等
- 实现组件交互：按钮点击、表单提交、图表钻取等
- 添加组件状态：加载中、错误、空状态
- 实现组件嵌套：卡片里包含按钮、图表里包含表格等
- 添加主题定制：深色模式、品牌色、字体等
- 实现组件缓存：相同组件数据复用渲染结果
- 探索 MCP 协议集成：通过 MCP Server 提供组件渲染能力

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/02-enterprise-backend/24-agui-protocol

包含本文的完整可运行代码示例（FastAPI + LangChain AGUI 流式组件渲染 + 前端渲染页面）。

---

**上一篇**：[给 Agent 加上语音交互](./23_给Agent加上语音交互.md) | **下一篇**：[图编排引擎 - LangGraph 和多 Agent 架构](./25_图编排引擎-LangGraph和多Agent架构.md)
