# FastAPI + Tool 实现 OpenClaw 同款定时任务功能（下）

> **Python 版** | 基于 FastAPI + APScheduler + LangChain Python 技术栈
> 前置知识：[定时任务（上）](./21_FastAPI+tool实现定时任务(上).md)、FastAPI 基础

---

## 本文目标

上一篇实现了基础的 Tool 功能和 AI 接口。这篇继续完善定时任务的核心功能：

| 功能 | 说明 |
|------|------|
| **任务持久化** | 用 JSON 文件存储任务，重启不丢失 |
| **任务管理 API** | 增删改查定时任务 |
| **任务执行日志** | 记录每次执行的状态和结果 |
| **并发控制** | 防止同一任务重复执行 |
| **错误重试** | 任务失败后自动重试 |

## 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI 应用                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ 任务 API  │  │ APScheduler  │  │  任务执行器    │ │
│  │ (增删改查)│  │  (调度器)    │  │ (Agent Loop)  │ │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘ │
│       │                 │                    │         │
│       └────────┬────────┴────────────────────┘         │
│                ▼                                         │
│       ┌─────────────────┐    ┌─────────────────┐      │
│       │  tasks.json     │    │ task_logs.json  │      │
│       │  (任务持久化)    │    │  (执行日志)      │      │
│       └─────────────────┘    └─────────────────┘      │
└─────────────────────────────────────────────────────┘
```

---

## 安装依赖

```bash
pip install fastapi uvicorn apscheduler langchain langchain-openai pydantic python-dotenv
```

| 依赖包 | 用途 |
|--------|------|
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `apscheduler` | 定时任务调度器 |
| `langchain` | LLM 应用框架 |
| `langchain-openai` | OpenAI 兼容模型封装 |
| `pydantic` | 数据验证 |

---

## 完整实现

### 1. 数据模型

```python
"""
scheduler_app.py - Agent 定时任务服务
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Agent 定时任务服务", version="1.0.0")
scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


# ============ 数据模型 ============

class TaskCreate(BaseModel):
    """创建任务参数"""
    name: str = Field(description="任务名称")
    cron: str = Field("0 9 * * *", description="Cron 表达式，默认每天9点")
    prompt: str = Field(description="任务执行时传给大模型的提示词")
    tool_name: Optional[str] = Field(None, description="要调用的工具名称")
    tool_args: Optional[Dict] = Field(None, description="工具参数")
    enabled: bool = Field(True, description="是否启用")
    max_retries: int = Field(2, description="最大重试次数")


class TaskUpdate(BaseModel):
    """更新任务参数"""
    name: Optional[str] = None
    cron: Optional[str] = None
    prompt: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    enabled: Optional[bool] = None
    max_retries: Optional[int] = None


class TaskInfo(BaseModel):
    """任务信息"""
    id: str
    name: str
    cron: str
    prompt: str
    enabled: bool
    max_retries: int = 2
    next_run: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None
    retry_count: int = 0
```

### 2. 工具定义

```python
# ============ 工具定义 ============

@tool
def send_notification(message: str) -> str:
    """
    发送通知消息

    Args:
        message: 通知内容

    Returns:
        str: 发送结果
    """
    print(f"[通知] {message}")
    return f"通知已发送: {message}"


@tool
def generate_report(topic: str) -> str:
    """
    生成报告

    Args:
        topic: 报告主题

    Returns:
        str: 报告内容
    """
    result = llm.invoke(f"生成关于{topic}的简短报告")
    return result.content


# 工具名称到工具对象的映射
tools_map = {
    "send_notification": send_notification,
    "generate_report": generate_report,
}
```

### 3. 任务持久化

```python
# ============ 任务存储（JSON 文件持久化） ============

TASKS_FILE = "tasks.json"
LOGS_FILE = "task_logs.json"

# 并发锁，防止同时读写文件
_file_lock = asyncio.Lock()


def load_tasks() -> Dict:
    """从 JSON 文件加载任务"""
    try:
        with open(TASKS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_tasks(tasks: Dict):
    """保存任务到 JSON 文件"""
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def append_log(task_id: str, status: str, result: str):
    """
    追加任务执行日志

    Args:
        task_id: 任务 ID
        status: 执行状态（success/error）
        result: 执行结果
    """
    try:
        with open(LOGS_FILE, encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = {}

    if task_id not in logs:
        logs[task_id] = []

    logs[task_id].append({
        "time": datetime.now().isoformat(),
        "status": status,
        "result": result[:500],  # 只保留前500字符
    })

    # 只保留最近50条日志
    logs[task_id] = logs[task_id][-50:]

    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
```

### 4. 任务执行函数

```python
# ============ 任务执行函数 ============

# 正在执行的任务集合，用于并发控制
_running_tasks: set = set()


async def execute_task(task_id: str):
    """
    执行定时任务

    Args:
        task_id: 任务 ID
    """
    # 并发控制：如果任务正在执行，跳过本次
    if task_id in _running_tasks:
        print(f"[跳过] 任务 {task_id} 正在执行中")
        return

    _running_tasks.add(task_id)

    try:
        tasks = load_tasks()
        if task_id not in tasks:
            return

        task = tasks[task_id]
        print(f"[执行任务] {task['name']} (ID: {task_id})")

        try:
            # 1. 调用大模型生成内容
            result = await llm.ainvoke(task["prompt"])
            content = result.content

            # 2. 如果指定了工具，调用工具
            if task.get("tool_name") and task["tool_name"] in tools_map:
                tool = tools_map[task["tool_name"]]
                tool_args = task.get("tool_args", {})
                tool_result = await tool.ainvoke(tool_args)
                content += f"\n\n工具执行结果: {tool_result}"

            # 3. 更新任务状态
            tasks[task_id]["last_run"] = datetime.now().isoformat()
            tasks[task_id]["last_status"] = "success"
            tasks[task_id]["retry_count"] = 0
            save_tasks(tasks)
            append_log(task_id, "success", content)
            print(f"[任务完成] {task['name']}")

        except Exception as e:
            # 错误处理和重试
            retry_count = tasks[task_id].get("retry_count", 0)
            max_retries = task.get("max_retries", 2)

            if retry_count < max_retries:
                # 重试：延迟后重新执行
                tasks[task_id]["retry_count"] = retry_count + 1
                tasks[task_id]["last_status"] = f"retrying ({retry_count + 1}/{max_retries})"
                save_tasks(tasks)
                append_log(task_id, "retry", f"第 {retry_count + 1} 次重试: {str(e)}")

                # 延迟重试（指数退避）
                delay = 2 ** retry_count
                print(f"[任务重试] {task['name']}，{delay}秒后第 {retry_count + 1} 次重试")
                await asyncio.sleep(delay)
                await execute_task(task_id)
            else:
                # 达到最大重试次数，标记失败
                tasks[task_id]["last_run"] = datetime.now().isoformat()
                tasks[task_id]["last_status"] = f"error: {str(e)}"
                save_tasks(tasks)
                append_log(task_id, "error", str(e))
                print(f"[任务失败] {task['name']}: {e}")

    finally:
        # 释放并发锁
        _running_tasks.discard(task_id)
```

### 5. API 接口

```python
# ============ API 接口 ============

@app.post("/tasks", response_model=TaskInfo, summary="创建定时任务")
def create_task(task: TaskCreate):
    """
    创建定时任务

    - **name**: 任务名称
    - **cron**: Cron 表达式（如 `0 9 * * *` 表示每天9点）
    - **prompt**: 任务执行时传给大模型的提示词
    - **tool_name**: 要调用的工具名称（可选）
    - **tool_args**: 工具参数（可选）
    - **enabled**: 是否启用（默认 true）
    - **max_retries**: 最大重试次数（默认 2）
    """
    tasks = load_tasks()
    task_id = str(uuid.uuid4())[:8]

    tasks[task_id] = {
        "id": task_id,
        "name": task.name,
        "cron": task.cron,
        "prompt": task.prompt,
        "tool_name": task.tool_name,
        "tool_args": task.tool_args,
        "enabled": task.enabled,
        "max_retries": task.max_retries,
        "last_run": None,
        "last_status": None,
        "retry_count": 0,
    }

    save_tasks(tasks)

    # 如果启用，添加到调度器
    if task.enabled:
        scheduler.add_job(
            execute_task,
            CronTrigger.from_crontab(task.cron),
            args=[task_id],
            id=task_id,
            replace_existing=True,
            misfire_grace_time=300,  # 错过执行时间后5分钟内仍可执行
        )

    return tasks[task_id]


@app.get("/tasks", response_model=List[TaskInfo], summary="获取所有任务")
def list_tasks():
    """获取所有定时任务列表"""
    tasks = load_tasks()
    result = []

    for tid, task in tasks.items():
        job = scheduler.get_job(tid)
        task["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
        result.append(task)

    return result


@app.get("/tasks/{task_id}", response_model=TaskInfo, summary="获取任务详情")
def get_task(task_id: str):
    """根据 ID 获取任务详情"""
    tasks = load_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = scheduler.get_job(task_id)
    tasks[task_id]["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
    return tasks[task_id]


@app.put("/tasks/{task_id}", summary="更新任务")
def update_task(task_id: str, task: TaskUpdate):
    """更新定时任务配置"""
    tasks = load_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    update_data = task.dict(exclude_unset=True)
    tasks[task_id].update(update_data)
    save_tasks(tasks)

    # 重新调度
    if scheduler.get_job(task_id):
        scheduler.remove_job(task_id)

    if tasks[task_id]["enabled"]:
        scheduler.add_job(
            execute_task,
            CronTrigger.from_crontab(tasks[task_id]["cron"]),
            args=[task_id],
            id=task_id,
            replace_existing=True,
            misfire_grace_time=300,
        )

    return tasks[task_id]


@app.delete("/tasks/{task_id}", summary="删除任务")
def delete_task(task_id: str):
    """删除定时任务"""
    tasks = load_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    del tasks[task_id]
    save_tasks(tasks)

    if scheduler.get_job(task_id):
        scheduler.remove_job(task_id)

    return {"message": "删除成功"}


@app.post("/tasks/{task_id}/run", summary="立即执行任务")
async def run_task_now(task_id: str):
    """立即执行一次任务（不等待定时触发）"""
    tasks = load_tasks()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    await execute_task(task_id)
    return {"message": "执行完成"}


@app.get("/tasks/{task_id}/logs", summary="获取任务执行日志")
def get_task_logs(task_id: str, limit: int = 20):
    """
    获取任务执行日志

    - **task_id**: 任务 ID
    - **limit**: 返回最近 N 条日志（默认 20）
    """
    try:
        with open(LOGS_FILE, encoding="utf-8") as f:
            logs = json.load(f)
        return logs.get(task_id, [])[-limit:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []
```

### 6. 启动和恢复任务

```python
# ============ 启动 ============

@app.on_event("startup")
def start_scheduler():
    """应用启动时启动调度器并恢复已启用的任务"""
    scheduler.start()

    # 恢复已启用的任务
    tasks = load_tasks()
    restored_count = 0

    for tid, task in tasks.items():
        if task.get("enabled"):
            scheduler.add_job(
                execute_task,
                CronTrigger.from_crontab(task["cron"]),
                args=[tid],
                id=tid,
                replace_existing=True,
                misfire_grace_time=300,
            )
            restored_count += 1

    print(f"调度器已启动，恢复 {restored_count} 个任务")


@app.on_event("shutdown")
def shutdown_scheduler():
    """应用关闭时停止调度器"""
    scheduler.shutdown(wait=False)
    print("调度器已停止")


@app.get("/", summary="健康检查")
async def root():
    return {"message": "Agent 定时任务服务运行中", "tasks_count": len(load_tasks())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 使用示例

### 1. 启动服务

```bash
python scheduler_app.py
```

### 2. 创建定时任务

```bash
# 每天早上9点生成日报
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "每日日报",
    "cron": "0 9 * * *",
    "prompt": "生成今日AI行业新闻摘要",
    "tool_name": "send_notification",
    "tool_args": {"message": "日报已生成"},
    "enabled": true,
    "max_retries": 3
  }'
```

### 3. 查看所有任务

```bash
curl http://localhost:8000/tasks
```

### 4. 立即执行一次

```bash
curl -X POST http://localhost:8000/tasks/{task_id}/run
```

### 5. 查看执行日志

```bash
curl http://localhost:8000/tasks/{task_id}/logs?limit=10
```

### 6. 更新任务

```bash
curl -X PUT http://localhost:8000/tasks/{task_id} \
  -H "Content-Type: application/json" \
  -d '{"cron": "0 10 * * *", "enabled": false}'
```

### 7. 删除任务

```bash
curl -X DELETE http://localhost:8000/tasks/{task_id}
```

---

## Cron 表达式参考

| 表达式 | 说明 |
|--------|------|
| `0 9 * * *` | 每天早上9点 |
| `0 9 * * 1-5` | 工作日早上9点 |
| `0 0 1 * *` | 每月1号零点 |
| `*/30 * * * *` | 每30分钟 |
| `0 0 * * 0` | 每周日零点 |

Cron 表达式格式：`分 时 日 月 周`

---

## 学习要点

1. **APScheduler** 的 `AsyncIOScheduler` 配合 FastAPI 异步运行，支持异步任务执行
2. **CronTrigger.from_crontab()** 可以直接解析标准 cron 表达式，无需手动配置
3. **任务持久化**用 JSON 文件实现，简单易用；生产环境建议用数据库（SQLite/PostgreSQL）
4. **任务执行日志**记录每次执行的状态和结果，方便排查问题
5. **启动时恢复**已启用的任务，实现重启不丢失
6. **并发控制**用集合记录正在执行的任务，防止同一任务重复执行
7. **错误重试**用指数退避策略（2^n 秒延迟），达到最大重试次数后标记失败
8. **misfire_grace_time** 设置错过执行时间后的宽限期，避免任务因短暂停机而永久跳过
9. **FastAPI 自动文档**：访问 http://localhost:8000/docs 可以查看和测试所有 API

## 扩展方向

- 用数据库（SQLAlchemy + SQLite/PostgreSQL）替代 JSON 文件持久化
- 添加任务执行的 WebSocket 实时推送
- 实现任务依赖（一个任务完成后触发另一个任务）
- 添加任务执行的统计报表和可视化
- 实现心跳机制（定期主动执行任务，如健康检查）
- 集成 LangGraph 实现更复杂的 Agent 流程
- 添加任务模板（预设常用任务配置）
- 实现任务分组和标签管理

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/02-enterprise-backend/22-cron-job-tool-part2

包含本文的完整可运行代码示例（FastAPI + APScheduler 定时任务服务 + 任务持久化 + 执行日志 + 错误重试）。

---

**上一篇**：[FastAPI + Tool 实现定时任务（上）](./21_FastAPI+tool实现定时任务(上).md) | **下一篇**：[给 Agent 加上语音交互](./23_给Agent加上语音交互.md)
