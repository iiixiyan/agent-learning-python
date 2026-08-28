# Agent 全栈工程师转型之路 — Python 版

> 从零基础到企业级项目实战，全面掌握 AI Agent 全栈开发技能
> 技术栈：Python + FastAPI + LangChain + LangGraph + SQLAlchemy

---

## 项目简介

本项目是「Agent 全栈工程师转型之路：企业级知识库项目」系列文章的 **Python 版本**。通过 LangChain、LangGraph 等 Agent 框架学习 Agent 开发基础，并且学习 Milvus、ElasticSearch、PostgreSQL 等后端基础，然后实现企业级知识库等 Agent 项目，全面掌握 AI Agent 全栈开发技能。

- **文章总数**：56 篇（全部完成 ✅）
- **五个阶段**：Agent 基础入门 → 企业级后端 → 检索增强 → 存储监控 → 简历面试实战
- **图片资源**：435+ 张（33 个文章图片目录）
- **配套代码库**：每篇文章末尾均附有对应代码示例地址

## 技术栈映射

| 原技术栈（Node.js） | Python 版对应 |
|---------------------|---------------|
| Nest.js (TypeScript) | FastAPI (Python) |
| LangChain JS | LangChain Python |
| LangGraph JS | LangGraph Python |
| npm / yarn / pnpm | pip / poetry / uv |
| `package.json` | `pyproject.toml` / `requirements.txt` |
| `@nestjs/schedule` | APScheduler / Celery Beat |
| SSE (`@sse`) | FastAPI `StreamingResponse` |
| Prisma / TypeORM | SQLAlchemy / Tortoise-ORM |
| `@zilliz/milvus2-sdk-node` | `pymilvus` |
| `ioredis` | `redis-py` |
| `neo4j-driver` | `neo4j` (Python driver) |

## 文章目录

### 第一阶段：Agent 基础入门（第 1-19 篇）✅

| 序号 | 标题 | 状态 |
|------|------|------|
| 1 | 学员真实转型 Agent 成功经验 | ✅ 已完成 |
| 2 | 前言：Agent 课程脉络梳理 | ✅ 已完成 |
| 3 | AI Agent 开发要学什么？ | ✅ 已完成 |
| 4 | 从 Tool 开始：让大模型自动调工具 | ✅ 已完成 |
| 5 | 实现 mini cursor：自动调用 tool | ✅ 已完成 |
| 6 | MCP：可跨进程调用的 Tool | ✅ 已完成 |
| 7 | 高德 MCP + 浏览器 MCP | ✅ 已完成 |
| 8 | RAG：文档向量化与语义搜索 | ✅ 已完成 |
| 9 | 知识库的 loader 和 splitter | ✅ 已完成 |
| 10 | LangChain 全部 Splitter 解析 | ✅ 已完成 |
| 11 | 向量数据库 Milvus | ✅ 已完成 |
| 12 | Milvus + RAG 实战：电子书检索 | ✅ 已完成 |
| 13 | Memory 管理的三大策略 | ✅ 已完成 |
| 14 | 结构化大模型输出：output parser | ✅ 已完成 |
| 15 | Output Parser 实战 | ✅ 已完成 |
| 16 | Prompt Template：组件化管理 | ✅ 已完成 |
| 17 | Runnable：把写逻辑变成组装 chain | ✅ 已完成 |
| 18 | 实战练习 LCEL 组装 chain | ✅ 已完成 |
| 19 | LangChain 整体总结 | ✅ 已完成 |

### 第二阶段：企业级后端与进阶框架（第 20-28 篇）✅

| 序号 | 标题 | 状态 |
|------|------|------|
| 20 | FastAPI + LangChain 实现 SSE 流式接口 | ✅ 已完成 |
| 21 | FastAPI + tool 实现定时任务（上） | ✅ 已完成 |
| 22 | FastAPI + tool 实现定时任务（下） | ✅ 已完成 |
| 23 | 给 Agent 加上语音交互：ASR + TTS | ✅ 已完成 |
| 24 | AGUI 协议：流式组件渲染 | ✅ 已完成 |
| 25 | 图编排引擎：LangGraph 和多 Agent | ✅ 已完成 |
| 26 | Agentic RAG：自主决策的 RAG 闭环 | ✅ 已完成 |
| 27 | Docker Compose 本地开发与部署 | ✅ 已完成 |
| 28 | ElasticSearch 全文检索 | ✅ 已完成 |

### 第三阶段：检索增强与知识图谱（第 29-36 篇）✅

| 序号 | 标题 | 状态 |
|------|------|------|
| 29 | 混合检索 RAG：多路召回 + 重排 | ✅ 已完成 |
| 30 | Neo4j 知识图谱和 Graph RAG | ✅ 已完成 |
| 31 | LangSmith 全链路观测 | ✅ 已完成 |
| 32 | DeepAgents：skill 与上下文压缩 | ✅ 已完成 |
| 33 | DeepAgents 实战：深度调研助手 | ✅ 已完成 |
| 34 | PostgreSQL：AI 时代最适合的数据库 | ✅ 已完成 |
| 35 | Redis：Agent 短期记忆存储 | ✅ 已完成 |
| 36 | Mem0：分层记忆 + 三路召回 | ✅ 已完成 |

### 第四阶段：存储、消息与监控（第 37-43 篇）✅

| 序号 | 标题 | 状态 |
|------|------|------|
| 37 | FastAPI 进阶：企业级后端框架 | ✅ 已完成 |
| 38 | Agent 的对象存储方案 | ✅ 已完成 |
| 39 | 多模态与 OSS 前端直传实战 | ✅ 已完成 |
| 40 | RabbitMQ：异步处理标配方案 | ✅ 已完成 |
| 41 | LangFuse：开源全链路监测 | ✅ 已完成 |
| 42 | 图解 Transformer 架构 | ✅ 已完成 |
| 43 | 大模型训练、推理全流程 | ✅ 已完成 |

### 第五阶段：简历、面试与实战（第 44-56 篇）✅

| 序号 | 标题 | 状态 |
|------|------|------|
| 44 | 真实 Agent 简历参考库 | ✅ 已完成 |
| 45 | 真实 Agent 简历参考库（二） | ✅ 已完成 |
| 46 | Agent 面试题押题精讲：RAG 篇 | ✅ 已完成 |
| 47 | 真实 Agent 简历参考库（RAG专场） | ✅ 已完成 |
| 48 | 企业级知识库项目：项目介绍 | ✅ 已完成 |
| 49 | 企业级知识库项目：数据库设计 | ✅ 已完成 |
| 50 | 企业级知识库项目：文件解析为 md | ✅ 已完成 |
| 51 | 企业级知识库项目：异步 RAG 流水线 | ✅ 已完成 |
| 52 | 企业级知识库项目：全文检索链路 | ✅ 已完成 |
| 53 | 企业级知识库项目：抽取 Neo4j 实体 | ✅ 已完成 |
| 54 | 企业级知识库项目：文档审核机制 | ✅ 已完成 |
| 55 | 企业级知识库项目：用户鉴权流程 | ✅ 已完成 |
| 56 | 企业级知识库项目：用户模块完善 | ✅ 已完成 |

## 目录结构

```
agent-learning-python/
├── index.html                         ← 主页阅读界面（在线浏览）
├── README.md                          ← 本文件
├── 文章目录.md                         ← 完整文章目录
├── IMG/                               ← 图片资源（435+张，33个目录）
├── 01-第一阶段-Agent基础入门/          ← 19篇
├── 02-第二阶段-企业级后端与进阶框架/    ← 9篇
├── 03-第三阶段-检索增强与知识图谱/      ← 8篇
├── 04-第四阶段-存储消息与监控/          ← 7篇
└── 05-第五阶段-简历面试与实战/          ← 13篇
```

## 在线阅读

打开 `index.html` 即可在浏览器中阅读全部文章，支持：
- 📂 侧边栏目录导航（按阶段分组，可折叠）
- 🔍 全文搜索
- 📱 响应式设计（支持手机/平板/电脑）
- 🎨 代码语法高亮
- 🖼️ 图片自动加载

## 每篇文章包含

- ✅ 完全适配 Python（FastAPI + LangChain + SQLAlchemy）
- ✅ 适合新手学习（循序渐进、代码完整、注释清晰）
- ✅ 去除原始作者信息
- ✅ 图片与文章一一对应
- ✅ 每篇末尾添加配套代码库地址
- ✅ 添加学习要点和扩展方向
- ✅ 核心概念表格对比
- ✅ 完整可运行代码示例

## 学习路线

| 阶段 | 主题 | 核心技术 |
|------|------|----------|
| 一 | Agent 基础入门 | Tool、MCP、RAG、Milvus、Memory、LangChain LCEL |
| 二 | 企业级后端与进阶框架 | FastAPI、SSE、语音交互、AGUI、LangGraph、Docker |
| 三 | 检索增强与知识图谱 | 混合检索、Graph RAG、LangSmith、DeepAgents、PostgreSQL、Redis、Mem0 |
| 四 | 存储、消息与监控 | 对象存储、OSS 直传、RabbitMQ、LangFuse、Transformer 原理 |
| 五 | 简历、面试与企业级项目实战 | 简历参考、面试题、企业级知识库全流程开发 |

## 快速开始

```bash
# 1. 克隆仓库
git clone <仓库地址>
cd agent-learning-python

# 2. 在线阅读（直接用浏览器打开）
open index.html  # macOS
# 或
start index.html  # Windows
# 或
xdg-open index.html  # Linux

# 3. 安装 Python 依赖（运行代码示例时）
pip install langchain langchain-openai fastapi uvicorn sqlalchemy pymilvus
```

---

*本项目为学习用途，所有文章均已优化为 Python 版本，适合新手学习 AI Agent 全栈开发。*
