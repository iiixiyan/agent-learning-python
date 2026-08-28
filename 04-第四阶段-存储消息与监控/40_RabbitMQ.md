# RabbitMQ：Agent 中异步处理的标配方案

> **Python 版** | 基于 RabbitMQ + Python pika + FastAPI 技术栈
> 前置知识：[FastAPI 进阶](./37_FastAPI进阶.md)、[对象存储方案](./38_Agent的对象存储方案.md)

---

## 为什么需要消息队列？

Agent 系统中很多操作耗时很长（文档解析、向量化、大模型调用），如果同步等待会导致：

| 问题 | 说明 |
|------|------|
| **用户体验差** | 请求超时，用户长时间等待 |
| **系统脆弱** | 一个慢任务拖垮整个服务 |
| **无法削峰** | 突发流量直接打垮后端 |
| **耦合严重** | 生产者和消费者强依赖，难以独立扩展 |

**RabbitMQ 消息队列**解决这些问题：

| 能力 | 说明 |
|------|------|
| **异步** | 耗时任务丢到队列，立即返回，后台慢慢处理 |
| **解耦** | 生产者和消费者互不依赖，独立开发和部署 |
| **削峰** | 队列缓冲突发流量，消费者按自己的速度处理 |
| **可靠** | 消息持久化、确认机制、死信队列，保证消息不丢失 |

### 异步处理流程

```
同步方式（慢）：
用户请求 → 文档解析(10s) → 向量化(5s) → 索引(2s) → 返回结果
总耗时: 17s，用户等待 17s

异步方式（快）：
用户请求 → 发送任务到队列 → 立即返回"处理中"
总耗时: <1s，用户立即得到响应

后台消费者：
队列 → 文档解析(10s) → 向量化(5s) → 索引(2s) → 更新状态为"完成"
用户轮询或 WebSocket 通知获取最终结果
```

---

## 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **Producer（生产者）** | 发送消息的一方 | 寄信人 |
| **Consumer（消费者）** | 接收并处理消息的一方 | 收信人 |
| **Queue（队列）** | 存储消息的缓冲区 | 邮箱 |
| **Exchange（交换机）** | 接收生产者消息，路由到队列 | 邮局分拣中心 |
| **Binding（绑定）** | 交换机和队列的绑定关系 | 分拣规则 |
| **Routing Key（路由键）** | 消息的路由标识 | 邮政编码 |
| **Virtual Host（虚拟主机）** | 逻辑隔离的环境 | 不同的邮政系统 |

### 交换机类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **Direct（直连）** | 精确匹配 Routing Key | 点对点消息、任务分发 |
| **Fanout（广播）** | 忽略 Routing Key，广播到所有绑定队列 | 发布订阅、通知广播 |
| **Topic（主题）** | 通配符匹配 Routing Key（`*`匹配一个词，`#`匹配多个词） | 按主题分类、日志分级 |
| **Headers（头匹配）** | 按消息头属性匹配 | 复杂路由规则 |

---

## Docker 启动

```yaml
# docker-compose.yml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq-dev
    ports:
      - "5672:5672"    # AMQP 协议端口（程序连接用）
      - "15672:15672"  # 管理界面端口（浏览器访问）
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: always

volumes:
  rabbitmq_data:
```

```bash
# 启动 RabbitMQ
docker compose up -d

# 访问管理界面
# 浏览器打开 http://localhost:15672
# 用户名: admin, 密码: password
```

### 管理界面功能

| 功能 | 说明 |
|------|------|
| **Overview** | 概览：消息速率、连接数、队列数 |
| **Connections** | 连接：查看当前所有连接 |
| **Channels** | 信道：查看当前所有信道 |
| **Exchanges** | 交换机：查看和管理交换机 |
| **Queues** | 队列：查看队列消息数、消费者数 |
| **Admin** | 管理：用户、虚拟主机、权限 |

---

## Python 实现

### 安装依赖

```bash
pip install pika python-dotenv
```

### 完整示例

创建 `rabbitmq_demo.py`：

```python
"""
rabbitmq_demo.py - RabbitMQ 完整示例
包含：生产者、消费者、消息确认、死信队列、多阶段流水线
"""
import os
import json
import time
from dotenv import load_dotenv
import pika

load_dotenv()


# ========== 连接配置 ==========
def get_connection():
    """
    获取 RabbitMQ 连接

    Returns:
        pika.BlockingConnection: RabbitMQ 连接
    """
    credentials = pika.PlainCredentials(
        username=os.getenv("RABBITMQ_USER", "admin"),
        password=os.getenv("RABBITMQ_PASSWORD", "password"),
    )
    parameters = pika.ConnectionParameters(
        host=os.getenv("RABBITMQ_HOST", "localhost"),
        port=int(os.getenv("RABBITMQ_PORT", 5672)),
        credentials=credentials,
        heartbeat=600,  # 心跳间隔
        blocked_connection_timeout=300,
    )
    return pika.BlockingConnection(parameters)


# ========== 1. 生产者 ==========
def publish_task(task_type: str, data: dict, queue_name: str = "agent_tasks"):
    """
    发布任务到队列

    Args:
        task_type: 任务类型
        data: 任务数据
        queue_name: 队列名称
    """
    conn = get_connection()
    ch = conn.channel()

    # 声明队列（durable=True 持久化，RabbitMQ 重启后队列不丢失）
    ch.queue_declare(queue=queue_name, durable=True)

    # 构造消息
    message = {
        "type": task_type,
        "data": data,
        "timestamp": time.time(),
    }

    # 发布消息
    ch.basic_publish(
        exchange="",  # 空字符串表示默认交换机（Direct 类型）
        routing_key=queue_name,
        body=json.dumps(message, ensure_ascii=False),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 2=持久化消息，RabbitMQ 重启后消息不丢失
            content_type="application/json",
        ),
    )

    conn.close()
    print(f"✅ 已发布任务: {task_type} -> {queue_name}")


# ========== 2. 消费者 ==========
def process_task(ch, method, properties, body):
    """
    处理任务的回调函数

    Args:
        ch: 信道
        method: 方法
        properties: 属性
        body: 消息体
    """
    task = json.loads(body)
    task_type = task["type"]
    task_data = task["data"]

    print(f"\n📥 收到任务: {task_type}")
    print(f"   数据: {json.dumps(task_data, ensure_ascii=False, indent=2)}")

    try:
        # 根据任务类型处理
        if task_type == "parse_document":
            # 文档解析（耗时操作）
            print(f"   📄 正在解析文档: {task_data.get('file_path')}")
            time.sleep(2)  # 模拟耗时
            print(f"   ✅ 文档解析完成")

        elif task_type == "embed_document":
            # 向量化（耗时操作）
            print(f"   🔢 正在向量化文档: {task_data.get('doc_id')}")
            time.sleep(1)  # 模拟耗时
            print(f"   ✅ 向量化完成")

        elif task_type == "index_document":
            # 索引（耗时操作）
            print(f"   📚 正在索引文档: {task_data.get('doc_id')}")
            time.sleep(0.5)  # 模拟耗时
            print(f"   ✅ 索引完成")

        else:
            print(f"   ⚠️  未知任务类型: {task_type}")

        # 确认消息处理成功（重要！不确认的话消息会一直留在队列中）
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"   ✅ 任务处理完成，已确认")

    except Exception as e:
        print(f"   ❌ 任务处理失败: {e}")
        # 处理失败，重新入队（requeue=True）或拒绝（requeue=False 进入死信队列）
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        print(f"   🔄 消息已重新入队")


def start_consumer(queue_name: str = "agent_tasks"):
    """
    启动消费者

    Args:
        queue_name: 队列名称
    """
    conn = get_connection()
    ch = conn.channel()

    # 声明队列
    ch.queue_declare(queue=queue_name, durable=True)

    # 公平分发：一次只处理一个任务，处理完再取下一个
    # 避免一个消费者被压垮，其他消费者空闲
    ch.basic_qos(prefetch_count=1)

    # 消费消息
    ch.basic_consume(
        queue=queue_name,
        on_message_callback=process_task,
        # auto_ack=False（默认），需要手动确认
    )

    print(f"\n🚀 消费者已启动，等待任务...")
    print(f"   队列: {queue_name}")
    print(f"   按 Ctrl+C 停止\n")

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        print("\n⏹️  消费者已停止")
        conn.close()


# ========== 使用示例 ==========
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "consumer":
        # 启动消费者
        start_consumer()
    else:
        # 生产者：发送测试任务
        print("="*60)
        print("📤 生产者模式：发送测试任务")
        print("="*60)

        # 发送文档解析任务
        publish_task("parse_document", {
            "doc_id": "doc_001",
            "file_path": "/docs/report.pdf",
            "file_type": "pdf",
        })

        # 发送向量化任务
        publish_task("embed_document", {
            "doc_id": "doc_001",
            "model": "text-embedding-3-small",
        })

        # 发送索引任务
        publish_task("index_document", {
            "doc_id": "doc_001",
            "index_name": "knowledge_base",
        })

        print("\n💡 提示：运行 'python rabbitmq_demo.py consumer' 启动消费者处理任务")
```

### 运行示例

```bash
# 终端1：启动消费者
python rabbitmq_demo.py consumer

# 终端2：发送任务
python rabbitmq_demo.py
```

---

## 死信队列（DLX）

处理失败的消息如果一直重新入队，会导致无限重试。**死信队列（Dead Letter Exchange）**用于处理失败消息。

### 配置死信队列

```python
"""
dlx_demo.py - 死信队列配置示例
处理失败的消息进入死信队列，避免无限重试
"""
import pika
import json

conn = pika.BlockingConnection(pika.ConnectionParameters(
    'localhost', credentials=pika.PlainCredentials('admin', 'password')
))
ch = conn.channel()

# ========== 1. 声明死信交换机和死信队列 ==========
# 死信交换机
ch.exchange_declare(exchange='dlx_exchange', exchange_type='direct', durable=True)
# 死信队列
ch.queue_declare(queue='dead_letter_queue', durable=True)
# 绑定
ch.queue_bind(queue='dead_letter_queue', exchange='dlx_exchange', routing_key='dead_letter')

# ========== 2. 声明业务队列（指定死信交换机） ==========
ch.queue_declare(
    queue='agent_tasks_with_dlx',
    durable=True,
    arguments={
        # 指定死信交换机
        'x-dead-letter-exchange': 'dlx_exchange',
        # 指定死信路由键（可选）
        'x-dead-letter-routing-key': 'dead_letter',
        # 消息最大重试次数（可选，需要配合插件）
        # 'x-message-ttl': 60000,  # 消息过期时间（毫秒）
    }
)

# ========== 3. 消费者：失败时拒绝消息（不重新入队） ==========
def callback_with_dlx(ch, method, properties, body):
    task = json.loads(body)
    try:
        # 处理任务
        print(f"处理任务: {task['type']}")
        # ... 实际处理逻辑 ...

        # 模拟失败
        if task.get("should_fail"):
            raise Exception("模拟处理失败")

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("✅ 处理成功")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        # 拒绝消息，不重新入队（requeue=False）
        # 消息会进入死信队列
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        print("🔄 消息已进入死信队列")


ch.basic_qos(prefetch_count=1)
ch.basic_consume(queue='agent_tasks_with_dlx', on_message_callback=callback_with_dlx)

print("等待任务...")
ch.start_consuming()
```

### 死信队列处理

```python
"""
dlx_consumer.py - 死信队列消费者
处理死信队列中的消息：记录日志、告警、人工介入
"""
import pika
import json
from datetime import datetime

conn = pika.BlockingConnection(pika.ConnectionParameters(
    'localhost', credentials=pika.PlainCredentials('admin', 'password')
))
ch = conn.channel()


def process_dead_letter(ch, method, properties, body):
    """处理死信消息"""
    task = json.loads(body)
    print(f"\n💀 收到死信消息:")
    print(f"   时间: {datetime.now().isoformat()}")
    print(f"   任务: {json.dumps(task, ensure_ascii=False, indent=2)}")

    # 记录日志、发送告警、人工介入
    # ...

    # 确认处理
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("   ✅ 死信消息已处理")


ch.basic_consume(queue='dead_letter_queue', on_message_callback=process_dead_letter)
print("💀 死信队列消费者已启动...")
ch.start_consuming()
```

---

## Agent 系统中的典型应用

### 场景：RAG 知识库多阶段流水线

用户上传文档 → 异步解析 → 向量化 → 存入知识库

```
用户上传文档
     │
     ▼
┌─────────────────────────────────────────────┐
│  API 层（FastAPI）                           │
│  1. 保存文件到 OSS                           │
│  2. 发送 parse 任务到队列，立即返回          │
│  返回: {doc_id, status: "processing"}       │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  队列: rag_pipeline                          │
│  [parse, embed, index] 三个阶段的任务       │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  消费者（Worker）                            │
│  按 stage 字段处理不同阶段                    │
│  parse → 解析文档 → 发送 embed 任务         │
│  embed → 向量化 → 发送 index 任务           │
│  index → 存入索引 → 更新状态为"完成"        │
└─────────────────────────────────────────────┘
```

### FastAPI 集成示例

```python
"""
rag_pipeline_api.py - RAG 知识库流水线 API
FastAPI + RabbitMQ 实现异步文档处理
"""
import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pika
import json
import boto3
from botocore.config import Config

load_dotenv()

app = FastAPI(title="RAG 知识库流水线 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 文档状态存储（实际用 PostgreSQL）
doc_status = {}


# ========== RabbitMQ 连接 ==========
def get_rabbitmq_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(
        host=os.getenv("RABBITMQ_HOST", "localhost"),
        credentials=pika.PlainCredentials(
            os.getenv("RABBITMQ_USER", "admin"),
            os.getenv("RABBITMQ_PASSWORD", "password")
        )
    ))


# ========== OSS 连接 ==========
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("OSS_ENDPOINT"),
    aws_access_key_id=os.getenv("OSS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("OSS_ACCESS_KEY_SECRET"),
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)


# ========== API 接口 ==========
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档接口
    1. 保存文件到 OSS
    2. 发送解析任务到队列
    3. 立即返回 doc_id 和状态
    """
    # 1. 生成文档 ID
    doc_id = str(uuid.uuid4())[:8]
    object_key = f"documents/{doc_id}/{file.filename}"

    # 2. 保存文件到 OSS
    file_content = await file.read()
    s3.put_object(
        Bucket=os.getenv("OSS_BUCKET"),
        Key=object_key,
        Body=file_content,
        ContentType=file.content_type,
    )

    # 3. 初始化状态
    doc_status[doc_id] = {
        "filename": file.filename,
        "object_key": object_key,
        "status": "processing",
        "stage": "parse",
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }

    # 4. 发送解析任务到队列
    conn = get_rabbitmq_connection()
    ch = conn.channel()
    ch.queue_declare(queue="rag_pipeline", durable=True)
    ch.basic_publish(
        exchange="",
        routing_key="rag_pipeline",
        body=json.dumps({
            "doc_id": doc_id,
            "stage": "parse",
            "object_key": object_key,
        }, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    conn.close()

    return {"doc_id": doc_id, "status": "processing", "message": "文档已上传，正在处理中"}


@app.get("/api/documents/{doc_id}/status")
def get_document_status(doc_id: str):
    """查询文档处理状态"""
    if doc_id not in doc_status:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc_status[doc_id]


# ========== 消费者（Worker）==========
def rag_pipeline_worker():
    """RAG 流水线消费者：按阶段处理文档"""
    conn = get_rabbitmq_connection()
    ch = conn.channel()
    ch.queue_declare(queue="rag_pipeline", durable=True)
    ch.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        task = json.loads(body)
        doc_id = task["doc_id"]
        stage = task["stage"]

        print(f"\n📥 处理文档: {doc_id}, 阶段: {stage}")

        try:
            if stage == "parse":
                # 1. 从 OSS 下载文件
                # 2. 解析文档（PDF/Word/Markdown）
                # 3. 文本切分
                print("   📄 解析文档...")
                # parse_document(task["object_key"])

                # 发送下一阶段任务
                ch.basic_publish(
                    exchange="", routing_key="rag_pipeline",
                    body=json.dumps({"doc_id": doc_id, "stage": "embed"}, ensure_ascii=False),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
                doc_status[doc_id]["stage"] = "embed"

            elif stage == "embed":
                # 4. 向量化（调用 Embedding 模型）
                print("   🔢 向量化...")
                # embed_document(doc_id)

                # 发送下一阶段任务
                ch.basic_publish(
                    exchange="", routing_key="rag_pipeline",
                    body=json.dumps({"doc_id": doc_id, "stage": "index"}, ensure_ascii=False),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
                doc_status[doc_id]["stage"] = "index"

            elif stage == "index":
                # 5. 存入向量索引
                print("   📚 存入索引...")
                # index_document(doc_id)

                # 更新状态为完成
                doc_status[doc_id]["status"] = "ready"
                doc_status[doc_id]["stage"] = "complete"
                print("   ✅ 文档处理完成！")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            doc_status[doc_id]["status"] = "failed"
            doc_status[doc_id]["error"] = str(e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    ch.basic_consume(queue="rag_pipeline", on_message_callback=callback)
    print("🚀 RAG 流水线消费者已启动...")
    ch.start_consuming()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        rag_pipeline_worker()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 运行方式

```bash
# 终端1：启动 API 服务
python rag_pipeline_api.py

# 终端2：启动消费者（Worker）
python rag_pipeline_api.py worker

# 终端3：上传文档测试
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.pdf"

# 查询状态
curl http://localhost:8000/api/documents/{doc_id}/status
```

---

## 学习要点

1. **RabbitMQ** 用 AMQP 协议，支持持久化、确认机制、死信队列，是 Agent 异步处理的标配方案
2. **核心概念**：Producer（生产者）、Consumer（消费者）、Queue（队列）、Exchange（交换机）、Binding（绑定）、Routing Key（路由键）
3. **交换机类型**：Direct（精确匹配）、Fanout（广播）、Topic（通配符）、Headers（头匹配）
4. **消息持久化**：队列 durable=True + 消息 delivery_mode=2，RabbitMQ 重启后不丢失
5. **消息确认**：处理成功要 `basic_ack`，失败要 `basic_nack` 并决定是否重新入队
6. **公平分发**：`basic_qos(prefetch_count=1)` 一次只处理一个任务，避免一个消费者被压垮
7. **死信队列（DLX）**：处理失败的消息进入死信队列，避免无限重试，用于告警和人工介入
8. **多阶段流水线**：通过消息中的 stage 字段实现，每个阶段处理完发送下一阶段任务
9. **典型应用**：RAG 知识库流水线（解析→向量化→索引）、大模型调用异步化、邮件/通知发送、日志异步处理
10. **生产环境**：配置死信队列、消息过期时间、监控队列长度、消费者异常告警、集群部署保证高可用

## 扩展方向

- 学习 RabbitMQ 集群部署（镜像队列、仲裁队列、高可用）
- 探索 RabbitMQ 插件（延迟消息插件、消息追踪、管理插件）
- 学习 RabbitMQ 性能优化（预取数、批量确认、消息压缩）
- 探索其他消息队列（Kafka、RocketMQ、Redis Stream、NATS）
- 学习消息队列选型对比（RabbitMQ vs Kafka vs RocketMQ）
- 探索分布式任务队列（Celery、RQ、Dramatiq）
- 学习 RabbitMQ 监控（Prometheus + Grafana、管理 API）
- 探索消息队列事务和幂等性设计
- 学习 RabbitMQ 安全配置（用户权限、TLS、虚拟主机隔离）
- 探索事件驱动架构（EDA）和事件溯源（Event Sourcing）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/04-storage-monitoring/40-rabbitmq-async

包含本文的完整可运行代码示例（生产者+消费者+死信队列+RAG流水线完整前后端）。

---

**上一篇**：[多模态与 OSS 直传](./39_多模态与OSS前端直传实战.md) | **下一篇**：[LangFuse 可观测平台](./41_LangFuse.md)
