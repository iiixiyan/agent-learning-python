# 企业级知识库项目：基于消息队列的异步 RAG 流水线

> **Python 版** | RabbitMQ + Celery 异步文档处理流水线 + 完整状态机 + 死信队列 + 监控
> 前置知识：[RabbitMQ 消息队列](../04-第四阶段-存储消息与监控/40_RabbitMQ.md)、[RAG 检索增强生成](../01-第一阶段-Agent基础入门/11_RAG检索增强生成.md)、[文件解析为 md](./50_企业级知识库项目-文件解析为md.md)

---

## 为什么需要异步流水线？

用户上传一个 100 页的 PDF，同步处理需要几分钟，用户会等得不耐烦甚至超时。

**异步方案**：上传后立即返回"处理中"，后台用消息队列异步处理，处理完通知用户。

| 处理方式 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| **同步处理** | 实现简单，即时反馈 | 大文档超时，用户体验差 | 小文档（<10页） |
| **异步流水线** | 不阻塞，可扩展，容错好 | 实现复杂，需要状态管理 | 大文档、生产环境 |

---

## 流水线架构

```
用户上传文档
    ↓
保存文件到 OSS/MinIO
    ↓
数据库状态: pending → parsing
    ↓
发送消息到 RabbitMQ (doc_parse_queue)
    ↓立即返回 doc_id
    ↓
消费者按阶段处理：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  文档解析     │ → │  文本切分     │ → │  向量化       │ → │  建立索引     │
│  (parse)     │    │  (split)     │    │  (embed)     │    │  (index)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    ↓                  ↓                  ↓                  ↓
状态: parsing      状态: splitting    状态: embedding    状态: indexed(完成)
```

### 各阶段职责

| 阶段 | 队列 | 职责 | 输入 | 输出 |
|------|------|------|------|------|
| **文档解析** | doc_parse_queue | PDF/DOCX/PPTX → Markdown | 原始文件 | Markdown 文本 |
| **文本切分** | doc_split_queue | Markdown → Chunks | Markdown 文本 | 分块列表 |
| **向量化** | doc_embed_queue | Chunks → 向量 | 分块列表 | 向量数据 |
| **建立索引** | doc_index_queue | 向量 → PGVector/ES/Neo4j | 向量数据 | 索引完成 |

---

## 完整实现（RabbitMQ + pika）

### 1. FastAPI 上传接口

```python
"""
main.py - FastAPI 文档上传接口
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import json
import pika
from minio import Minio
from datetime import datetime

app = FastAPI()

# MinIO 客户端
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password",
    secure=False,
)

# RabbitMQ 连接工厂
def get_rabbit_channel():
    conn = pika.BlockingConnection(
        pika.ConnectionParameters('localhost', heartbeat=600)
    )
    return conn.channel()


class UploadResponse(BaseModel):
    doc_id: str
    status: str
    message: str


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并触发异步处理流水线

    - 保存文件到 MinIO
    - 创建数据库记录（status=pending）
    - 发送消息到解析队列
    - 立即返回 doc_id
    """
    doc_id = str(uuid.uuid4())

    try:
        # 1. 保存文件到 MinIO
        file_path = f"documents/{doc_id}/{file.filename}"
        content = await file.read()

        # 确保 bucket 存在
        if not minio_client.bucket_exists("kb-bucket"):
            minio_client.make_bucket("kb-bucket")

        minio_client.put_object(
            "kb-bucket",
            file_path,
            data=__import__('io').BytesIO(content),
            length=len(content),
            content_type=file.content_type,
        )

        # 2. 数据库插入记录（status=pending）
        # 实际项目中使用 SQLAlchemy
        # doc = Document(id=doc_id, file_name=file.filename, status='pending', ...)
        # db.add(doc); db.commit()

        # 3. 发送消息到解析队列
        channel = get_rabbit_channel()

        # 声明队列（durable=True 持久化）
        channel.queue_declare(queue='doc_parse_queue', durable=True)

        # 发送消息（delivery_mode=2 持久化消息）
        message = {
            'doc_id': doc_id,
            'file_path': file_path,
            'file_name': file.filename,
            'file_size': len(content),
            'stage': 'parse',
            'created_at': datetime.now().isoformat(),
        }

        channel.basic_publish(
            exchange='',
            routing_key='doc_parse_queue',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化消息
                content_type='application/json',
            ),
        )
        channel.close()

        return {
            "doc_id": doc_id,
            "status": "processing",
            "message": "文档已上传，正在处理中",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/documents/{doc_id}/status")
async def get_status(doc_id: str):
    """查询文档处理状态和进度"""
    # 从数据库查询状态
    # doc = db.query(Document).filter_by(id=doc_id).first()
    # if not doc: raise HTTPException(404)

    return {
        "doc_id": doc_id,
        "status": "parsing",  # pending/parsing/splitting/embedding/indexing/ready/failed
        "progress": 50,  # 0-100
        "current_stage": "parse",
        "chunk_count": 0,
        "error_message": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:01:00",
    }
```

### 2. 消费者：文档解析阶段

```python
"""
worker_parse.py - 文档解析消费者
"""
import pika
import json
import os
import tempfile
from minio import Minio
from unstructured.partition.auto import partition

# MinIO 客户端
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password",
    secure=False,
)


def parse_document(ch, method, properties, body):
    """
    文档解析处理函数

    1. 从 MinIO 下载文件
    2. 用 unstructured 解析为文本
    3. 保存解析结果到 MongoDB/数据库
    4. 更新状态，发送到下一阶段队列
    """
    task = json.loads(body)
    doc_id = task['doc_id']
    file_path = task['file_path']

    try:
        print(f"[解析] 开始处理文档: {doc_id}")

        # 1. 从 MinIO 下载文件到临时目录
        response = minio_client.get_object("kb-bucket", file_path)
        file_data = response.read()

        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, task['file_name'])
        with open(temp_file, 'wb') as f:
            f.write(file_data)

        # 2. 用 unstructured 解析
        elements = partition(filename=temp_file)
        text_content = "\n".join([str(e) for e in elements])

        print(f"[解析] 解析完成，文本长度: {len(text_content)}")

        # 3. 保存解析结果到 MongoDB/数据库
        # mongo_db.document_contents.insert_one({
        #     "document_id": doc_id,
        #     "raw_text": text_content,
        #     "markdown": text_content,
        #     "parsed_at": datetime.now(),
        # })

        # 更新数据库状态
        # db.query(Document).filter_by(id=doc_id).update({"status": "splitting"})

        # 4. 发送到下一阶段队列（切分）
        ch.queue_declare(queue='doc_split_queue', durable=True)
        next_task = {
            'doc_id': doc_id,
            'stage': 'split',
            'text_length': len(text_content),
        }
        ch.basic_publish(
            exchange='',
            routing_key='doc_split_queue',
            body=json.dumps(next_task),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # 确认消息
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[解析] 完成文档: {doc_id}")

        # 清理临时文件
        os.remove(temp_file)
        os.rmdir(temp_dir)

    except Exception as e:
        print(f"[解析] 失败文档: {doc_id}, 错误: {e}")

        # 更新状态为 failed
        # db.query(Document).filter_by(id=doc_id).update({
        #     "status": "failed",
        #     "error_message": str(e),
        # })

        # 发送到死信队列
        ch.basic_publish(
            exchange='',
            routing_key='doc_parse_dlq',
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )

        # 拒绝消息（不重新入队）
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# 启动消费者
if __name__ == "__main__":
    conn = pika.BlockingConnection(
        pika.ConnectionParameters('localhost', heartbeat=600)
    )
    channel = conn.channel()

    # 声明队列
    channel.queue_declare(queue='doc_parse_queue', durable=True)
    channel.queue_declare(queue='doc_parse_dlq', durable=True)

    # 预取计数：每个消费者一次只处理一个任务
    channel.basic_qos(prefetch_count=1)

    # 消费消息
    channel.basic_consume(
        queue='doc_parse_queue',
        on_message_callback=parse_document,
    )

    print("[解析] 等待解析任务...")
    channel.start_consuming()
```

### 3. 消费者：切分 + 向量化 + 索引（合并阶段）

```python
"""
worker_embed.py - 切分+向量化+索引消费者（合并阶段）
"""
import pika
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from elasticsearch import Elasticsearch

# 初始化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
es_client = Elasticsearch("http://localhost:9200")

CONNECTION_STRING = "postgresql+psycopg2://user:password@localhost:5432/kb"


def process_document(ch, method, properties, body):
    """
    切分+向量化+索引处理函数

    1. 从 MongoDB 读取解析后的文本
    2. 文本切分
    3. 向量化并存入 PGVector
    4. 建立 ElasticSearch 全文索引
    5. 更新状态为完成
    """
    task = json.loads(body)
    doc_id = task['doc_id']

    try:
        print(f"[处理] 开始处理文档: {doc_id}")

        # 1. 从 MongoDB 读取解析后的文本
        # doc = mongo_db.document_contents.find_one({"document_id": doc_id})
        text = "..."  # doc['raw_text']

        # 2. 文本切分
        print(f"[处理] 开始切分...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        print(f"[处理] 切分为 {len(chunks)} 块")

        # 更新状态
        # db.query(Document).filter_by(id=doc_id).update({
        #     "status": "embedding",
        #     "chunk_count": len(chunks),
        # })

        # 3. 向量化并存入 PGVector
        print(f"[处理] 开始向量化...")
        vectorstore = PGVector.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=doc_id,
            connection_string=CONNECTION_STRING,
        )
        print(f"[处理] 向量化完成")

        # 4. 建立 ElasticSearch 全文索引
        print(f"[处理] 建立 ES 索引...")
        index_name = f"documents_{doc_id}"

        # 创建索引（如果不存在）
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "chunk_index": {"type": "integer"},
                            "document_id": {"type": "keyword"},
                        }
                    }
                },
            )

        # 批量插入
        from elasticsearch.helpers import bulk
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "content": chunk,
                    "chunk_index": i,
                    "document_id": doc_id,
                },
            }
            for i, chunk in enumerate(chunks)
        ]
        bulk(es_client, actions)
        print(f"[处理] ES 索引完成")

        # 5. 更新状态为完成
        # db.query(Document).filter_by(id=doc_id).update({
        #     "status": "ready",
        #     "chunk_count": len(chunks),
        # })

        # 确认消息
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[处理] 完成文档: {doc_id}, 共 {len(chunks)} 块")

    except Exception as e:
        print(f"[处理] 失败文档: {doc_id}, 错误: {e}")

        # 更新状态为 failed
        # db.query(Document).filter_by(id=doc_id).update({
        #     "status": "failed",
        #     "error_message": str(e),
        # })

        # 发送到死信队列
        ch.basic_publish(
            exchange='',
            routing_key='doc_split_dlq',
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )

        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# 启动消费者
if __name__ == "__main__":
    conn = pika.BlockingConnection(
        pika.ConnectionParameters('localhost', heartbeat=600)
    )
    channel = conn.channel()

    channel.queue_declare(queue='doc_split_queue', durable=True)
    channel.queue_declare(queue='doc_split_dlq', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='doc_split_queue',
        on_message_callback=process_document,
    )

    print("[处理] 等待切分+向量化任务...")
    channel.start_consuming()
```

---

## Celery 实现方案（生产环境推荐）

生产环境推荐使用 Celery + Redis/RabbitMQ，更成熟、更易维护。

### 安装依赖

```bash
pip install celery redis
```

### Celery 配置

```python
"""
celery_app.py - Celery 应用配置
"""
from celery import Celery

app = Celery(
    'knowledge_base',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30分钟超时
    task_soft_time_limit=25 * 60,  # 25分钟软超时
    worker_prefetch_multiplier=1,  # 每个 worker 一次只取一个任务
    worker_max_tasks_per_child=100,  # 每个 worker 处理100个任务后重启
)

# 自动发现任务
app.autodiscover_tasks(['tasks'])
```

### Celery 任务定义

```python
"""
tasks/document_tasks.py - Celery 文档处理任务
"""
from celery_app import app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def parse_document_task(self, doc_id, file_path):
    """
    文档解析任务

    - bind=True: 可以访问 self（任务实例）
    - max_retries=3: 最多重试3次
    - default_retry_delay=60: 重试间隔60秒
    """
    try:
        logger.info(f"开始解析文档: {doc_id}")

        # 更新状态
        self.update_state(state='PROGRESS', meta={'stage': 'parse', 'progress': 10})

        # 1. 下载文件
        # 2. 解析文档
        # 3. 保存结果

        self.update_state(state='PROGRESS', meta={'stage': 'parse', 'progress': 100})
        logger.info(f"解析完成: {doc_id}")

        # 触发下一个任务
        split_document_task.delay(doc_id)

        return {'doc_id': doc_id, 'status': 'parsed'}

    except Exception as e:
        logger.error(f"解析失败: {doc_id}, {e}")
        # 重试
        raise self.retry(exc=e, countdown=60)


@app.task(bind=True, max_retries=3)
def split_document_task(self, doc_id):
    """文本切分任务"""
    try:
        logger.info(f"开始切分文档: {doc_id}")
        self.update_state(state='PROGRESS', meta={'stage': 'split', 'progress': 50})

        # 切分逻辑...

        self.update_state(state='PROGRESS', meta={'stage': 'split', 'progress': 100})
        embed_document_task.delay(doc_id)

        return {'doc_id': doc_id, 'status': 'split'}

    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=3)
def embed_document_task(self, doc_id):
    """向量化任务"""
    try:
        logger.info(f"开始向量化文档: {doc_id}")
        self.update_state(state='PROGRESS', meta={'stage': 'embed', 'progress': 50})

        # 向量化逻辑...

        self.update_state(state='PROGRESS', meta={'stage': 'embed', 'progress': 100})
        index_document_task.delay(doc_id)

        return {'doc_id': doc_id, 'status': 'embedded'}

    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=3)
def index_document_task(self, doc_id):
    """建立索引任务"""
    try:
        logger.info(f"开始建立索引: {doc_id}")
        self.update_state(state='PROGRESS', meta={'stage': 'index', 'progress': 50})

        # 建立索引逻辑...

        self.update_state(state='SUCCESS', meta={'stage': 'index', 'progress': 100})
        logger.info(f"文档处理完成: {doc_id}")

        return {'doc_id': doc_id, 'status': 'ready'}

    except Exception as e:
        raise self.retry(exc=e)
```

### 启动 Celery Worker

```bash
# 启动 worker（并发4）
celery -A celery_app worker --loglevel=info --concurrency=4

# 启动多个 worker（不同队列）
celery -A celery_app worker -Q parse_queue --loglevel=info --concurrency=2 -n parse@%h
celery -A celery_app worker -Q embed_queue --loglevel=info --concurrency=4 -n embed@%h

# 启动 Flower 监控面板
celery -A celery_app flower --port=5555
```

---

## 关键设计点

### 1. 状态机

```
pending → parsing → splitting → embedding → indexing → ready
                ↓           ↓           ↓           ↓
              failed      failed      failed      failed
```

每个阶段失败都要记录错误信息，支持重试。

| 状态 | 说明 | 可转换到 |
|------|------|----------|
| **pending** | 已上传，等待处理 | parsing |
| **parsing** | 正在解析文档 | splitting / failed |
| **splitting** | 正在切分文本 | embedding / failed |
| **embedding** | 正在向量化 | indexing / failed |
| **indexing** | 正在建立索引 | ready / failed |
| **ready** | 处理完成，可检索 | - |
| **failed** | 处理失败 | pending（重试） |

### 2. 死信队列

处理失败的消息进入死信队列，方便排查和手动重试。

```python
# 声明死信队列
channel.queue_declare(queue='doc_parse_dlq', durable=True)

# 失败时发送到死信队列
channel.basic_publish(
    exchange='',
    routing_key='doc_parse_dlq',
    body=body,
    properties=pika.BasicProperties(delivery_mode=2),
)

# 手动重试死信队列中的消息
def retry_dlq(queue_name, dlq_name):
    """将死信队列中的消息重新入队"""
    channel.queue_declare(queue=queue_name, durable=True)
    channel.queue_declare(queue=dlq_name, durable=True)

    while True:
        method_frame, header_frame, body = channel.basic_get(queue=dlq_name)
        if method_frame is None:
            break

        # 重新发送到主队列
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
        channel.basic_ack(method_frame.delivery_tag)
        print(f"重试消息: {body[:100]}")
```

### 3. 进度通知

处理完成后可以通过 WebSocket / SSE 通知前端，或者前端轮询状态接口。

```python
"""
WebSocket 进度通知示例
"""
from fastapi import WebSocket
from typing import List

active_connections: List[WebSocket] = []


@app.websocket("/ws/{doc_id}")
async def websocket_endpoint(websocket: WebSocket, doc_id: str):
    """WebSocket 实时推送文档处理进度"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # 查询数据库状态
            # doc = db.query(Document).filter_by(id=doc_id).first()
            status = "parsing"  # 示例
            progress = 50

            await websocket.send_json({
                "doc_id": doc_id,
                "status": status,
                "progress": progress,
            })

            if status in ["ready", "failed"]:
                break

            await asyncio.sleep(2)  # 每2秒推送一次

    finally:
        active_connections.remove(websocket)
```

### 4. 并发控制

用 `basic_qos(prefetch_count=1)` 让每个消费者一次只处理一个任务，避免被压垮。可以启动多个消费者实例提升吞吐量。

```python
# 每个消费者一次只处理一个任务
channel.basic_qos(prefetch_count=1)

# 启动多个消费者实例（水平扩展）
# python worker_parse.py  # 实例1
# python worker_parse.py  # 实例2
# python worker_parse.py  # 实例3
```

### 5. 幂等性设计

任务可能被重复执行（网络超时、重试），需要保证幂等性：

```python
def process_document(doc_id):
    """幂等处理：检查是否已处理"""
    # 1. 检查文档状态
    doc = db.query(Document).filter_by(id=doc_id).first()
    if doc.status == "ready":
        print(f"文档 {doc_id} 已处理，跳过")
        return

    # 2. 清理旧数据（如果有）
    # 删除旧的向量、索引
    # vectorstore.delete(where={"document_id": doc_id})
    # es_client.delete_by_query(index="documents_*", body={"query": {"term": {"document_id": doc_id}}})

    # 3. 重新处理
    # ...
```

---

## Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"  # 管理面板
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # Redis（Celery broker）
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # PostgreSQL + PGVector
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: kb
    volumes:
      - pg_data:/var/lib/postgresql/data

  # MinIO
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"  # 控制台
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  # ElasticSearch
  elasticsearch:
    image: elasticsearch:8.11.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  # FastAPI 应用
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - rabbitmq
      - postgres
      - minio
    environment:
      - RABBITMQ_URL=amqp://admin:password@rabbitmq:5672
      - DATABASE_URL=postgresql://user:password@postgres:5432/kb
      - MINIO_ENDPOINT=minio:9000

  # 解析 Worker
  worker-parse:
    build: .
    command: python worker_parse.py
    depends_on:
      - rabbitmq
      - postgres
      - minio
    deploy:
      replicas: 2  # 2个实例

  # 向量化 Worker
  worker-embed:
    build: .
    command: python worker_embed.py
    depends_on:
      - rabbitmq
      - postgres
      - elasticsearch
    deploy:
      replicas: 4  # 4个实例（向量化是瓶颈）

  # Flower 监控
  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis

volumes:
  rabbitmq_data:
  redis_data:
  pg_data:
  minio_data:
  es_data:
```

---

## 监控与告警

### 关键指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| **队列积压数** | 待处理消息数量 | > 100 |
| **处理成功率** | 成功/总数 | < 95% |
| **平均处理时长** | 每个文档处理时间 | > 5分钟 |
| **死信队列数** | 失败消息数量 | > 10 |
| **Worker 存活数** | 在线消费者数量 | < 预期 |

### Prometheus + Grafana 监控

```python
"""
metrics.py - Prometheus 指标采集
"""
from prometheus_client import Counter, Histogram, Gauge

# 计数器
parse_total = Counter('doc_parse_total', '文档解析总数', ['status'])
parse_duration = Histogram('doc_parse_duration_seconds', '文档解析耗时')

# 仪表盘
queue_depth = Gauge('doc_queue_depth', '队列积压数', ['queue'])
worker_count = Gauge('doc_worker_count', 'Worker 数量', ['type'])

# 使用示例
@parse_duration.time()
def parse_document(doc_id):
    try:
        # 解析逻辑...
        parse_total.labels(status='success').inc()
    except Exception:
        parse_total.labels(status='failed').inc()
        raise
```

---

## 学习要点

1. **异步流水线解决大文档处理超时问题**：上传后立即返回，后台异步处理，用户体验好
2. **用 RabbitMQ/Celery 解耦各阶段**：每个阶段独立扩展，解析慢就加解析 Worker，向量化慢就加向量化 Worker
3. **状态机设计是核心**：pending→parsing→splitting→embedding→indexing→ready，每个阶段都要有成功/失败状态
4. **死信队列处理失败消息**：失败消息进入 DLQ，方便排查和手动重试，避免丢失
5. **切分和向量化可以合并为一个阶段**：减少队列数量和复杂度，向量化是瓶颈可以多开 Worker
6. **生产环境用 Celery + Redis/RabbitMQ 更成熟**：Celery 提供重试、超时、监控、Flower 面板等企业级特性
7. **幂等性设计很重要**：任务可能被重复执行（网络超时、重试），需要检查状态、清理旧数据
8. **并发控制用 prefetch_count=1**：每个消费者一次只处理一个任务，避免被压垮，水平扩展加 Worker 实例
9. **进度通知用 WebSocket/SSE**：实时推送处理进度，前端体验好，也可以用轮询
10. **监控告警是生产必备**：队列积压、处理成功率、平均时长、死信队列数、Worker 存活数

## 扩展方向

- 学习 Celery 高级特性（任务链、任务组、Chord、回调、路由）
- 探索 Redis Stream 作为消息队列（轻量级、支持消费者组）
- 学习 Kafka 作为大规模消息队列（高吞吐、持久化、流处理）
- 探索任务调度（定时任务、延迟任务、优先级队列）
- 学习分布式锁（避免重复处理、资源竞争）
- 探索工作流引擎（Airflow、Prefect、Temporal）
- 学习容器化部署（Docker、Kubernetes、Helm）
- 探索可观测性（OpenTelemetry、Jaeger、Prometheus、Grafana）
- 学习性能优化（批量处理、并发向量化、模型缓存、连接池）
- 探索成本优化（Spot 实例、自动扩缩容、按需向量化、模型选择）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/05-resume-interview/51-kb-async-pipeline

包含本文的完整异步 RAG 流水线代码（RabbitMQ + pika 版本、Celery 版本）、FastAPI 接口、Worker 消费者、Docker Compose 部署、监控指标。

---

**上一篇**：[文件解析为 md](./50_企业级知识库项目-文件解析为md.md) | **下一篇**：[全文检索链路](./52_企业级知识库项目-全文检索链路.md)
