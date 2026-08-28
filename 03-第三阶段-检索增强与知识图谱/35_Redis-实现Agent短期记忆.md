# Redis：实现 Agent 短期记忆存储的最佳方案

> **Python 版** | 基于 Redis + Python redis-py + LangChain 技术栈
> 前置知识：[Agent 基础](../01-第一阶段-Agent基础入门/08_Agent基础.md)、[PostgreSQL 持久化存储](./34_PostgreSQL.md)

---

## 为什么需要短期记忆？

Agent 在对话过程中需要记住之前的对话内容，才能进行连贯的多轮对话。这就是**短期记忆**。

### 短期记忆 vs 长期记忆

| 维度 | 短期记忆 | 长期记忆 |
|------|----------|----------|
| **存储内容** | 当前对话的历史消息 | 用户偏好、项目信息、知识库 |
| **生命周期** | 会话级（会话结束后可清除） | 持久化（跨会话保留） |
| **访问频率** | 高（每次对话都要读写） | 中（按需检索） |
| **存储方案** | Redis（内存数据库，速度快） | PostgreSQL/MongoDB（持久化） |
| **典型大小** | 几十到几百条消息 | 无限增长 |
| **过期策略** | TTL 自动过期 | 手动管理或定期清理 |

### 为什么选 Redis？

| 特性 | Redis | 内存变量 | PostgreSQL | 文件存储 |
|------|-------|----------|------------|----------|
| **速度** | 极快（内存） | 极快 | 较快（磁盘） | 慢（磁盘IO） |
| **持久化** | 支持（RDB/AOF） | 不支持 | 强 | 强 |
| **过期策略** | 原生 TTL | 需手动实现 | 需手动实现 | 需手动实现 |
| **数据结构** | 丰富（String/List/Hash/Set/ZSet） | 简单 | 关系表 | 文本 |
| **并发安全** | 原子操作 | 不安全 | 事务 | 不安全 |
| **分布式支持** | 天然支持 | 不支持 | 支持 | 不支持 |

**结论**：Redis 是 Agent 短期记忆的最佳方案——速度快、支持 TTL 过期、数据结构丰富、天然支持分布式部署。

---

## Redis 数据结构选择

### 对话历史存储方案对比

| 方案 | 数据结构 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| **String + JSON** | String | 简单，直接存 JSON 数组 | 每次读写都要序列化整个数组 | 小对话（<50条） |
| **List** | List | 天然支持追加、范围查询 | 不支持按消息ID查询 | 顺序对话历史 |
| **Hash** | Hash | 支持按消息ID查询、更新 | 不保证顺序 | 需要随机访问消息 |
| **Sorted Set** | ZSet | 按时间戳排序，支持范围查询 | 内存占用较高 | 需要按时间范围查询 |
| **Stream** | Stream | 原生消息流，支持消费者组 | 复杂度较高 | 高并发消息处理 |

**推荐方案**：
- **简单对话**：List（按顺序存储，LPUSH/RPUSH + LRANGE）
- **需要随机访问**：Hash（消息ID → 消息内容）
- **需要时间范围查询**：Sorted Set（时间戳作为 score）

---

## 环境准备

### Docker 启动 Redis

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: redis-agent
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

volumes:
  redis_data:
```

```bash
# 启动 Redis
docker compose up -d

# 验证连接
docker exec -it redis-agent redis-cli ping
# 输出: PONG
```

### Python 依赖

```bash
pip install redis python-dotenv langchain langchain-openai
```

---

## 核心实现

### 方案一：List 存储对话历史（推荐）

创建 `redis_memory.py`：

```python
"""
redis_memory.py - Redis 实现 Agent 短期记忆
使用 List 数据结构存储对话历史，支持 TTL 过期、滑动窗口、摘要压缩
"""
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
import redis

load_dotenv()


class RedisShortTermMemory:
    """
    Redis 短期记忆：基于 List 数据结构

    特性：
    - 按会话ID存储对话历史
    - 支持 TTL 自动过期
    - 滑动窗口（只保留最近 N 条消息）
    - 摘要压缩（对话过长时自动摘要）
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 3600,           # 默认 1 小时过期
        max_messages: int = 50,     # 最多保留 50 条消息
        summary_threshold: int = 30, # 超过 30 条触发摘要
    ):
        """
        初始化短期记忆

        Args:
            redis_url: Redis 连接 URL
            ttl: 会话过期时间（秒）
            max_messages: 最大消息数（滑动窗口）
            summary_threshold: 触发摘要的消息数阈值
        """
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        print("✅ Redis 短期记忆初始化完成")

    def _get_key(self, session_id: str) -> str:
        """获取 Redis 键名"""
        return f"agent:memory:{session_id}"

    def _get_summary_key(self, session_id: str) -> str:
        """获取摘要键名"""
        return f"agent:memory:{session_id}:summary"

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """
        添加一条消息到对话历史

        Args:
            session_id: 会话ID
            role: 角色（user/assistant/system/tool）
            content: 消息内容
            metadata: 元数据（如工具名、时间戳等）
        """
        key = self._get_key(session_id)
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }

        # 追加到 List（右侧追加）
        self.redis.rpush(key, json.dumps(message, ensure_ascii=False))

        # 滑动窗口：只保留最近 max_messages 条
        self.redis.ltrim(key, -self.max_messages, -1)

        # 设置 TTL（每次写入刷新过期时间）
        self.redis.expire(key, self.ttl)

        print(f"  [记忆] 已添加消息: {role} - {content[:50]}...")

    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            limit: 返回最近 N 条消息（None 表示全部）

        Returns:
            List[Dict]: 消息列表
        """
        key = self._get_key(session_id)

        if limit is None:
            messages = self.redis.lrange(key, 0, -1)
        else:
            messages = self.redis.lrange(key, -limit, -1)

        return [json.loads(msg) for msg in messages]

    def get_messages_for_llm(self, session_id: str, limit: int = 20) -> List[Dict]:
        """
        获取适合传给大模型的消息列表（OpenAI 格式）

        Args:
            session_id: 会话ID
            limit: 返回最近 N 条消息

        Returns:
            List[Dict]: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
        """
        history = self.get_history(session_id, limit=limit)

        # 检查是否有摘要
        summary = self.get_summary(session_id)
        messages = []

        # 如果有摘要，在开头添加摘要消息
        if summary:
            messages.append({
                "role": "system",
                "content": f"【对话历史摘要】\n{summary}"
            })

        # 添加最近的消息
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        return messages

    def get_summary(self, session_id: str) -> Optional[str]:
        """获取对话摘要"""
        return self.redis.get(self._get_summary_key(session_id))

    def set_summary(self, session_id: str, summary: str):
        """
        设置对话摘要

        Args:
            session_id: 会话ID
            summary: 摘要内容
        """
        key = self._get_summary_key(session_id)
        self.redis.set(key, summary, ex=self.ttl)
        print(f"  [记忆] 已更新对话摘要: {summary[:80]}...")

    def should_summarize(self, session_id: str) -> bool:
        """判断是否需要触发摘要压缩"""
        key = self._get_key(session_id)
        count = self.redis.llen(key)
        return count >= self.summary_threshold

    def clear(self, session_id: str):
        """清除会话的所有记忆"""
        key = self._get_key(session_id)
        summary_key = self._get_summary_key(session_id)
        self.redis.delete(key, summary_key)
        print(f"  [记忆] 已清除会话 {session_id} 的所有记忆")

    def get_session_info(self, session_id: str) -> Dict:
        """获取会话信息（消息数、TTL等）"""
        key = self._get_key(session_id)
        return {
            "session_id": session_id,
            "message_count": self.redis.llen(key),
            "ttl": self.redis.ttl(key),
            "has_summary": self.redis.exists(self._get_summary_key(session_id)) > 0,
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 初始化记忆
    memory = RedisShortTermMemory(
        ttl=3600,           # 1小时过期
        max_messages=50,     # 最多50条
        summary_threshold=30, # 30条触发摘要
    )

    session_id = "user_123_conversation_001"

    # 模拟对话
    print("\n" + "="*60)
    print("模拟多轮对话")
    print("="*60)

    conversation = [
        ("user", "你好，我想学习Python"),
        ("assistant", "你好！Python是一种简单易学的编程语言，我可以帮你学习。你想从哪里开始？"),
        ("user", "先介绍一下Python的基本语法吧"),
        ("assistant", "Python的基本语法包括：变量、数据类型、条件语句、循环、函数等。"),
        ("user", "什么是列表和元组的区别？"),
        ("assistant", "列表是可变的，用[]表示；元组是不可变的，用()表示。"),
    ]

    for role, content in conversation:
        memory.add_message(session_id, role, content)

    # 获取对话历史
    print("\n" + "="*60)
    print("获取对话历史")
    print("="*60)
    history = memory.get_history(session_id)
    for msg in history:
        print(f"  [{msg['role']}] {msg['content'][:60]}...")

    # 获取适合 LLM 的消息
    print("\n" + "="*60)
    print("获取适合 LLM 的消息（最近3条）")
    print("="*60)
    llm_messages = memory.get_messages_for_llm(session_id, limit=3)
    for msg in llm_messages:
        print(f"  [{msg['role']}] {msg['content'][:60]}...")

    # 会话信息
    print("\n" + "="*60)
    print("会话信息")
    print("="*60)
    info = memory.get_session_info(session_id)
    print(f"  会话ID: {info['session_id']}")
    print(f"  消息数: {info['message_count']}")
    print(f"  TTL: {info['ttl']} 秒")
    print(f"  有摘要: {info['has_summary']}")

    # 清除记忆（取消注释测试）
    # memory.clear(session_id)

    print("\n✅ 短期记忆演示完成")
```

### 运行示例

```bash
# 1. 创建 .env 文件
echo "REDIS_URL=redis://localhost:6379/0" > .env

# 2. 运行短期记忆演示
python redis_memory.py
```

---

### 方案二：Hash 存储（支持随机访问）

```python
"""
redis_hash_memory.py - Redis Hash 实现短期记忆
支持按消息ID查询、更新、删除
"""
import json
import time
import uuid
import redis


class RedisHashMemory:
    """基于 Hash 的短期记忆，支持随机访问消息"""

    def __init__(self, redis_url="redis://localhost:6379/0", ttl=3600):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl

    def _get_hash_key(self, session_id):
        return f"agent:hash_memory:{session_id}"

    def _get_index_key(self, session_id):
        return f"agent:hash_memory:{session_id}:index"

    def add_message(self, session_id, role, content, metadata=None):
        """添加消息，返回消息ID"""
        msg_id = str(uuid.uuid4())[:8]
        message = {
            "id": msg_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        hash_key = self._get_hash_key(session_id)
        index_key = self._get_index_key(session_id)

        # 存储消息到 Hash
        self.redis.hset(hash_key, msg_id, json.dumps(message, ensure_ascii=False))

        # 维护有序索引（List 存储消息ID，保证顺序）
        self.redis.rpush(index_key, msg_id)

        # 设置 TTL
        self.redis.expire(hash_key, self.ttl)
        self.redis.expire(index_key, self.ttl)

        return msg_id

    def get_message(self, session_id, msg_id):
        """按消息ID获取消息"""
        hash_key = self._get_hash_key(session_id)
        msg = self.redis.hget(hash_key, msg_id)
        return json.loads(msg) if msg else None

    def update_message(self, session_id, msg_id, content):
        """更新消息内容"""
        hash_key = self._get_hash_key(session_id)
        msg = self.get_message(session_id, msg_id)
        if msg:
            msg["content"] = content
            msg["updated_at"] = time.time()
            self.redis.hset(hash_key, msg_id, json.dumps(msg, ensure_ascii=False))
            return True
        return False

    def delete_message(self, session_id, msg_id):
        """删除消息"""
        hash_key = self._get_hash_key(session_id)
        index_key = self._get_index_key(session_id)
        self.redis.hdel(hash_key, msg_id)
        self.redis.lrem(index_key, 0, msg_id)

    def get_history(self, session_id, limit=None):
        """按顺序获取对话历史"""
        index_key = self._get_index_key(session_id)
        if limit:
            msg_ids = self.redis.lrange(index_key, -limit, -1)
        else:
            msg_ids = self.redis.lrange(index_key, 0, -1)

        hash_key = self._get_hash_key(session_id)
        messages = []
        for msg_id in msg_ids:
            msg = self.redis.hget(hash_key, msg_id)
            if msg:
                messages.append(json.loads(msg))
        return messages
```

---

## 摘要压缩策略

对话过长时，需要对历史消息进行摘要压缩，控制 token 消耗。

### 摘要压缩流程

```
对话历史增长
     │
     ▼
消息数 >= 阈值？
     │
   ┌─┴─┐
   否   是
   │    │
   │    ▼
   │  对旧消息生成摘要
   │    │
   │    ▼
   │  清除旧消息，保留摘要 + 最近 N 条
   │    │
   └────┘
        │
        ▼
   继续对话
```

### 完整摘要压缩实现

```python
"""
summary_compression.py - 对话摘要压缩
对话过长时自动摘要旧消息，控制 token 消耗
"""
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from redis_memory import RedisShortTermMemory

load_dotenv()


class SummaryCompressor:
    """对话摘要压缩器"""

    def __init__(self, memory: RedisShortTermMemory):
        self.memory = memory
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "qwen-plus"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0,
        )

    def compress(self, session_id: str, keep_recent: int = 10):
        """
        压缩对话历史：摘要旧消息，保留最近 N 条

        Args:
            session_id: 会话ID
            keep_recent: 保留最近 N 条消息不摘要
        """
        history = self.memory.get_history(session_id)

        if len(history) <= keep_recent:
            print(f"  [压缩] 消息数 {len(history)} <= {keep_recent}，无需压缩")
            return

        # 需要摘要的旧消息
        to_summarize = history[:-keep_recent]
        # 保留的最近消息
        recent = history[-keep_recent:]

        print(f"  [压缩] 摘要 {len(to_summarize)} 条旧消息，保留 {len(recent)} 条最近消息")

        # 生成摘要
        summary = self._generate_summary(to_summarize)

        # 更新摘要
        old_summary = self.memory.get_summary(session_id)
        if old_summary:
            # 合并旧摘要和新摘要
            combined_summary = self._merge_summaries(old_summary, summary)
            self.memory.set_summary(session_id, combined_summary)
        else:
            self.memory.set_summary(session_id, summary)

        # 清除旧消息，只保留最近消息
        key = self.memory._get_key(session_id)
        self.memory.redis.delete(key)
        for msg in recent:
            self.memory.redis.rpush(key, json.dumps(msg, ensure_ascii=False))
        self.memory.redis.expire(key, self.memory.ttl)

        print(f"  [压缩] 压缩完成，当前消息数: {self.memory.redis.llen(key)}")

    def _generate_summary(self, messages: list) -> str:
        """用大模型生成对话摘要"""
        conversation_text = "\n".join([
            f"[{msg['role']}]: {msg['content']}"
            for msg in messages
        ])

        prompt = f"""请总结以下对话的关键信息，保持简洁，不超过200字。
包含：讨论的主要话题、达成的关键结论、用户的重要需求。

对话：
{conversation_text}

摘要："""

        response = self.llm.invoke(prompt)
        return response.content

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """合并旧摘要和新摘要"""
        prompt = f"""请合并以下两个对话摘要，去重并保持简洁，不超过300字。

旧摘要：
{old_summary}

新摘要：
{new_summary}

合并后的摘要："""

        response = self.llm.invoke(prompt)
        return response.content


# 使用示例
if __name__ == "__main__":
    memory = RedisShortTermMemory(ttl=3600, max_messages=100, summary_threshold=20)
    compressor = SummaryCompressor(memory)

    session_id = "test_compression"

    # 模拟长对话（30条消息）
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"这是第 {i+1} 条消息，讨论的主题是 Python 学习。"
        memory.add_message(session_id, role, content)

    # 检查是否需要压缩
    if memory.should_summarize(session_id):
        print("\n触发摘要压缩...")
        compressor.compress(session_id, keep_recent=10)

    # 查看压缩后的状态
    info = memory.get_session_info(session_id)
    print(f"\n压缩后状态:")
    print(f"  消息数: {info['message_count']}")
    print(f"  有摘要: {info['has_summary']}")
    print(f"  摘要内容: {memory.get_summary(session_id)[:100]}...")
```

---

## LangChain 集成

### RedisChatMessageHistory

```python
"""
langchain_redis_memory.py - LangChain 集成 Redis 短期记忆
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，用中文回答问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 创建 Chain
chain = prompt | llm

# 获取会话历史的函数
def get_session_history(session_id: str):
    """
    获取 Redis 中的会话历史

    Args:
        session_id: 会话ID

    Returns:
        RedisChatMessageHistory: Redis 会话历史对象
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
        key_prefix="agent:langchain_memory:",
        ttl=3600,  # 1小时过期
    )

# 包装 Chain，添加消息历史
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 使用示例
if __name__ == "__main__":
    session_id = "user_123"

    # 第一轮对话
    print("第一轮:")
    response = chain_with_history.invoke(
        {"input": "我叫小明，我喜欢Python编程"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"  AI: {response.content}")

    # 第二轮对话（AI 应该记得用户名字）
    print("\n第二轮:")
    response = chain_with_history.invoke(
        {"input": "我叫什么名字？我喜欢什么？"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"  AI: {response.content}")

    # 查看 Redis 中的历史
    history = get_session_history(session_id)
    print(f"\nRedis 中的消息数: {len(history.messages)}")
    for msg in history.messages:
        print(f"  [{msg.type}] {msg.content[:50]}...")
```

---

## 生产环境最佳实践

### 1. Redis 配置优化

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `maxmemory` | 256mb-1gb | 根据并发会话数调整 |
| `maxmemory-policy` | `allkeys-lru` | 内存不足时淘汰最久未使用的会话 |
| `appendonly` | `yes` | 开启 AOF 持久化，防止重启丢失数据 |
| `appendfsync` | `everysec` | 每秒同步一次，平衡性能和安全 |
| `tcp-keepalive` | `300` | TCP 保活时间，防止连接断开 |

### 2. 会话管理策略

```python
"""
session_manager.py - 会话管理最佳实践
"""
import redis
import time


class SessionManager:
    """会话管理器：活跃会话监控、过期清理、统计"""

    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def get_active_sessions(self, pattern="agent:memory:*") -> list:
        """获取所有活跃会话"""
        keys = self.redis.keys(pattern)
        sessions = []
        for key in keys:
            if key.endswith(":summary"):
                continue
            ttl = self.redis.ttl(key)
            msg_count = self.redis.llen(key)
            sessions.append({
                "session_id": key.replace("agent:memory:", ""),
                "message_count": msg_count,
                "ttl": ttl,
            })
        return sessions

    def get_stats(self) -> dict:
        """获取记忆统计"""
        memory_keys = self.redis.keys("agent:memory:*")
        summary_keys = self.redis.keys("agent:memory:*:summary")
        return {
            "total_sessions": len(memory_keys) - len(summary_keys),
            "total_summaries": len(summary_keys),
            "used_memory": self.redis.info("memory")["used_memory_human"],
        }

    def cleanup_expired(self):
        """手动清理过期会话（Redis 会自动过期，这里是兜底）"""
        keys = self.redis.keys("agent:memory:*")
        cleaned = 0
        for key in keys:
            if self.redis.ttl(key) == -1:  # 没有设置 TTL
                self.redis.delete(key)
                cleaned += 1
        return cleaned

    def force_expire(self, session_id: str):
        """强制使会话过期"""
        key = f"agent:memory:{session_id}"
        summary_key = f"agent:memory:{session_id}:summary"
        self.redis.delete(key, summary_key)
```

### 3. 安全注意事项

| 安全项 | 说明 |
|--------|------|
| **密码认证** | 生产环境必须设置 Redis 密码（`requirepass`） |
| **网络隔离** | Redis 不要暴露在公网，只允许内网访问 |
| **数据加密** | 敏感对话内容建议加密后存储 |
| **访问控制** | 按用户隔离会话，防止越权访问他人对话 |
| **数据保留** | 遵守数据隐私法规，设置合理的 TTL |

---

## 学习要点

1. **短期记忆**是 Agent 多轮对话的基础，存储当前会话的历史消息
2. **Redis** 是短期记忆的最佳方案：速度快（内存）、支持 TTL 自动过期、数据结构丰富、天然支持分布式
3. **List 数据结构**最适合存储对话历史：天然支持追加（RPUSH）、范围查询（LRANGE）、截断（LTRIM）
4. **滑动窗口**控制消息数量：只保留最近 N 条消息，防止内存溢出
5. **TTL 过期**自动清理不活跃会话：每次写入刷新 TTL，会话结束后自动过期
6. **摘要压缩**控制 token 消耗：对话过长时摘要旧消息，只保留摘要 + 最近 N 条
7. **Hash 数据结构**适合需要随机访问消息的场景：按消息ID查询、更新、删除
8. **LangChain 集成**：使用 `RedisChatMessageHistory` 快速集成 Redis 记忆
9. **生产环境**：配置 maxmemory 和淘汰策略、开启 AOF 持久化、设置密码认证、网络隔离
10. **短期记忆 vs 长期记忆**：短期用 Redis（会话级、高速），长期用 PostgreSQL/MongoDB（持久化、跨会话）

## 扩展方向

- 学习 Redis Stream 数据结构，实现高并发消息处理
- 探索 Redis Pub/Sub，实现多端实时消息同步
- 学习 Redis Cluster，实现大规模分布式记忆存储
- 探索 RedisJSON 模块，原生支持 JSON 操作和查询
- 学习 RediSearch 模块，实现对话内容的全文检索
- 探索 Redis 向量检索（RediSearch + Vector），实现短期记忆的语义检索
- 学习记忆重要性评分，自动保留重要消息、遗忘不重要的消息
- 探索多模态记忆（文本 + 图片 + 音频）的存储方案
- 学习记忆脱敏和隐私保护技术
- 探索跨会话记忆迁移（短期记忆 → 长期记忆的自动提取）

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/03-retrieval-knowledge/35-redis-short-term-memory

包含本文的完整可运行代码示例（Redis 短期记忆 + List/Hash 两种实现 + 摘要压缩 + LangChain 集成 + 会话管理）。

---

**上一篇**：[PostgreSQL 持久化存储](./34_PostgreSQL.md) | **下一篇**：[Mem0 长期记忆方案](./36_Mem0-长期记忆方案.md)
