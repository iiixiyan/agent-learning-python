# LangChain 全部 Splitter 详解：其实只需要其中一个

> **Python 版** | 基于 FastAPI + LangChain Python 技术栈

---

## 为什么需要了解各种 Splitter？

上节我们学了 Loader 和 Splitter。

知识可能有各种来源，比如一个视频、一个 PDF、一个网页、一个 Word 文档。这时候就需要通过各种 Loader 从中提取信息，把它们转换成 Document。

但是 Document 可能会很大，需要用 Splitter 分割成一个个比较小的 Document（Chunk），之后用嵌入模型把分块的文档向量化后存入向量数据库。

```
各种知识来源 → Loader → 大 Document → Splitter → 多个小 Document(Chunk) → 嵌入模型向量化 → 向量数据库
```

这节我们把所有的 Splitter 过一遍，最后告诉你哪个最好用。

## 核心概念：Separator 和 Chunk Size

首先要区分 **Separator** 和 **Chunk Size** 的概念。

比如我们这样分割 Document：

```
原始文本："句子一。句子二。句子三。句子四。句子五。"

第一步：按照 Separator "。" 分割
→ ["句子一", "句子二", "句子三", "句子四", "句子五"]

第二步：按照 Chunk Size 组合成 Chunk
→ Chunk 1: "句子一。句子二。"
→ Chunk 2: "句子三。句子四。"
→ Chunk 3: "句子五。"
```

如果分割后还是大于 Chunk Size，就需要按照后面的 Separator 继续分割，然后加上 Overlap：

```
原始长文本超过 Chunk Size
→ 先用 Separator1 分割 → 还是太大
→ 再用 Separator2 分割 → 还是太大
→ 再用 Separator3 分割 → 合适了
→ 分成两个 Chunk，中间加 Overlap
```

### Overlap（重叠）

注意：**Overlap 只有文本超过 Chunk Size、文本被打断了才会加**，不是所有的块都会有 Overlap。

比如上面那段话超过了 Chunk Size，分割到两个 Chunk 里，第二个 Chunk 就会按照设置重复一部分内容。

```
Chunk 1: [内容A 内容B 内容C]
Chunk 2:         [内容C 内容D 内容E]
                    ↑ Overlap 部分
```

设置这个是为了保证语义连贯性。通常设置为 Chunk Size 的 10% - 20%。牺牲了一点存储空间（因为数据重复了），换取了模型对上下文理解的完整性。

## LangChain 有哪些 Splitter？

我们看下 `langchain_text_splitters` 这个包，所有的 Splitter 以及它们的继承关系：

```
TextSplitter（基类）
├── CharacterTextSplitter
├── RecursiveCharacterTextSplitter
│   ├── MarkdownTextSplitter
│   ├── LatexTextSplitter
│   ├── PythonCodeTextSplitter
│   └── ...（各种语言的代码分割器）
└── TokenTextSplitter
```

所有的 Splitter 都继承自 `TextSplitter`，包括 `RecursiveCharacterTextSplitter` 等。

而 `MarkdownTextSplitter`、`LatexTextSplitter` 又继承自 `RecursiveCharacterTextSplitter`。

其实很容易理解：

| Splitter | 分割方式 |
|----------|----------|
| CharacterTextSplitter | 按照某个字符来分割，比如按照句号 |
| RecursiveCharacterTextSplitter | 递归分割，比如"。？！"就是先尝试按照。分割，如果分割后大于 Chunk 剩余空间再按照？分割，是一个递归过程 |
| MarkdownTextSplitter | 按照 #、##、### 等一级级标题来递归分割 |
| LatexTextSplitter | 按照数学公式语法来递归分割 |
| TokenTextSplitter | 按照 Token 数来分割 |

## Token 是什么？

在讲 TokenTextSplitter 之前，先了解一下 Token。

我们按照字符分割，分割出来的文档的 Token 大小是不一定的。

Token 是大模型输入的一个单位，可能一个单词是 1 到 2 个 Token：
- `apple` 是 1 个 Token
- `pineapple` 是 2 个 Token
- `苹果` 是 1-2 个 Token

我们用 Python 的 `tiktoken` 包来测试一下，它是 OpenAI 模型的分词器。

### 安装依赖

```bash
pip install tiktoken langchain langchain-text-splitters langchain-core
```

### 测试 Token 数量

创建 `src/tiktoken_test.py`：

```python
"""
测试 tiktoken 分词器，了解 Token 和字符的关系
"""
import tiktoken

# 获取 gpt-4 模型的编码名称
encoding_name = tiktoken.encoding_name_for_model("gpt-4")
print(f"gpt-4 的编码名称: {encoding_name}")

# 使用 cl100k_base 编码
enc = tiktoken.get_encoding("cl100k_base")

# 测试不同文本的 Token 数量
test_texts = [
    "apple",
    "pineapple",
    "苹果",
    "吃饭",
    "一二三",
    "Hello, World!",
    "这是一段中文测试文本",
]

print("\n各文本的 Token 数量:")
for text in test_texts:
    token_count = len(enc.encode(text))
    char_count = len(text)
    print(f"  '{text}': {char_count} 字符, {token_count} Token")
```

运行：

```bash
python src/tiktoken_test.py
```

可以看到，字符和 Token 数量并没有一个确定的关系，与不同模型的分词器有关。

这样我们按照字符数来计算 Chunk Size 就没法准确估算 Token 大小。对于需要精准控制 Token 数量的场景就不大合适了。这时候就可以用 TokenTextSplitter，它是按照 Token 数来分割的。

## 实战：各种 Splitter 的 Python 实现

### 1. CharacterTextSplitter

创建 `src/character_splitter_test.py`：

```python
"""
CharacterTextSplitter 示例：按照指定字符分割
"""
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# 测试文本：一段日志
log_text = """[2024-01-15 10:00:00] INFO: Application started
[2024-01-15 10:00:05] DEBUG: Loading configuration file
[2024-01-15 10:00:10] INFO: Database connection established
[2024-01-15 10:00:15] WARNING: Rate limit approaching
[2024-01-15 10:00:20] ERROR: Failed to process request
[2024-01-15 10:00:25] INFO: Retrying operation
[2024-01-15 10:00:30] SUCCESS: Operation completed"""

log_document = Document(page_content=log_text)

# 创建 CharacterTextSplitter
# 按照换行符分割，每个块 200 字符，重叠 20 字符
log_text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
)

# 分割文档
split_documents = log_text_splitter.split_documents([log_document])

# 使用 tiktoken 计算 Token 数量
enc = tiktoken.get_encoding("cl100k_base")

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    token_len = len(enc.encode(doc.page_content))
    print(f"[分块 {i + 1}] {char_len} 字符, {token_len} Token")
    print(f"内容: {doc.page_content[:80]}...")
    print()
```

运行：

```bash
python src/character_splitter_test.py
```

可以看到，按照换行符分割文本，然后按照 Chunk Size 放到了多个块里。

有同学可能会问，Chunk 的大小也没有到 200 啊？

因为 Splitter 会优先保证语义完整，宁愿 Chunk 小一点。这里到了 160 左右字符的时候，发现加上下一个文本就超过 200 了，所以会放到下一个块。

这里因为没有被断开的文本，所以就没有需要加 Overlap 重复的。**只有被断开的文本才有 Overlap**。

### CharacterTextSplitter 的问题

我们加一个长的文本试一下：

```python
# 在上面的 log_text 末尾追加一段超长文本
log_text += """
[2026-01-10 14:30:00] INFO: 系统开始执行大规模数据迁移任务，本次迁移涉及核心业务数据库中的用户表、订单表、商品库存表、物流信息表、支付记录表、评论数据表等共计十二个关键业务表，预计处理数据量约500万条记录，数据总大小预估为280GB，迁移过程将采用分批次增量更新策略以减少对生产环境的影响，同时启用双写机制确保数据一致性，任务预计总耗时约3小时15分钟，迁移完成后将自动触发全面的数据一致性校验流程以及性能基准测试，请相关运维人员和DBA团队密切关注系统资源使用情况、网络带宽占用率以及任务执行进度，如遇异常情况请立即启动应急预案并通知技术负责人"""
```

看到问题了么？

**CharacterTextSplitter 非常死板**，你告诉它按照换行符分割，它就会严格按照这个，就算超过了 Chunk Size 也不拆分。

所以一般还是用 RecursiveCharacterTextSplitter。

---

### 2. RecursiveCharacterTextSplitter（推荐）

创建 `src/recursive_splitter_test.py`：

```python
"""
RecursiveCharacterTextSplitter 示例：递归字符分割（最常用）
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# 测试文本：一段日志 + 超长文本
log_text = """[2024-01-15 10:00:00] INFO: Application started
[2024-01-15 10:00:05] DEBUG: Loading configuration file
[2024-01-15 10:00:10] INFO: Database connection established
[2024-01-15 10:00:15] WARNING: Rate limit approaching
[2024-01-15 10:00:20] ERROR: Failed to process request
[2024-01-15 10:00:25] INFO: Retrying operation
[2024-01-15 10:00:30] SUCCESS: Operation completed
[2026-01-10 14:30:00] INFO: 系统开始执行大规模数据迁移任务，本次迁移涉及核心业务数据库中的用户表、订单表、商品库存表、物流信息表、支付记录表、评论数据表等共计十二个关键业务表，预计处理数据量约500万条记录，数据总大小预估为280GB，迁移过程将采用分批次增量更新策略以减少对生产环境的影响，同时启用双写机制确保数据一致性，任务预计总耗时约3小时15分钟，迁移完成后将自动触发全面的数据一致性校验流程以及性能基准测试，请相关运维人员和DBA团队密切关注系统资源使用情况、网络带宽占用率以及任务执行进度，如遇异常情况请立即启动应急预案并通知技术负责人"""

log_document = Document(page_content=log_text)

# 创建 RecursiveCharacterTextSplitter
# 指定多个分隔符：先尝试换行符，再尝试句号，最后尝试逗号
log_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n", "。", "，", " ", ""],
)

# 分割文档
split_documents = log_text_splitter.split_documents([log_document])

# 使用 tiktoken 计算 Token 数量
enc = tiktoken.get_encoding("cl100k_base")

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    token_len = len(enc.encode(doc.page_content))
    print(f"[分块 {i + 1}] {char_len} 字符, {token_len} Token")
    print(f"内容: {doc.page_content[:80]}...")
    print()
```

运行：

```bash
python src/recursive_splitter_test.py
```

它可以指定多个分隔符：

```
separators = ["\n", "。", "，", " ", ""]

分割过程：
1. 先用 "\n" 分割 → 如果某块还是太大
2. 再用 "。" 分割 → 如果某块还是太大
3. 再用 "，" 分割 → 如果某块还是太大
4. 再用 " " 分割 → 如果某块还是太大
5. 最后用 "" 分割（按字符强制分割）
```

这样就明显好很多。前两段文本是用换行符分割的。按照换行符分割后下面的文本超过 Chunk Size，就会尝试按照句号逗号分割，然后加上 Overlap。

所以说 RecursiveCharacterTextSplitter 这种递归的方式灵活太多了。**绝大多数情况下，用这个就可以了**。

---

### 3. TokenTextSplitter

创建 `src/token_splitter_test.py`：

```python
"""
TokenTextSplitter 示例：按照 Token 数分割
"""
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
import tiktoken

# 测试文本：一段日志
log_text = """[2024-01-15 10:00:00] INFO: Application started
[2024-01-15 10:00:05] DEBUG: Loading configuration file
[2024-01-15 10:00:10] INFO: Database connection established
[2024-01-15 10:00:15] WARNING: Rate limit approaching
[2024-01-15 10:00:20] ERROR: Failed to process request
[2024-01-15 10:00:25] INFO: Retrying operation
[2024-01-15 10:00:30] SUCCESS: Operation completed"""

log_document = Document(page_content=log_text)

# 创建 TokenTextSplitter
# 每个块最多 50 个 Token，块之间重叠 10 个 Token
log_text_splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    encoding_name="cl100k_base",  # OpenAI 使用的编码方式
)

# 分割文档
split_documents = log_text_splitter.split_documents([log_document])

# 使用 tiktoken 计算 Token 数量
enc = tiktoken.get_encoding("cl100k_base")

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    token_len = len(enc.encode(doc.page_content))
    print(f"[分块 {i + 1}] {char_len} 字符, {token_len} Token")
    print(f"内容: {doc.page_content[:80]}...")
    print()
```

运行：

```bash
python src/token_splitter_test.py
```

可以看到，它优先保证 Token 正好是 50，为了这个不惜强行打断文本。当然，打断后也加了 Overlap。

### TokenTextSplitter 的问题

- RecursiveCharacterTextSplitter 分出的 Chunk 可能大于 Chunk Size，也可以小，优先保证语义完整，是按照分割符来分割
- 但是 TokenTextSplitter 不是，它只会保证 Token 数量

这种不管不顾的分割显然不靠谱，不一定在什么地方就断开了。还是 RecursiveCharacterTextSplitter 那种更科学。

### 最佳方案：RecursiveCharacterTextSplitter + Token 长度计算

那能不能用 RecursiveCharacterTextSplitter 的分割方式，然后按照 Token 长度来设置 Chunk Size 呢？

可以的。重写一下它的长度计算函数就可以了：

```python
"""
RecursiveCharacterTextSplitter + Token 长度计算
结合递归分割的灵活性和 Token 计数的精准性
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# 测试文本
log_text = """[2024-01-15 10:00:00] INFO: Application started
[2024-01-15 10:00:05] DEBUG: Loading configuration file
[2024-01-15 10:00:10] INFO: Database connection established
[2024-01-15 10:00:15] WARNING: Rate limit approaching
[2024-01-15 10:00:20] ERROR: Failed to process request
[2024-01-15 10:00:25] INFO: Retrying operation
[2024-01-15 10:00:30] SUCCESS: Operation completed"""

log_document = Document(page_content=log_text)

# 初始化分词器
enc = tiktoken.get_encoding("cl100k_base")

# 创建 RecursiveCharacterTextSplitter，使用 Token 作为长度计算依据
log_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,        # 现在指的是 Token 数量
    chunk_overlap=20,       # Token 重叠数量
    separators=["\n", "。", "，", " ", ""],
    length_function=lambda text: len(enc.encode(text)),  # 用 Token 计算长度
)

# 分割文档
split_documents = log_text_splitter.split_documents([log_document])

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    token_len = len(enc.encode(doc.page_content))
    print(f"[分块 {i + 1}] {char_len} 字符, {token_len} Token")
    print(f"内容: {doc.page_content[:80]}...")
    print()
```

这样，Chunk Size 指的就是 Token 的长度。现在就是按照 Token 数量作为分割依据了。这样就完全不需要用 TokenTextSplitter。

---

### 4. MarkdownTextSplitter

创建 `src/markdown_splitter_test.py`：

```python
"""
MarkdownTextSplitter 示例：按照 Markdown 标题递归分割
"""
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

# 测试文本：一个 README.md
readme_text = """# Project Name

> A brief description of your project

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
pip install project-name
```

## Usage

### Basic Usage

```python
from project import Project

project = Project()
project.init()
```

### Advanced Usage

```python
project = Project(
    config={
        "api_key": "your-api-key",
        "timeout": 5000,
    }
)
project.run()
```

## API Reference

### Project

Main class for the project.

#### Methods

- init(): Initialize the project
- run(): Run the project
- stop(): Stop the project

## Contributing

Contributions are welcome!

## License

MIT License"""

readme_doc = Document(page_content=readme_text)

# 创建 MarkdownTextSplitter，不用指定分割符，内置了
markdown_text_splitter = MarkdownTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
)

# 分割文档
split_documents = markdown_text_splitter.split_documents([readme_doc])

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    print(f"[分块 {i + 1}] {char_len} 字符")
    print(f"内容:\n{doc.page_content}")
    print("-" * 60)
```

运行：

```bash
python src/markdown_splitter_test.py
```

可以看到，都是从标题处断开的，也就是根据语法分割的。

---

### 5. LatexTextSplitter

创建 `src/latex_splitter_test.py`：

```python
"""
LatexTextSplitter 示例：按照 LaTeX 数学公式语法分割
"""
from langchain_text_splitters import LatexTextSplitter
from langchain_core.documents import Document

# 测试文本：LaTeX 数学公式
latex_text = r"""
\section{Integration}

The integral of $x^n$ is:

$$
\int x^n dx = \frac{x^{n+1}}{n+1} + C
$$

\subsection{Trigonometric Integrals}

$$
\int \sin(x) dx = -\cos(x) + C
$$

$$
\int \cos(x) dx = \sin(x) + C
$$

\section{Matrix Operations}

A 3x3 matrix:

$$
\begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix}
$$

The determinant is calculated using the cofactor expansion.
"""

latex_doc = Document(page_content=latex_text)

# 创建 LatexTextSplitter
latex_text_splitter = LatexTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
)

# 分割文档
split_documents = latex_text_splitter.split_documents([latex_doc])

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    print(f"[分块 {i + 1}] {char_len} 字符")
    print(f"内容:\n{doc.page_content}")
    print("-" * 60)
```

运行：

```bash
python src/latex_splitter_test.py
```

也是按照正确的语法分割的。

---

### 6. 代码分割（Python）

创建 `src/code_splitter_test.py`：

```python
"""
代码分割示例：使用 RecursiveCharacterTextSplitter.from_language
按照编程语言语法分割代码
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 测试文本：Python 代码（购物车实现）
python_code = '''# Complete shopping cart implementation

class Product:
    def __init__(self, id, name, price, description):
        self.id = id
        self.name = name
        self.price = price
        self.description = description

    def get_formatted_price(self):
        return f"${self.price:.2f}"


class ShoppingCart:
    def __init__(self):
        self.items = []
        self.discount_code = None
        self.tax_rate = 0.08

    def add_item(self, product, quantity=1):
        existing_item = next(
            (item for item in self.items if item["product"].id == product.id),
            None
        )
        if existing_item:
            existing_item["quantity"] += quantity
        else:
            self.items.append({"product": product, "quantity": quantity})
        return self

    def remove_item(self, product_id):
        self.items = [item for item in self.items if item["product"].id != product_id]
        return self

    def calculate_subtotal(self):
        return sum(item["product"].price * item["quantity"] for item in self.items)

    def calculate_total(self):
        subtotal = self.calculate_subtotal()
        discount = self.calculate_discount()
        tax = (subtotal - discount) * self.tax_rate
        return subtotal - discount + tax

    def calculate_discount(self):
        if not self.discount_code:
            return 0
        discounts = {"SAVE10": 0.10, "SAVE20": 0.20, "WELCOME": 0.15}
        return self.calculate_subtotal() * discounts.get(self.discount_code, 0)


# Usage example
product1 = Product(1, "Laptop", 999.99, "High-performance laptop")
product2 = Product(2, "Mouse", 29.99, "Wireless mouse")
cart = ShoppingCart()
cart.add_item(product1, 1).add_item(product2, 2)
print(f"Total: ${cart.calculate_total():.2f}")
'''

python_code_doc = Document(page_content=python_code)

# 使用 from_language 静态方法，指定语言为 python
# 支持的语言有很多，包括：java、go、js、html、python、rust、swift、markdown 等
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=300,
    chunk_overlap=60,
)

# 分割文档
split_documents = code_splitter.split_documents([python_code_doc])

print(f"分割完成，共 {len(split_documents)} 个分块\n")
for i, doc in enumerate(split_documents):
    char_len = len(doc.page_content)
    print(f"[分块 {i + 1}] {char_len} 字符")
    print(f"内容:\n{doc.page_content}")
    print("-" * 60)
```

运行：

```bash
python src/code_splitter_test.py
```

用 `RecursiveCharacterTextSplitter.from_language` 这个方法，指定语言，就会按照对应的语法来分割。

支持的语言有很多，包括：java、go、js、html、python、rust、swift、markdown 等。

可以看到，完全没有破坏代码完整性，确实是按照语法分割的。

## 各种 Splitter 对比总结

| Splitter | 分割方式 | 优点 | 缺点 | 适用场景 |
|----------|----------|------|------|----------|
| CharacterTextSplitter | 按单个字符分割 | 简单直接 | 死板，超过 Chunk Size 也不拆分 | 简单文本，格式规整 |
| **RecursiveCharacterTextSplitter** | **递归多分隔符分割** | **灵活，优先保证语义完整** | **无明显缺点** | **绝大多数场景（推荐）** |
| TokenTextSplitter | 按 Token 数分割 | 精准控制 Token | 强行打断文本，破坏语义 | 需要严格计费的场景 |
| MarkdownTextSplitter | 按 Markdown 标题递归 | 保留文档结构 | 仅适用于 Markdown | Markdown 文档 |
| LatexTextSplitter | 按 LaTeX 语法递归 | 保留公式结构 | 仅适用于 LaTeX | LaTeX 数学文档 |
| 代码分割器 | 按编程语言语法递归 | 保留代码结构 | 仅适用于代码 | 代码文档 |

## 最终结论

**基本就用 RecursiveCharacterTextSplitter 就行。**

另外两个都有很明显的缺点：
- CharacterTextSplitter 的功能 RecursiveCharacterTextSplitter 里都有
- TokenTextSplitter 严格按照 Token，会破坏文档语义，不如 RecursiveCharacterTextSplitter 重写 length_function

另外几个（Markdown、Latex、代码）则是 RecursiveCharacterTextSplitter 的子功能。

## 学习要点

1. **Splitter** 是先按照 Separator 来分割，然后按照 Chunk Size 放到一个个 Chunk 里
2. Chunk 的实际大小可能小于 Chunk Size 也可以大于，优先保证语义完整
3. 如果分割后文本长度大于 Chunk Size，会继续按照后面的 Separator 拆分，然后放到两个 Chunk 里，加上 Overlap 来保证语义连贯
4. 如果从前到后尝试 Separator，尝试到最后一个，拆分完还是大于 Chunk Size 就不会再拆分了
5. 默认是按照字符计数，如果你想严格控制 Token 大小，比如需要计费的场景，就可以实现 length_function 用 Token 的方式计算长度
6. RecursiveCharacterTextSplitter 还支持代码分割，用 from_language 的静态方法，这个在处理代码文档的时候很有用
7. **虽然这节讲了很多，但是结论很简单，就是用 RecursiveCharacterTextSplitter 就好了**

## 扩展方向

- 探索语义分割（Semantic Chunking）
- 学习父文档检索器（Parent Document Retriever）
- 优化 Chunk Size 和 Overlap 参数
- 结合元数据过滤检索结果
- 下节会学习各种 Loader 的详细用法

---

## 配套代码库

**代码库地址**：https://github.com/iiixiyan/agent-learning-code/tree/main/01-agent-basics/10-all-splitters

包含本文的完整可运行代码示例（7种 Splitter 的 Python 实现 + 对比测试）。

---

**上一篇**：[知识库的 Loader 和 Splitter](./09_知识库的loader和splitter.md) | **下一篇**：[LangChain 全部 Loader 详解](./11_LangChain全部Loader.md)
