<div align="center">

# 🤖 通用 AI 智能体 + Agentic RAG

<p align="center">
  <b>LangChain · Agent · RAG · GraphRAG · Hybrid Search · ChromaDB · BGE · OCR</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/LangChain-1.3-green" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-orange" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/status-v2.0-brightgreen" />
</p>

<p align="center">
  🧠 AI 智能体自主决策 · 📚 RAG 知识库检索 · 🔗 GraphRAG 知识图谱 · 📄 多模态文档 OCR
</p>

</div>

---

## 📌 项目简介

面向独立开发者多项目并行的信息管理场景，基于 LangChain Agent 框架构建的通用 AI 智能体，集成 **Agentic RAG、Hybrid Search、GraphRAG、多模态 OCR** 等能力，让 AI 自主决定何时查知识库、何时搜网络、何时读写文件。

> 🎯 **和传统 RAG 的区别**：LLM 是"大脑"不是"流水线"——它自己判断该用什么工具，而不是固定走"检索→生成"的流程。

---

## 🏗️ 架构总览

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────┐
│           🤖 LangChain Agent                  │
│         (LLM 自主决策下一步)                    │
│                                                │
│  ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ 🌐 搜索  │ │ 📁 文件  │ │ 🧠 知识库     │ │
│  │ DuckGoGo │ │ 读写     │ │                │ │
│  └──────────┘ └──────────┘ │ ┌────────────┐ │ │
│                             │ │ 向量检索   │ │ │
│                             │ │ (BGE+BM25) │ │ │
│                             │ ├────────────┤ │ │
│                             │ │知识图谱检索│ │ │
│                             │ │ (GraphRAG) │ │ │
│                             │ ├────────────┤ │ │
│                             │ │ Reranker   │ │ │
│                             │ └────────────┘ │ │
│                             └────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## ✨ 核心能力

### 🧠 Agentic RAG（2025 前沿架构）

将知识库检索、知识图谱、联网搜索封装为 Agent Tool，LLM **自主决策** 何时检索、如何改写查询、用哪个工具。替代传统"问题→检索→回答"的固定流水线式 RAG。

### 🔍 Hybrid Search + GraphRAG 双路检索引擎

| 检索方式 | 适合场景 | 原理 |
|---|---|---|
| **BGE 向量检索** | 语义匹配、同义词 | 中文 BGE Embedding → 余弦相似度 |
| **BM25 关键词** | 精确关键词匹配 | 词频统计 + 中英文分词 |
| **RRF 融合排序** | 综合排序 | Reciprocal Rank Fusion 公式 |
| **LLM Reranker** | 去噪、二次筛选 | 大模型 0-10 分逐条评分重排 |
| **GraphRAG 社区检测** | 全局性主题分析 | NetworkX 知识图谱实体关联 |

> 📊 BM25 索引缓存加速，重复查询毫秒级响应。

### 📄 多模态文档解析

| 文件类型 | 支持方式 |
|---|---|
| **文本 PDF** | pypdf 直接提取文字 |
| **扫描件 PDF** | EasyOCR 自动识别（图片→文字） |
| **图片 JPG/PNG** | OCR 直接识别图中文字 |
| **Markdown / TXT / 代码** | 直接读取 |

> 🔄 文本不足时自动切换 OCR，无需手动干预。

### 🧩 Parent-Child Small-to-Big 切分

- **Child 块 (250 token)** → 向量检索匹配（精度高）
- **Parent 块 (1000 token)** → 提供给 LLM 生成（上下文完整）

### 🛡️ 生产级工程特性

| 特性 | 说明 |
|---|---|
| **增量索引** | MD5 文件哈希热更新，只重编变更文件 |
| **加密存储** | Fernet 对称加密，密钥本地管理 |
| **死循环检测** | Agent 反复调用同一工具时自动终止 |
| **质量评估** | LLM-as-Judge 四维评分（Faithfulness / Precision / Recall / Relevancy） |
| **Unicode 容错** | Windows 终端编码兼容 |
| **多模型切换** | 一行 .env 配置切换 GPT / Claude / DeepSeek |
| **离线部署** | BGE 模型 33MB，下载后完全离线可用 |

---

## 🚀 快速开始

### 前置条件

- Python 3.11+
- 中转站 API 密钥（OpenAI 兼容格式）

### 安装

```bash
# 克隆
git clone https://github.com/lr20123254/agent.git
cd agent

# 虚拟环境
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate          # Windows

# 依赖（含 ChromaDB、BGE、EasyOCR、NetworkX 等）
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API 密钥和模型名
```

### 启动

```bash
python main.py
```

### 导入文档

```
/ingest 文档.pdf            # 导入单个文件（自动 OCR）
/ingest_all ./docs/         # 批量导入目录
/kb                         # 查看知识库状态
/graph_build                # 构建知识图谱（从已有文档）
```

---

## 📁 项目结构

```
agent/
├── main.py                   # 主入口：Agent 初始化 + CLI
├── tools/
│   ├── web_search.py         # 🌐 联网搜索（DuckDuckGo）
│   ├── file_tools.py         # 📁 文件读写
│   └── rag/                  # 🧠 RAG 知识库模块
│       ├── config.py         #   配置参数
│       ├── embeddings.py     #   BGE 本地编码器
│       ├── chunking.py       #   Parent-Child 切分
│       ├── vectorstore.py    #   ChromaDB 向量库
│       ├── retriever.py      #   Hybrid Search 检索引擎
│       ├── rag_tool.py       #   Agentic RAG 工具
│       ├── ingest.py         #   文档导入管道
│       ├── graph_rag.py      #   🔗 GraphRAG 知识图谱
│       ├── encryption.py     #   🔒 Fernet 加密存储
│       └── eval.py           #   📊 质量评估框架
├── .env.example              # 环境变量模板
├── requirements.txt           # 依赖清单
└── RESUME_README.md           # 简历项目介绍
```

---

## 🧪 快速验证

```
/kb            # 查看知识库
/ingest README.md     # 导入项目README
graph_build    # 构建知识图谱
graph_stats    # 查看图谱统计
/kb_eval       # RAG 质量评估
```

---

## 📊 相关资源

- [简历项目介绍](RESUME_README.md) — 含中英文简历描述、面试话术、STAR 法则模板
- [技术栈详解](tools/rag/) — 各模块源码

---

## 📄 License

MIT License

---

<div align="center">
  <sub>Built with ❤️ by LR</sub>
</div>