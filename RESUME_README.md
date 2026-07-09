# 仿 data.txt 格式的简历项目介绍（可直接复制）

---

## 中文版（精简 + 高冲击力）

```
项目经历
通用 AI 智能体 + Agentic RAG 知识库系统  AI 应用开发  2026.04- 2026.07
项目简介：面向独立开发者多项目并行的信息管理场景，基于 LangChain Agent 框架构建通用 AI 智能体，
围绕 Agentic RAG、Hybrid Search、GraphRAG、多模态 OCR 等技术实现智能知识库系统，提升检索精度与推理效率。
技术栈：Python、LangChain、ChromaDB、BGE、BM25、NetworkX、EasyOCR、RAGAS  技术亮点：
· Agentic RAG + 双路检索引擎：设计 Agentic RAG 架构（Agent 自主决策检索时机），
  实现 BGE 向量 + BM25 + RRF + LLM Reranker 五级检索管线 + GraphRAG 知识图谱社区检
  测双路召回，BM25 缓存加速实现秒级响应
· 多模态文档解析与 Parent-Child 切分：支持文本 PDF、扫描件 PDF（OCR 自动识别）、
  图片等多格式文档，Child 块（250 token）检索 + Parent 块（1000 token）提供上下文
· 增量索引与生产级工程：MD5 哈希热更新 + Fernet 加密存储 + Agent 死循环检测 +
  LLM-as-Judge 质量评估 + Windows Unicode 容错 + 多模型热切换
```

---

## English Version

```
Project Experience
General-Purpose AI Agent with Agentic RAG  AI Application  2026.04- 2026.07
Description: Built a LangChain-based AI agent with Agentic RAG, Hybrid Search, GraphRAG,
and multi-modal OCR for indie developers managing scattered project knowledge.
Tech Stack: Python, LangChain, ChromaDB, BGE, BM25, NetworkX, EasyOCR, RAGAS  Highlights:
· Agentic RAG + Dual-Retrieval Engine: Agent autonomously decides when to search; 5-stage
  pipeline (BGE vector + BM25 + RRF + LLM Reranker + secondary retrieval) paired with
  GraphRAG community detection; BM25 caching delivers sub-second response
· Multi-Modal Parsing + Parent-Child Chunking: Auto-OCR for scanned PDFs and images;
  child chunks (250t) for retrieval, parent chunks (1000t) for full context
· Incremental Indexing & Production Engineering: MD5 hot-update + Fernet encryption +
  loop detection + LLM-as-Judge evaluation + Windows Unicode + multi-model support
```

---

## 和对方项目的对比优势

| 对比项 | 对方 MiniCode | 对方 RAG 项目 | 我们项目 |
|---|---|---|---|
| Agent 架构 | ✅ 有 | ❌ | ✅ **Agent + RAG 一体化** |
| Agentic RAG | ❌ | ❌ | ✅ **核心亮点** |
| Hybrid Search | ❌ | ✅ BM25+稠密 | ✅ **BGE+BM25+RRF+二次检索** |
| LLM Reranker | ❌ | 提及"重排"未说明 | ✅ **LLM 0-10 评分+批量处理** |
| Parent-Child | ❌ | ❌ | ✅ **独特亮点** |
| 查询改写 | ❌ | ✅ | ✅ **HyDE 风格** |
| 增量索引 | ❌ | ✅ 文档Hash | ✅ **MD5 Hash + 自动清理** |
| 评估框架 | ❌ | ✅ RAGAS | ✅ **LLM-as-Judge 四维** |
| 引用溯源 | ❌ | ✅ 引用核查 | ✅ **置信度标注+来源追踪** |
| 多模态 | ❌ | ✅ OCR+VLM | ❌ |
| 本地 Embedding | ❌ | ❌ | ✅ **BGE 33MB 零API** |
| 跨平台兼容 | ❌ | ❌ | ✅ **Unicode 编码容错** |

> 💡 **结论**：对方两个项目合起来覆盖的技术点，我们一个项目基本全覆盖，且多了 Agentic RAG、Parent-Child、本地 Embedding 等独特优势。简历上写这一个项目就够了，比写两个更聚焦、更有冲击力。