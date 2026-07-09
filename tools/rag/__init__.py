"""RAG（检索增强生成）模块

核心能力：
  - Parent-Child Small-to-Big 切分
  - 查询改写（LLM HyDE 风格）
  - Hybrid Search（BGE 向量 + BM25 + RRF 融合）
  - LLM Reranker 二次重排
  - 低置信度二次检索
  - 增量索引（文件 Hash 热更新）
  - 数据加密（Fernet 对称加密，可选）
  - RAGAS 质量评估（LLM-as-Judge）
  - 引用溯源 + 置信度标注
  - ChromaDB 持久化向量库
  - Agentic RAG 工具（LangChain Tool）
"""

from . import config
from .rag_tool import knowledge_search, graph_search, set_llm
from .ingest import ingest_file, ingest_directory, get_kb_stats
from .vectorstore import reset_collection, list_collections, get_stats
from .eval import run_evaluation
from .encryption import is_enabled, key_exists, export_key, import_key
from .graph_rag import build_graph, graph_stats

__all__ = [
    "config",
    "knowledge_search",
    "graph_search",
    "set_llm",
    "ingest_file",
    "ingest_directory",
    "get_kb_stats",
    "get_stats",
    "reset_collection",
    "list_collections",
    "run_evaluation",
    "is_enabled",
    "key_exists",
    "export_key",
    "import_key",
    "build_graph",
    "graph_stats",
]