"""RAG 配置 —— 从 .env 读取"""

import os
from pathlib import Path

# ── 存储路径 ──────────────────────────────────────
# Chroma 向量库 & BM25 索引持久化目录
RAG_DIR = Path(os.getenv("RAG_DIR", str(Path.cwd() / ".rag")))
CHROMA_DIR = str(RAG_DIR / "chroma")

# ── Embedding（通过中转站 / OpenAI 兼容 API） ────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))     # 每批最多 10 条（防 TPM 限流）
EMBEDDING_BATCH_DELAY = float(os.getenv("EMBEDDING_BATCH_DELAY", "0.8")) # 批间延迟（秒）
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))    # 429 时重试次数

# ── 文档切分（Parent-Child / Small-to-Big） ──────
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1000"))          # parent chunk tokens
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "250"))             # child chunk tokens（检索用）
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# ── 检索 ──────────────────────────────────────────
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "10"))   # 向量检索取 top-K
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))       # BM25 检索取 top-K
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))       # RRF + Rerank 后最终给 LLM 的块数
RRF_WEIGHT = float(os.getenv("RRF_WEIGHT", "0.6"))    # RRF 中 vector 权重（BM25 = 1 - RRF_WEIGHT）

# ── 收录文档类型 ──────────────────────────────────
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".json", ".yaml", ".yml", ".csv", ".jpg", ".jpeg", ".png"}