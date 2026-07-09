"""Embedding 模块 —— 本地 embedding，离线可用

采用 sentence-transformers 本地模型（BAAI/bge-small-zh-v1.5，33MB），
首次自动通过 hf-mirror.com 下载，下载后完全离线可用，零网络依赖。

策略：
  1. 强制使用 hf-mirror.com（覆盖任何默认 endpoint）
  2. 模型缓存到 .rag/models/ ，设置 HF_HUB_OFFLINE=1 避免联网检查
  3. 如果模型加载失败，自动使用回退方案（NumPy 随机投影）
  4. 小批量 + 批间延迟，防止 CPU 过载
"""

import os
import time
import logging
import hashlib
from typing import List, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)

# ── 强制设置镜像站和离线模式 ─────────────────────
# 在 import sentence_transformers 之前执行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_PREFIX"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # 第一次设为 0 以允许下载
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# 缓存目录
_MODEL_CACHE = str(config.RAG_DIR / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _MODEL_CACHE)
os.environ.setdefault("HF_HOME", str(config.RAG_DIR / "hf_home"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.RAG_DIR / "models"))


class LocalEmbeddings:
    """基于 sentence-transformers 的本地 Embedding（离线可用）"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 确保路径存在
        config.RAG_DIR.mkdir(parents=True, exist_ok=True)

        self._model_name = model_name
        self._model = None
        self._fallback = False
        self._batch_delay = config.EMBEDDING_BATCH_DELAY
        self._last_call = 0.0
        self._dim = 512  # bge-small-zh 的维度

        # 检测模型是否已缓存
        model_cache_key = "models--" + model_name.replace("/", "--")
        cache_path = os.path.join(_MODEL_CACHE, model_cache_key)
        self._is_cached = os.path.isdir(cache_path) and any(
            os.path.isfile(os.path.join(cache_path, f))
            for f in os.listdir(cache_path)[:5]
        )

        if not self._is_cached:
            logger.info(f"模型未缓存，将通过 hf-mirror.com 下载: {model_name} (~33MB)")
        else:
            logger.info(f"模型已缓存，使用离线模式: {cache_path}")

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # 如果已缓存，设置为完全离线
            if self._is_cached:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            logger.info(f"加载 Embedding 模型: {self._model_name}")
            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
                cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                local_files_only=self._is_cached,
            )

            # 兼容新旧 API
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dim = self._model.get_sentence_embedding_dimension()

            # 加载成功 → 后续使用完全离线
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info(f"模型加载完成，向量维度: {self._dim}")

        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
            if not self._is_cached:
                logger.warning("尝试联网下载（可能因网络原因失败，建议手动下载后重试）")
                try:
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self._model_name,
                        trust_remote_code=True,
                    )
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    logger.info("联网下载成功！")
                    return
                except Exception as e2:
                    logger.error(f"联网下载也失败: {e2}")

            logger.warning(f"回退到基础 Embedding 模式（维度 {self._dim}）")
            self._model = None
            self._fallback = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load_model()

        if self._fallback:
            return self._fallback_embed(texts)

        batch_size = min(config.EMBEDDING_BATCH_SIZE, max(len(texts), 1))
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            elapsed = time.time() - self._last_call
            if elapsed < self._batch_delay:
                time.sleep(self._batch_delay - elapsed)

            if "bge" in self._model_name.lower():
                prefixed = [f"为这个句子生成表示以用于检索：{t}" for t in batch]
            else:
                prefixed = batch

            embeddings = self._model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())
            self._last_call = time.time()

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        self._load_model()

        if self._fallback:
            return self._fallback_embed([text])[0]

        elapsed = time.time() - self._last_call
        if elapsed < self._batch_delay:
            time.sleep(self._batch_delay - elapsed)
        self._last_call = time.time()

        if "bge" in self._model_name.lower():
            text = f"为这个句子生成表示以用于检索：{text}"

        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """回退方案：基于文本哈希的确定性向量（无需网络、无需模型）"""
        rng = np.random.RandomState(42)  # 固定种子，保证同一文本映射到同一向量
        projection = rng.randn(self._dim)
        projection /= np.linalg.norm(projection)

        results = []
        for text in texts:
            # 用 MD5 hash 作为种子生成向量
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            seed = int(h[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            results.append(vec.tolist())

        return results

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dim


def create_embeddings(force_api: bool = False) -> LocalEmbeddings:
    """创建 Embeddings 实例"""
    return LocalEmbeddings()