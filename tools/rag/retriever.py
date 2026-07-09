"""混合检索器 —— Hybrid Search (向量 + BM25) + LLM Reranker + 查询改写

流程：
  0. 查询改写（LLM 扩展/改写原始查询）
  1. 向量检索 child chunk（语义匹配）
  2. BM25 检索 child chunk（关键词匹配）- 缓存 BM25 避免重复构建
  3. RRF (Reciprocal Rank Fusion) 融合排序
  4. 映射到 parent chunk（Small-to-Big）
  5. LLM Reranker 重排（用大模型判断相关性）
  6. 低置信度二次检索
  7. 返回 top-K + 引用 + 置信度
"""

import logging
import time
import re
from typing import List, Optional, Tuple

from . import config
from .chunking import _num_tokens
from .embeddings import create_embeddings
from .vectorstore import get_client, get_or_create_collection, decrypt_chroma_docs

logger = logging.getLogger(__name__)

# ── LLM 引用（从外部注入） ────────────────────────
_llm_instance = None
_bm25_cache = {"chunks_hash": "", "bm25": None, "texts": [], "ids": [], "metas": []}


def set_llm(llm) -> None:
    global _llm_instance
    _llm_instance = llm


def clear_bm25_cache() -> None:
    """清除 BM25 缓存（文档变更后调用）"""
    global _bm25_cache
    _bm25_cache = {"chunks_hash": "", "bm25": None, "texts": [], "ids": [], "metas": []}
    logger.info("BM25 缓存已清除")


# ── 查询改写 ────────────────────────────────────────


def rewrite_query(query: str) -> str:
    """用 LLM 改写/扩展查询，提升检索命中率（HyDE 思想）"""
    global _llm_instance
    if _llm_instance is None:
        return query

    prompt = (
        "你是一个专业的查询改写助手。请将用户的提问改写为更适合搜索关键词匹配的形式。\n"
        "规则：\n"
        "1. 提取核心关键词和实体\n"
        "2. 去掉语气词和冗余表述\n"
        "3. 如果是中文问题，保持中文\n"
        "4. 返回改写后的一句话查询，不要解释\n\n"
        f"原始查询: {query}\n"
        f"改写后:"
    )
    try:
        resp = _llm_instance.invoke(prompt)
        rewritten = resp.content.strip().strip('"').strip("'")
        if rewritten and rewritten != query:
            logger.info(f"查询改写: \"{query}\" → \"{rewritten[:80]}...\"")
            return rewritten
    except Exception as e:
        logger.warning(f"查询改写失败: {e}，使用原始查询")
    return query


# ── BM25（带缓存） ─────────────────────────────────


def _get_bm25(collection_name: str = "knowledge_base"):
    """获取 BM25 模型（缓存，只有 chunk 数量变化时才重建）"""
    global _bm25_cache
    from rank_bm25 import BM25Okapi

    client = get_client()
    coll = get_or_create_collection(client, collection_name)
    all_count = coll.count()

    # 检查缓存是否有效
    current_hash = f"{collection_name}:{all_count}"
    if _bm25_cache["chunks_hash"] == current_hash and _bm25_cache["bm25"] is not None:
        return _bm25_cache["bm25"], _bm25_cache["texts"], _bm25_cache["ids"], _bm25_cache["metas"]

    # 重建 BM25
    all_data = coll.get(where={"level": "child"})
    if not all_data["ids"]:
        all_data = coll.get()

    # 解密
    all_data = decrypt_chroma_docs(all_data)

    child_ids = all_data["ids"]
    child_texts = all_data["documents"]
    child_metas = all_data["metadatas"]

    if not child_ids:
        return None, [], [], []

    tokenized = [_simple_tokenize(t) for t in child_texts]
    bm25 = BM25Okapi(tokenized)

    # 缓存
    _bm25_cache = {
        "chunks_hash": current_hash,
        "bm25": bm25,
        "texts": child_texts,
        "ids": child_ids,
        "metas": child_metas,
    }
    logger.info(f"BM25 缓存重建: {len(child_ids)} child chunks")
    return bm25, child_texts, child_ids, child_metas


def _simple_tokenize(text: str) -> List[str]:
    tokens = []
    for word in re.findall(r"[a-zA-Z0-9_\.\-]+", text):
        tokens.append(word.lower())
    chars = re.findall(r"[一-鿿]", text)
    tokens.extend(chars)
    tokens.extend([chars[i] + chars[i + 1] for i in range(len(chars) - 1)])
    return tokens


def _rrf(rank_1: int, rank_2: int, k: int = 60) -> float:
    score = 0.0
    if rank_1 >= 0:
        score += config.RRF_WEIGHT / (rank_1 + k)
    if rank_2 >= 0:
        score += (1 - config.RRF_WEIGHT) / (rank_2 + k)
    return score


# ── LLM Reranker ────────────────────────────────────


def _rerank_with_llm(
    query: str,
    contexts: List[Tuple[str, str, float, str]],
    top_k: int = None,
) -> List[Tuple[str, float, str]]:
    if top_k is None:
        top_k = config.TOP_K_FINAL

    llm = _llm_instance
    if not llm or not contexts:
        return [(text, score, cid) for text, _, score, cid in contexts[:top_k]]

    scored: List[Tuple[str, float, str]] = []

    # 如果结果很少，批量打分
    for i in range(0, len(contexts), 5):
        batch = contexts[i : i + 5]
        prompt = _build_rerank_prompt(query, [(t, c) for t, c, _, _ in batch])
        try:
            resp = llm.invoke(prompt)
            scores = _parse_rerank_scores(resp.content.strip(), len(batch))
            for (text, _, _, cid), score in zip(batch, scores):
                scored.append((text, score, cid))
        except Exception as e:
            logger.warning(f"Reranker 失败: {e}，回退 RRF")
            for text, _, rrf, cid in batch:
                scored.append((text, rrf, cid))

        if i + 5 < len(contexts):
            time.sleep(0.2)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _build_rerank_prompt(query: str, contexts: List[Tuple[str, str]]) -> str:
    lines = [
        f"## 用户查询\n{query}\n",
        "## 候选段落\n请评估以下每个段落与用户查询的相关性（0 = 完全不相关, 10 = 高度相关）。",
        "只输出一行数字，用空格分隔，例如: 8 3 9 1\n",
    ]
    for idx, (parent_text, child_text) in enumerate(contexts, 1):
        excerpt = (parent_text or child_text)[:500]
        lines.append(f"[{idx}]\n{excerpt}\n")
    return "\n".join(lines)


def _parse_rerank_scores(answer: str, count: int) -> List[float]:
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", answer)
    scores = [float(n) for n in numbers if 0 <= float(n) <= 10]
    while len(scores) < count:
        scores.append(5.0)
    return scores[:count]


# ── 置信度评估 ──────────────────────────────────────


def _assess_confidence(results: List[dict]) -> Tuple[str, bool]:
    if not results:
        return "low", True
    top_score = results[0].get("score", 0)
    if top_score >= 8:
        return "high", False
    elif top_score >= 5:
        return "medium", True
    else:
        return "low", True


# ═══════════════════════════════════════════════════
#  主检索接口
# ═══════════════════════════════════════════════════


def hybrid_search(
    query: str,
    collection_name: str = "knowledge_base",
    llm=None,
    top_k: int = None,
    enable_rewrite: bool = True,
) -> List[dict]:
    """混合搜索主入口。

    llm 参数已废弃——优先使用模块级 _llm_instance
    """
    global _llm_instance
    if top_k is None:
        top_k = config.TOP_K_FINAL
    llm = _llm_instance  # 强制使用注入的 LLM

    # ── 0. 查询改写 ────────────────────────────
    original_query = query
    if enable_rewrite and llm:
        rewritten = rewrite_query(query)
        if rewritten != query:
            query = rewritten

    emb = create_embeddings()
    client = get_client()
    coll = get_or_create_collection(client, collection_name)

    all_count = coll.count()
    if all_count == 0:
        return []

    # ── 1. 获取 child chunks（通过 BM25 缓存获取，避免重复查 Chroma） ──
    bm25, child_texts, child_ids, child_metas = _get_bm25(collection_name)
    if not child_ids:
        return []

    id_to_text = dict(zip(child_ids, child_texts))
    id_to_meta = dict(zip(child_ids, child_metas))

    # ── 2. 向量检索 ────────────────────────────
    query_emb = emb.embed_query(query)
    vr = coll.query(
        query_embeddings=[query_emb],
        n_results=min(config.TOP_K_VECTOR, len(child_ids)),
        where={"level": "child"},
    )
    vector_ranked = {
        id_: idx for idx, id_ in enumerate(vr["ids"][0])
    } if vr["ids"] else {}

    # ── 3. BM25 检索（使用缓存） ────────────────
    bm25_scores = bm25.get_scores(_simple_tokenize(query))
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[: config.TOP_K_BM25]
    bm25_ranked = {child_ids[i]: idx for idx, i in enumerate(bm25_indices)}

    # ── 4. RRF 融合 ────────────────────────────
    all_candidate_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
    rrf_results = []
    for cid in all_candidate_ids:
        score = _rrf(vector_ranked.get(cid, -1), bm25_ranked.get(cid, -1))
        rrf_results.append((cid, score))
    rrf_results.sort(key=lambda x: x[1], reverse=True)

    # ── 5. 映射到 parent ──────────────────────
    seen_parents: set = set()
    parent_candidates: List[Tuple[str, str, float, str]] = []

    for cid, score in rrf_results:
        meta = id_to_meta.get(cid, {}) or {}
        parent_id = meta.get("parent_id", "")

        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_data = coll.get(ids=[parent_id])
            parent_data = decrypt_chroma_docs(parent_data)
            if parent_data["documents"]:
                parent_candidates.append((
                    parent_data["documents"][0],
                    id_to_text.get(cid, ""),
                    score,
                    cid,
                ))
        elif not parent_id and cid not in seen_parents:
            seen_parents.add(cid)
            parent_candidates.append((id_to_text.get(cid, ""), "", score, cid))

    if not parent_candidates:
        # 直接返回 child chunks
        for cid, score in rrf_results[:top_k]:
            parent_candidates.append((id_to_text.get(cid, ""), "", score, cid))

    # ── 6. Rerank ──────────────────────────────
    reranked = _rerank_with_llm(query, parent_candidates, top_k)

    # ── 7. 构建返回 ────────────────────────────
    result = []
    for text, score, chunk_id in reranked:
        meta = id_to_meta.get(chunk_id, {}) or {}
        source = meta.get("source", "")
        parent_id = meta.get("parent_id", "")
        confidence = "high" if score >= 8 else ("medium" if score >= 5 else "low")
        result.append({
            "text": text,
            "score": round(score, 2),
            "confidence": confidence,
            "source": source,
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "original_query": original_query,
        })

    return result


def simple_search(
    query: str,
    collection_name: str = "knowledge_base",
    top_k: int = None,
) -> List[dict]:
    """简化搜索 —— 仅向量检索（回退用）"""
    if top_k is None:
        top_k = config.TOP_K_FINAL

    emb = create_embeddings()
    client = get_client()
    coll = get_or_create_collection(client, collection_name)

    total = coll.count()
    if total == 0:
        return []

    query_emb = emb.embed_query(query)
    vr = coll.query(
        query_embeddings=[query_emb],
        n_results=min(top_k * 3, total),
        include=["documents", "metadatas", "distances"],
    )

    if not vr["ids"] or not vr["ids"][0]:
        return []

    # 解密
    vr = decrypt_chroma_docs(vr)

    id_to_meta = dict(zip(vr["ids"][0], vr["metadatas"][0]))
    result = []
    seen_parents = set()

    for doc, meta, dist in zip(
        vr["documents"][0],
        vr["metadatas"][0],
        vr["distances"][0],
    ):
        meta = meta or {}
        parent_id = meta.get("parent_id", "")
        source = meta.get("source", "")

        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)

        score = round(1 - dist, 3) if dist else 0
        result.append({
            "text": doc,
            "score": score,
            "confidence": "high" if score >= 0.70 else "medium",
            "source": source,
            "parent_id": parent_id,
            "chunk_id": "",
        })

    return result[:top_k]