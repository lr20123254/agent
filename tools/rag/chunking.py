"""Parent-Child 切分策略（Small-to-Big Retrieval 核心）

实现：
  - Child chunk = 小块，用于语义检索（精细匹配）
  - Parent chunk = 大块，包含完整上下文（给 LLM 生成用）
  - 通过 chunk_id 映射关联 parent ↔ child
"""

import uuid
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def make_parent_child_chunks(
    text: str,
    metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """对一段文本做 Parent-Child 切分。

    返回:
      [
        {
          "id": str,
          "parent_id": str,       # Parent 的 id
          "level": "parent" | "child",
          "text": str,
          "tokens": int,
          "metadata": dict,
        },
        ...
      ]
    """
    metadata = metadata or {}

    # ── Parent 块 ─────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        length_function=_num_tokens,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    parent_chunks = parent_splitter.split_text(text)

    all_chunks = []
    parent_id_map: dict = {}  # parent_index -> parent_id

    for pi, parent_text in enumerate(parent_chunks):
        pid = _id()
        parent_id_map[pi] = pid
        parent_tokens = _num_tokens(parent_text)

        all_chunks.append({
            "id": pid,
            "parent_id": pid,          # parent 的 parent_id 指向自己
            "level": "parent",
            "text": parent_text,
            "tokens": parent_tokens,
            "metadata": {**metadata, "chunk_index": pi, "chunk_level": "parent"},
        })

        # ── Child 块（从 parent 中再切分） ─────────
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP,
            length_function=_num_tokens,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        child_texts = child_splitter.split_text(parent_text)

        for ci, child_text in enumerate(child_texts):
            # 跳过太短或几乎等于 parent 的 child（避免冗余）
            if _num_tokens(child_text) < 20:
                continue
            if child_text.strip() == parent_text.strip():
                continue

            all_chunks.append({
                "id": _id(),
                "parent_id": pid,
                "level": "child",
                "text": child_text,
                "tokens": _num_tokens(child_text),
                "metadata": {**metadata, "chunk_index": f"{pi}.{ci}", "chunk_level": "child"},
            })

    return all_chunks


def _num_tokens(text: str) -> int:
    """粗略计 token 数（中英文混合近似）"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback: 中文≈1.5 token/字，英文≈0.25 token/字母
        chinese = sum(1 for c in text if '一' <= c <= '鿿')
        english = len(text) - chinese
        return int(chinese * 1.5 + english * 0.25)


def _id() -> str:
    return uuid.uuid4().hex[:16]