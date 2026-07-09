"""ChromaDB 向量库管理端

提供：集合创建、文档批量索引、相似检索
支持透明加密：开启后存入时自动加密，取出后自动解密
"""

import logging
from typing import List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from . import config
from .embeddings import create_embeddings
from .encryption import encrypt_documents, decrypt_documents

logger = logging.getLogger(__name__)


def get_client() -> chromadb.PersistentClient:
    """获取 Chroma 持久化客户端"""
    return chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=False,
        ),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = "knowledge_base",
) -> chromadb.Collection:
    """获取或创建集合（不设置 embedding_func，我们手动传 embedding）"""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(name: str = "knowledge_base") -> None:
    """清空集合"""
    client = get_client()
    try:
        client.delete_collection(name)
    except ValueError:
        pass


def list_collections() -> List[str]:
    """列出所有集合"""
    client = get_client()
    return client.list_collections()


def decrypt_chroma_docs(data: dict) -> dict:
    """解密 Chroma 查询结果中的 documents 字段（透明解密）"""
    if "documents" in data and data["documents"]:
        # Chroma query 返回嵌套列表 [[doc1, doc2, ...]]
        if data["documents"] and isinstance(data["documents"][0], list):
            data["documents"] = [
                decrypt_documents(batch) for batch in data["documents"]
            ]
        else:
            data["documents"] = decrypt_documents(data["documents"])
    return data


def _chunked(lst: List, n: int):
    """分批工具"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def index_chunks(
    chunks: Sequence[dict],
    collection_name: str = "knowledge_base",
) -> int:
    """将切分好的 chunks 存入 Chroma（预计算 embeddings）。

    Args:
        chunks: [{"id", "parent_id", "level", "text", "metadata"}]
        collection_name: Chroma 集合名

    Returns:
        索引条目数
    """
    if not chunks:
        return 0

    emb = create_embeddings()
    client = get_client()
    coll = get_or_create_collection(client, collection_name)

    # 准备数据
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[dict] = []

    for chunk in chunks:
        ids.append(chunk["id"])
        texts.append(chunk["text"])
        metadatas.append({
            **chunk.get("metadata", {}),
            "level": chunk["level"],
            "parent_id": chunk["parent_id"],
        })

    # 加密文档（如启用）
    encrypted_texts = encrypt_documents(texts)

    # 预计算所有 embeddings（用明文计算，加密后存入）
    logger.info(f"正在编码 {len(texts)} 个文本...")
    try:
        all_embeddings = emb.embed_documents(texts)
    except Exception as e:
        logger.error(f"Embedding 编码失败: {e}")
        return 0

    logger.info(f"编码完成，存入 Chroma...")

    # 分批写入 Chroma（存储加密后的文本）
    count = 0
    total_batches = (len(ids) + config.EMBEDDING_BATCH_SIZE - 1) // config.EMBEDDING_BATCH_SIZE

    for batch_idx, (bid, bt, bm, be) in enumerate(zip(
        _chunked(ids, config.EMBEDDING_BATCH_SIZE),
        _chunked(encrypted_texts, config.EMBEDDING_BATCH_SIZE),
        _chunked(metadatas, config.EMBEDDING_BATCH_SIZE),
        _chunked(all_embeddings, config.EMBEDDING_BATCH_SIZE),
    )):
        coll.add(ids=bid, embeddings=be, documents=bt, metadatas=bm)
        count += len(bid)
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  写入进度: {count}/{len(ids)}")

    logger.info(f"索引完成: {count} 条")
    return count


def delete_document(
    source_path: str,
    collection_name: str = "knowledge_base",
) -> int:
    """删除指定来源文档的所有 chunk"""
    client = get_client()
    coll = get_or_create_collection(client, collection_name)
    results = coll.get(where={"source": source_path})
    if results["ids"]:
        coll.delete(ids=results["ids"])
        count = len(results["ids"])
        logger.info(f"删除 {source_path}: {count} 个 chunk")
        return count
    return 0


def get_stats(collection_name: str = "knowledge_base") -> dict:
    """获取当前知识库统计"""
    client = get_client()
    try:
        coll = get_or_create_collection(client, collection_name)
        count = coll.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "parent_chunks": 0,
                "child_chunks": 0,
                "sources": [],
                "document_count": 0,
            }

        all_meta = coll.get(limit=count)["metadatas"]
        sources = set()
        child_count = 0
        parent_count = 0
        for m in all_meta:
            if m and "level" in m:
                if m["level"] == "child":
                    child_count += 1
                else:
                    parent_count += 1
            if m and "source" in m:
                sources.add(m["source"])

        return {
            "total_chunks": count,
            "parent_chunks": parent_count,
            "child_chunks": child_count,
            "sources": sorted(sources),
            "document_count": len(sources),
        }
    except ValueError:
        return {"error": "knowledge_base 集合不存在，请先 /ingest 导入文档"}