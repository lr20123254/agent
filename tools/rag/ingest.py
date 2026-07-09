"""文档导入管道 —— 加载文档 → Parent-Child 切分 → 索引到 Chroma

支持格式: .txt .md .pdf .py .json .yaml .yml .csv

功能：
  - 首次导入：全量切分 + 索引
  - 增量导入：基于文件 Hash 检测变化，只更新有变动的文件
  - 清理同步：检测已删除的文件，自动清理索引
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from . import config
from .chunking import make_parent_child_chunks
from .vectorstore import index_chunks, delete_document, get_stats
from .vectorstore import get_client, get_or_create_collection

logger = logging.getLogger(__name__)


def _file_hash(file_path: Path) -> str:
    """计算文件内容的 MD5 哈希（用于增量检测）"""
    file_path = Path(file_path)  # 兼容 str 参数
    hasher = hashlib.md5()
    try:
        content = _read_file(file_path)
        if content:
            hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()
    except Exception:
        return ""


def _get_stored_hashes(collection_name: str = "knowledge_base") -> dict:
    """从 Chroma 中获取已存储的文件哈希表。

    Returns:
        {source_path: file_hash}
    """
    try:
        client = get_client()
        coll = get_or_create_collection(client, collection_name)
        count = coll.count()
        if count == 0:
            return {}

        all_data = coll.get(limit=count)
        hashes = {}
        for meta in all_data["metadatas"]:
            meta = meta or {}
            source = meta.get("source", "")
            fhash = meta.get("file_hash", "")
            if source and fhash:
                hashes[source] = fhash
        return hashes
    except Exception:
        return {}


def ingest_file(
    file_path: str,
    force: bool = False,
    collection_name: str = "knowledge_base",
) -> dict:
    """导入单个文件到知识库（支持增量检测）。

    Args:
        file_path: 文件路径
        force: 强制重新导入（忽略 hash 检测）
        collection_name: Chroma 集合名

    Returns:
        {"status": "ok"|"skip"|"unchanged"|"error", "file": str, "chunks": int, "message": str}
    """
    p = Path(file_path).resolve()
    if not p.exists():
        return {"status": "error", "file": str(p), "chunks": 0, "message": "文件不存在"}
    if p.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return {"status": "skip", "file": str(p), "chunks": 0, "message": f"不支持的文件类型: {p.suffix}"}

    source = str(p)
    file_hash = _file_hash(p)

    # ── 增量检测：如果文件未变化且已索引，跳过 ──
    if not force:
        stored_hashes = _get_stored_hashes(collection_name)
        if source in stored_hashes and stored_hashes[source] == file_hash:
            logger.info(f"文件未变化，跳过: {p.name}")
            return {"status": "unchanged", "file": source, "chunks": 0, "message": "文件未变化，跳过"}

    try:
        text = _read_file(p)
        if not text or len(text.strip()) < 20:
            return {"status": "skip", "file": source, "chunks": 0, "message": "文件内容为空或过短"}

        # 删除旧索引
        delete_document(source, collection_name)

        # Parent-Child 切分（metadata 中携带 file_hash）
        chunks = make_parent_child_chunks(
            text,
            metadata={"source": source, "file_hash": file_hash},
        )

        # 索引到 Chroma
        count = index_chunks(chunks, collection_name)

        # 清除 BM25 缓存，下次检索自动重建
        if count > 0:
            try:
                from .retriever import clear_bm25_cache
                clear_bm25_cache()
            except Exception:
                pass

        logger.info(f"导入成功: {p.name} → {count} 个 chunk")
        return {
            "status": "ok",
            "file": source,
            "chunks": count,
            "message": f"成功导入 {p.name}（{count} 个 chunk）",
        }

    except Exception as e:
        logger.error(f"导入失败 {file_path}: {e}")
        return {"status": "error", "file": source, "chunks": 0, "message": str(e)}


def ingest_directory(
    dir_path: str,
    force: bool = False,
    collection_name: str = "knowledge_base",
) -> List[dict]:
    """批量导入目录下所有支持的文档（支持增量检测）。

    Args:
        dir_path: 目录路径
        force: 强制重新导入所有文件
        collection_name: Chroma 集合名

    Returns:
        [{"status": ..., "file": ..., "chunks": ..., "message": ...}]
    """
    p = Path(dir_path).resolve()
    if not p.is_dir():
        return [{"status": "error", "file": str(p), "chunks": 0, "message": "目录不存在"}]

    results = []
    for ext in config.SUPPORTED_EXTENSIONS:
        for file_path in sorted(p.rglob(f"*{ext}")):
            result = ingest_file(str(file_path), force=force, collection_name=collection_name)
            results.append(result)

    # ── 清理已删除的文件索引 ──
    cleanup = _cleanup_deleted(p, results, collection_name)

    results.append({
        "status": "cleanup",
        "file": str(p),
        "chunks": cleanup,
        "message": f"清理了 {cleanup} 个已删除文件的索引",
    })

    return results


def _cleanup_deleted(
    dir_path: Path,
    results: list,
    collection_name: str = "knowledge_base",
) -> int:
    """清理已删除文件的索引"""
    indexed_sources = set()
    for r in results:
        if r["status"] in ("ok", "unchanged"):
            indexed_sources.add(r["file"])

    stored_hashes = _get_stored_hashes(collection_name)
    cleaned = 0
    for source in stored_hashes:
        source_path = Path(source)
        try:
            source_path.relative_to(dir_path)
        except ValueError:
            continue  # 不在本目录下，跳过

        if source not in indexed_sources and not source_path.exists():
            deleted = delete_document(source, collection_name)
            if deleted:
                logger.info(f"清理已删除文件索引: {source_path.name}")
                cleaned += 1

    return cleaned


def _read_file(path: Path) -> Optional[str]:
    """读取文件内容，支持 txt/md/pdf，扫描件 PDF 自动 OCR"""
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return _read_pdf(path)
        elif suffix in (".jpg", ".jpeg", ".png"):
            return _ocr_image(path)
        else:
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"读取失败 {path}: {e}")
        return None


def _read_pdf(path: Path) -> str:
    """读取 PDF 文件（文本型直接提取，扫描件自动 OCR 识别）"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    num_pages = len(reader.pages)

    # 先尝试提取文本
    text_parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)

    extracted_text = "\n\n".join(text_parts)
    total_chars = len(extracted_text.strip())

    # 判断是否为扫描件：每页平均字符数 < 50 则视为扫描件
    threshold_per_page = 50
    if num_pages > 0 and total_chars // num_pages < threshold_per_page:
        logger.info(f"PDF 文本内容过少 (平均 {total_chars//max(num_pages,1)} 字符/页)，"
                    f"启用 OCR 识别: {path.name}")
        return _ocr_pdf(path, reader, num_pages)

    return extracted_text


def _ocr_pdf(path: Path, reader, num_pages: int) -> str:
    """用 OCR 扫描 PDF 每页图片"""
    ocr = _get_ocr_reader()
    all_text = []

    for page_num in range(num_pages):
        page = reader.pages[page_num]
        # 尝试用 pypdf 提取嵌入式图片
        try:
            for img_idx, image in enumerate(page.images):
                img_data = image.data
                # OCR 识别图片中的文字
                result = ocr.readtext(img_data, detail=0, paragraph=True)
                if result:
                    all_text.append(f"\n--- 第 {page_num + 1} 页 (图片 {img_idx + 1}) ---")
                    all_text.extend(result)
        except Exception:
            pass

    text = "\n".join(all_text)
    logger.info(f"OCR 识别完成: {path.name}, {len(text)} 字符")
    return text


def _ocr_image(path: Path) -> str:
    """OCR 识别单张图片文件"""
    ocr = _get_ocr_reader()
    result = ocr.readtext(str(path), detail=0, paragraph=True)
    return "\n".join(result) if result else ""


_ocr_reader = None


def _get_ocr_reader():
    """延迟加载 OCR 模型（首次调用时下载 ~100MB 模型）"""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("加载 OCR 模型 (中英文)...")
        _ocr_reader = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=False,         # CPU 模式，兼容性好
            model_storage_directory=str(config.RAG_DIR / "ocr_models"),
            download_enabled=True,
        )
        logger.info("OCR 模型加载完成")
    return _ocr_reader


def get_kb_stats() -> dict:
    """获取知识库统计"""
    return get_stats()