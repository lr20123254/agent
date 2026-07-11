"""文件读写工具 - 读写本地文件 + OCR 图片文字识别"""

from pathlib import Path
from langchain.tools import tool


@tool
def read_file(path: str) -> str:
    """
    读取指定文件的内容。路径可以是绝对路径或相对于当前工作目录的路径。
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"文件不存在: {path}"
        if not p.is_file():
            return f"这不是一个文件: {path}"
        content = p.read_text(encoding="utf-8")
        return f"--- {p} ---\n{content}"
    except Exception as e:
        return f"读取文件失败: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。如果文件不存在则创建，存在则覆盖。
    参数:
      path: 文件路径
      content: 要写入的内容
    """
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"成功写入文件: {p} ({len(content)} 字符)"
    except Exception as e:
        return f"写入文件失败: {e}"


@tool
def ocr_image(path: str) -> str:
    """
    【OCR 图片文字识别】识别图片（JPG/PNG）或 PDF 文件中的文字。
    当你需要读取截图、扫描件、图片中的文字内容时使用此工具。
    支持：图片文件（JPG, PNG）、PDF 文件（含扫描件）。
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"文件不存在: {path}"

        suffix = p.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png"):
            from tools.rag.ingest import _get_ocr_reader
            ocr = _get_ocr_reader()
            import numpy as np
            from PIL import Image
            img = Image.open(str(p))
            img_array = np.array(img)
            result = ocr.readtext(img_array, detail=0, paragraph=True)
            text = "\n".join(result) if result else ""
            return f"--- OCR 识别结果: {p.name} ---\n{text}" if text else f"未从图片中识别出文字: {path}"

        elif suffix == ".pdf":
            from tools.rag.ingest import _read_pdf
            text = _read_pdf(p)
            return f"--- {p.name} 内容 ---\n{text}" if text.strip() else f"未能读取 PDF 内容: {path}"

        else:
            return f"不支持的文件类型: {suffix}，仅支持 JPG/PNG/PDF"

    except Exception as e:
        return f"OCR 识别失败: {e}"