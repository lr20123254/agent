"""文件读写工具 - 读写本地文件"""

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