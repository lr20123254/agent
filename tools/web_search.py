"""联网搜索工具 - 使用 DuckDuckGo"""

from langchain.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索信息，返回最新的搜索结果。
    当你需要实时信息、最新新闻、或不了解的知识时使用。
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "未找到相关结果。"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            snippet = r.get("body", "无摘要")
            link = r.get("href", "")
            formatted.append(f"{i}. {title}\n   {snippet}\n   来源: {link}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"搜索失败: {e}"