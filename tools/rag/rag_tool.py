"""Agentic RAG 工具集 —— 作为 LangChain Tool 供智能体自主调用

核心工具：
  - knowledge_search: 混合检索（向量+BM25+Reranker），适合具体事实查询
  - graph_search: 知识图谱检索，适合全局性/主题性查询
"""

import logging

from langchain.tools import tool

from .retriever import hybrid_search, simple_search, _assess_confidence, set_llm
from .graph_rag import graph_search as _graph_search
from .graph_rag import format_graph_results, load_graph

logger = logging.getLogger(__name__)


@tool
def knowledge_search(query: str) -> str:
    """
    【RAG 知识库检索】从本地知识库中搜索与查询相关的信息。
    当你需要回答与本地文档、项目代码、知识库相关的问题时使用此工具。
    支持语义搜索和关键词搜索混合匹配，能够从已有文档中找到精确答案。
    参数 query 请尽量详细、具体，以获得最佳检索效果。
    """
    try:
        results = _do_search(query)
        if not results:
            return "知识库中未找到与查询相关的内容。你可以先使用 /ingest 命令导入文档。"

        # 置信度评估
        confidence_level, needs_secondary = _assess_confidence(results)
        confidence_tag = {
            "high": "✅ 高置信度",
            "medium": "⚠️ 中置信度（可能不完整）",
            "low": "❓ 低置信度（建议补充搜索）",
        }.get(confidence_level, "")

        output = [
            f"在知识库中找到 {len(results)} 条相关结果{confidence_tag}：\n"
        ]

        for i, r in enumerate(results, 1):
            text = r["text"].strip()
            source = r.get("source", "")
            score = r.get("score", 0)
            confidence = r.get("confidence", "medium")

            # 置信度标签
            badge = {"high": "⭐⭐⭐", "medium": "⭐⭐", "low": "⭐",
                     "supplement": "📎"}.get(confidence, "⭐")

            if len(text) > 600:
                text = text[:600] + "..."

            output.append(f"[{i}] {badge} 评分: {score} | 来源: {source}")
            output.append(f"    {text}\n")

        # 低置信度提示
        if needs_secondary and confidence_level != "high":
            output.append(
                "💡 提示：以上结果置信度一般，建议补充更精确的查询，"
                "或结合联网搜索获取完整信息。"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"knowledge_search 失败: {e}")
        return f"知识库检索出错: {e}"


def _do_search(query: str) -> list:
    """执行搜索，优先 hybrid，回退 simple"""
    top_k = 5

    # 尝试 Hybrid Search
    try:
        results = hybrid_search(query, top_k=top_k, enable_rewrite=True)
        if results:
            logger.info(f"Hybrid search 返回 {len(results)} 条结果")
            return results
        else:
            logger.info("Hybrid search 返回空结果")
    except Exception as e:
        logger.warning(f"Hybrid search 失败: {e}")

    # 回退到纯向量检索
    try:
        results = simple_search(query, top_k=top_k)
        if results:
            logger.info(f"Simple search 返回 {len(results)} 条结果（回退）")
            return results
    except Exception as e:
        logger.warning(f"Simple search 也失败: {e}")

    # 最终尝试：用原始查询做一次最简单的向量检索
    try:
        results = simple_search(query, top_k=top_k)
        if results:
            return results
    except Exception:
        pass

    return []


@tool
def graph_search(query: str) -> str:
    """
    【知识图谱搜索】从知识图谱中搜索与查询相关的实体和主题关系。
    当你需要回答"主要讨论了什么主题"、"有哪些技术类别"、"实体间有什么关联"
    等全局性问题时使用此工具。
    适合宏观分析，不适合具体细节查询。
    参数 query 请描述你想要了解的主题或方向。
    """
    try:
        results = _graph_search(query)
        if not results:
            return (
                "知识图谱中未找到相关实体。你可以先使用 /graph_build 构建知识图谱。"
            )
        return format_graph_results(results)
    except Exception as e:
        logger.error(f"graph_search 失败: {e}")
        return f"知识图谱检索出错: {e}"