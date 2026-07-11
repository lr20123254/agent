"""Agentic RAG 工具集 —— 作为 LangChain Tool 供智能体自主调用

核心工具：
  - knowledge_search: 混合检索（向量+BM25+Reranker），适合具体事实查询
  - graph_search: 知识图谱检索，适合全局性/主题性查询

Corrective RAG 流程：
  检索 → 相关性评分 → 低分？→ 改写查询重试 → 仍低分？→ 联网搜索兜底
"""

import logging
import re

from langchain.tools import tool

from .retriever import hybrid_search, simple_search, set_llm, rewrite_query, _llm_instance
from .graph_rag import graph_search as _graph_search
from .graph_rag import format_graph_results

logger = logging.getLogger(__name__)


@tool
def knowledge_search(query: str) -> str:
    """
    从本地知识库（向量检索）中搜索已导入文档中的具体事实信息。
    仅当问题明确关于某个已知文档、代码、笔记、PDF 中的具体内容时使用。
    对于一般性知识问题（如"AI 是什么"）、时间问题（"现在几点"）、
    常识问题，不要调用此工具——知识库里没有这些内容。
    支持语义搜索和关键词搜索混合匹配。
    参数 query 请尽量详细、具体，以获得最佳检索效果。
    """
    try:
        results, chain = _corrective_search(query)

        if not results:
            return "知识库中未找到与查询相关的内容。你可以先使用 /ingest 命令导入文档。"

        # 构建输出
        top_score = results[0].get("score", 0)
        tag = "✅ 高置信度" if top_score >= 8 else (
            "⚠️ 中置信度" if top_score >= 5 else "❓ 低置信度"
        )

        output = [f"在知识库中找到 {len(results)} 条相关结果 {tag}：\n"]
        for i, r in enumerate(results, 1):
            text = r["text"].strip()
            score = r.get("score", 0)
            source = r.get("source", "")
            confidence = r.get("confidence", "medium")
            badge = {"high": "⭐⭐⭐", "medium": "⭐⭐", "low": "⭐",
                     "supplement": "📎", "web": "🌐"}.get(confidence, "⭐")
            if len(text) > 600:
                text = text[:600] + "..."
            output.append(f"[{i}] {badge} 评分: {score} | 来源: {source}")
            output.append(f"    {text}\n")

        # 显示纠正链
        if chain:
            hints = {
                "rewritten": "（已尝试查询改写，仍无满意结果）",
                "web_fallback": "（知识库无结果，以下为联网搜索结果）",
                "direct": "",
            }
            hint = hints.get(chain[-1], "")
            if hint:
                output.append(f"💡 {hint}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"knowledge_search 失败: {e}")
        return f"知识库检索出错: {e}"


# ── Corrective RAG 核心逻辑 ───────────────────────


def _grade_relevance(results: list, threshold: float = 5.0) -> bool:
    """判断检索结果是否足够相关。返回 True = 足够好，False = 需要纠正"""
    if not results:
        return False
    # 取 top-1 和 top-3 平均分
    top1 = results[0].get("score", 0)
    avg_top3 = sum(r.get("score", 0) for r in results[:3]) / min(3, len(results))
    # 至少有一条高分 或 整体平均不错
    return top1 >= threshold or avg_top3 >= threshold - 1.0


def _corrective_search(query: str) -> tuple:
    """Corrective RAG 搜索：检索 → 评分 → 改写重试 → 联网兜底

    Returns:
        (results, chain)
        results: 检索结果列表
        chain: 纠正链路标记 ["direct", "rewritten", "web_fallback"]
    """
    top_k = 5
    chain = []

    # ── 第 1 次检索 ──────────────────────────
    results = _try_search(query, top_k)
    if _grade_relevance(results):
        return results, ["direct"]

    # ── 低分 → 查询改写后重试 ────────────────
    logger.info("Corrective RAG: 检索质量不足，尝试查询改写")
    chain.append("rewritten")
    rewritten = rewrite_query(query)
    if rewritten and rewritten != query:
        results2 = _try_search(rewritten, top_k)
        if _grade_relevance(results2):
            return results2, chain

    # ── 仍低分 → 联网搜索兜底 ────────────────
    logger.info("Corrective RAG: 改写仍不足，联网搜索兜底")
    chain.append("web_fallback")
    web_results = _web_search_fallback(query)
    if web_results:
        return web_results, chain

    return [], chain


def _try_search(query: str, top_k: int) -> list:
    """尝试一次检索（hybrid → simple 两级回退）"""
    try:
        r = hybrid_search(query, top_k=top_k, enable_rewrite=False)
        if r:
            return r
    except Exception as e:
        logger.warning(f"hybrid_search 失败: {e}")
    try:
        r = simple_search(query, top_k=top_k)
        if r:
            return r
    except Exception as e:
        logger.warning(f"simple_search 失败: {e}")
    return []


def _web_search_fallback(query: str) -> list:
    """联网搜索兜底：调用 web_search 工具，包装为统一格式"""
    try:
        from tools.web_search import web_search
        result_text = web_search.invoke(query)
        # 包装为统一格式
        return [{
            "text": result_text,
            "score": 3.0,
            "confidence": "web",
            "source": "🌐 联网搜索（知识库无匹配）",
            "parent_id": "",
            "chunk_id": "",
            "original_query": query,
        }]
    except Exception as e:
        logger.warning(f"联网搜索兜底失败: {e}")
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