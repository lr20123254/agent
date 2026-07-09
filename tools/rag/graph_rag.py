"""GraphRAG —— 基于知识图谱的检索增强生成

核心流程：
  1. 用 LLM 从文档 chunk 中抽取实体与关系
  2. 构建 NetworkX 知识图谱
  3. 社区检测（贪婪模块度） + 社区摘要
  4. 查询时：提取查询实体 → 定位社区 → 返回社区上下文
  5. 与向量检索双路融合，增强全局性问题回答能力

对比传统 RAG：
  - 传统 RAG：擅长具体事实查询（"chunk_size 是多少"）
  - GraphRAG：擅长全局性查询（"文档主要讨论了哪些主题？"）
"""

import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from . import config
from .retriever import _llm_instance
from .vectorstore import get_client, get_or_create_collection, decrypt_chroma_docs

logger = logging.getLogger(__name__)

# ── 图持久化路径 ────────────────────────────────
_GRAPH_CACHE = config.RAG_DIR / "graph.pkl"


# ═══════════════════════════════════════════════════
#  1. 实体抽取
# ═══════════════════════════════════════════════════


def extract_entities_from_text(text: str, chunk_id: str = "") -> Tuple[List[dict], List[dict]]:
    """用 LLM 从文本中抽取实体和关系。

    Returns:
        (entities, relationships)
        entities: [{"name": str, "type": str, "description": str}]
        relationships: [{"source": str, "target": str, "relation": str}]
    """
    if _llm_instance is None:
        return [], []

    prompt = (
        "你是一个知识图谱构建专家。请从以下文本中提取实体和它们之间的关系。\n\n"
        "## 实体类型\n"
        "- concept（概念/技术）\n"
        "- tool（工具/框架）\n"
        "- person（人物）\n"
        "- organization（组织）\n"
        "- feature（功能/特性）\n"
        "- term（术语）\n\n"
        "## 输出格式（仅返回 JSON，不要解释）\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": "实体名", "type": "concept", "description": "一句话描述"}\n'
        "  ],\n"
        '  "relationships": [\n'
        '    {"source": "实体A", "target": "实体B", "relation": "关系描述"}\n'
        "  ]\n"
        "}\n\n"
        "如果没有实体，返回空列表 {\"entities\": [], \"relationships\": []}\n\n"
        f"## 文本\n{text[:2000]}\n"
    )

    try:
        resp = _llm_instance.invoke(prompt)
        content = resp.content.strip()
        # 提取 JSON
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return [], []
        data = json.loads(json_match.group())
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        # 为实体添加 chunk_id 引用
        for e in entities:
            e["chunk_id"] = chunk_id
            e["name"] = e["name"].strip()
        for r in relationships:
            r["source"] = r["source"].strip()
            r["target"] = r["target"].strip()
        return entities, relationships
    except Exception as e:
        logger.warning(f"实体抽取失败 (chunk {chunk_id[:8]}): {e}")
        return [], []


# ═══════════════════════════════════════════════════
#  2. 图构建
# ═══════════════════════════════════════════════════


def build_graph(
    collection_name: str = "knowledge_base",
    force: bool = False,
    max_chunks: int = 50,
) -> dict:
    """从 Chroma 中读取 chunks，抽取实体关系，构建知识图谱。

    Args:
        collection_name: Chroma 集合名
        force: 强制重新构建（忽略缓存）
        max_chunks: 最多处理 chunk 数（防止 API 调用过多）

    Returns:
        {"entity_count": int, "edge_count": int, "community_count": int}
    """
    # 缓存命中
    if not force and _GRAPH_CACHE.exists():
        logger.info("图缓存命中，直接加载")
        with open(_GRAPH_CACHE, "rb") as f:
            cached = pickle.load(f)
            return {
                "entity_count": cached.graph.get("entity_count", 0),
                "edge_count": cached.graph.get("edge_count", 0),
                "community_count": cached.graph.get("community_count", 0),
                "cached": True,
            }

    if _llm_instance is None:
        return {"error": "LLM 未初始化，无法构建知识图谱"}

    G = nx.Graph()
    G.graph["entity_count"] = 0
    G.graph["edge_count"] = 0
    G.graph["community_count"] = 0

    # 从 Chroma 获取 parent chunks
    client = get_client()
    coll = get_or_create_collection(client, collection_name)
    all_data = coll.get(where={"level": "parent"}, limit=max_chunks)
    all_data = decrypt_chroma_docs(all_data)

    if not all_data["ids"]:
        logger.warning("没有找到 parent chunks，尝试全量获取")
        all_data = coll.get(limit=max_chunks)
        all_data = decrypt_chroma_docs(all_data)

    chunks = list(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]))
    logger.info(f"开始构建知识图谱，处理 {len(chunks)} 个 chunk...")

    all_entities: List[dict] = []
    all_relationships: List[dict] = []

    for idx, (cid, doc, meta) in enumerate(chunks):
        if not doc:
            continue
        logger.info(f"  [{idx+1}/{len(chunks)}] 抽取实体...")
        entities, relationships = extract_entities_from_text(doc, cid)

        for e in entities:
            G.add_node(
                e["name"],
                type=e.get("type", "concept"),
                description=e.get("description", ""),
                chunk_id=cid,
            )
            all_entities.append(e)

        for r in relationships:
            G.add_edge(
                r["source"],
                r["target"],
                relation=r.get("relation", ""),
            )
            all_relationships.append(r)

        # 避免 API 限流
        if idx < len(chunks) - 1:
            time.sleep(0.5)

    # 社区检测
    communities = _detect_communities(G)

    # 生成社区摘要
    community_summaries = _summarize_communities(G, communities, all_entities)

    # 元数据
    G.graph["entity_count"] = G.number_of_nodes()
    G.graph["edge_count"] = G.number_of_edges()
    G.graph["community_count"] = len(communities)
    G.graph["community_summaries"] = community_summaries

    # 缓存到磁盘
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_GRAPH_CACHE, "wb") as f:
        pickle.dump(G, f)

    logger.info(
        f"知识图谱构建完成: {G.number_of_nodes()} 实体, "
        f"{G.number_of_edges()} 关系, {len(communities)} 社区"
    )

    return {
        "entity_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "community_count": len(communities),
        "cached": False,
    }


def _detect_communities(G: nx.Graph) -> List[Set[str]]:
    """社区检测：贪婪模块度最大化"""
    if G.number_of_nodes() < 2:
        return [set(G.nodes())]
    try:
        communities = list(greedy_modularity_communities(G))
        return communities
    except Exception as e:
        logger.warning(f"社区检测失败: {e}")
        return [set(G.nodes())]


def _summarize_communities(
    G: nx.Graph,
    communities: List[Set[str]],
    all_entities: List[dict],
) -> Dict[int, dict]:
    """生成社区摘要：统计社区中的实体类型分布和核心主题"""
    summaries = {}
    for idx, community in enumerate(communities):
        nodes = [G.nodes[n] for n in community if n in G]
        types = {}
        for n in nodes:
            t = n.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        descriptions = [n.get("description", "") for n in nodes if n.get("description")]
        summaries[idx] = {
            "size": len(community),
            "type_distribution": types,
            "sample_entities": list(community)[:10],
            "summary": f"该社区包含 {len(community)} 个实体，"
            f"主要类型为: {', '.join(f'{k}({v})' for k, v in sorted(types.items(), key=lambda x: -x[1]))}",
        }
    return summaries


# ═══════════════════════════════════════════════════
#  3. 图检索
# ═══════════════════════════════════════════════════


def load_graph() -> Optional[nx.Graph]:
    """从缓存加载知识图谱"""
    if _GRAPH_CACHE.exists():
        with open(_GRAPH_CACHE, "rb") as f:
            return pickle.load(f)
    return None


def graph_search(query: str, top_k: int = 5) -> List[dict]:
    """图检索：从查询中提取实体 → 查找相关社区 → 返回结构化上下文。

    适合回答全局性问题（如"文档主要讨论了什么主题？"）
    而不适合具体事实查询（如"chunk_size 是多少？"）。

    Returns:
        [{"entity": str, "type": str, "context": str, "community": int, "score": float}, ...]
    """
    G = load_graph()
    if G is None or G.number_of_nodes() == 0:
        return []

    if _llm_instance is None:
        return _keyword_graph_search(query, G, top_k)

    # ── 1. 从查询中提取实体 ──
    query_entities, _ = extract_entities_from_text(query)
    query_entity_names = set(e["name"] for e in query_entities)

    if not query_entity_names:
        # 实体提取失败，回退关键词匹配
        return _keyword_graph_search(query, G, top_k)

    # ── 2. 在图里匹配实体（模糊匹配） ──
    matched_nodes: Set[str] = set()
    for qe in query_entity_names:
        for node in G.nodes():
            if qe.lower() in node.lower() or node.lower() in qe.lower():
                matched_nodes.add(node)
                break

    if not matched_nodes:
        return _keyword_graph_search(query, G, top_k)

    # ── 3. 找到这些实体所在的社区 ──
    communities = _detect_communities(G)
    node_to_community = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = ci

    affected_communities = set()
    for node in matched_nodes:
        comm_id = node_to_community.get(node)
        if comm_id is not None:
            affected_communities.add(comm_id)

    # ── 4. 从相关社区中取实体 ──
    results = []
    for ci in affected_communities:
        community = list(communities[ci])
        community_summary = (
            G.graph.get("community_summaries", {}).get(ci, {}).get("summary", "")
        )

        for node_name in community:
            node_data = G.nodes[node_name]
            score = 1.0 if node_name in matched_nodes else 0.3
            results.append({
                "entity": node_name,
                "type": node_data.get("type", "unknown"),
                "description": node_data.get("description", ""),
                "community": ci,
                "community_summary": community_summary,
                "score": round(score, 2),
            })

    # 按相关性排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _keyword_graph_search(query: str, G: nx.Graph, top_k: int) -> List[dict]:
    """关键词匹配的图检索（回退方案）"""
    query_lower = query.lower()
    query_words = set(re.findall(r"[a-zA-Z0-9_一-鿿]+", query_lower))

    scored = []
    for node_name in G.nodes(data=False):
        node_lower = node_name.lower()
        # 计算关键词匹配度
        matched = sum(1 for w in query_words if w in node_lower)
        if matched > 0:
            node_data = G.nodes[node_name]
            scored.append({
                "entity": node_name,
                "type": node_data.get("type", "unknown"),
                "description": node_data.get("description", ""),
                "community": -1,
                "community_summary": "",
                "score": round(matched / max(len(query_words), 1), 2),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def graph_stats() -> dict:
    """获取知识图谱统计"""
    G = load_graph()
    if G is None:
        return {"status": "未构建，请先执行 /graph_build"}

    return {
        "entity_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "community_count": G.graph.get("community_count", 0),
        "density": round(nx.density(G), 4),
        "communities": G.graph.get("community_summaries", {}),
    }


def format_graph_results(results: List[dict]) -> str:
    """将图检索结果格式化为可读文本"""
    if not results:
        return "知识图谱中未找到相关实体。"

    communities = set(r.get("community", -1) for r in results if r.get("community", -1) >= 0)
    output = [f"在知识图谱中找到 {len(results)} 个相关实体：\n"]

    for i, r in enumerate(results, 1):
        output.append(
            f"[{i}] {r['entity']} ({r['type']})  [评分: {r['score']}]"
        )
        if r.get("description"):
            output.append(f"    描述: {r['description']}")
        output.append("")

    if communities:
        for ci in sorted(communities):
            summary = results[0].get("community_summary", "") if results else ""
            if summary:
                output.append(f"[社区 {ci}] {summary}")
    else:
        output.append("（未检测到社区结构）")

    return "\n".join(output)