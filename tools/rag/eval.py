"""RAG 评估框架 —— LLM-as-Judge 评估检索与生成质量

无需额外依赖，使用已有 LLM 进行评测。
通过 /kb_eval 命令触发，输出 HTML/Markdown 报告。

指标：
  1. Faithfulness（忠实度）：回答是否基于检索到的上下文
  2. Context Precision（上下文精度）：检索结果是否相关
  3. Context Recall（上下文召回）：检索结果是否覆盖了所需信息
  4. Answer Relevancy（回答相关性）：回答是否直接回应了问题

评估方式：LLM-as-Judge（0-10 分 + 分析）
"""

import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import config
from .retriever import _llm_instance, set_llm, hybrid_search, simple_search
from .vectorstore import get_client, get_or_create_collection, get_stats
from .embeddings import create_embeddings

logger = logging.getLogger(__name__)


# ── 测试集 ────────────────────────────────────────

PRESET_QUESTIONS = [
    "这个项目的主要功能是什么？",
    "技术栈用了哪些框架和工具？",
    "项目支持哪些文档格式导入？",
    "什么是 RAG？这个项目怎么实现检索的？",
    "Agentic RAG 和传统 RAG 有什么区别？",
]


def run_evaluation(
    collection_name: str = "knowledge_base",
    questions: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> dict:
    """运行 RAG 质量评估。

    Args:
        questions: 测试问题列表（默认用预置问题）
        output_file: 输出报告路径

    Returns:
        {"metrics": dict, "details": [逐题评估]}
    """
    if questions is None:
        questions = PRESET_QUESTIONS

    if _llm_instance is None:
        return {"error": "LLM 未初始化，无法进行评估"}

    stats = get_stats(collection_name)
    if stats.get("total_chunks", 0) == 0:
        return {"error": "知识库为空，请先 /ingest 导入文档后再评估"}

    llm = _llm_instance
    details = []

    logger.info(f"开始 RAG 评估，共 {len(questions)} 题...")

    for idx, question in enumerate(questions, 1):
        logger.info(f"[{idx}/{len(questions)}] 评估: {question[:50]}...")

        # 检索
        retrieval = hybrid_search(question, collection_name=collection_name, top_k=3)

        if not retrieval:
            details.append({
                "question": question,
                "contexts": [],
                "answer": "（知识库中未找到相关内容）",
                "metrics": {
                    "faithfulness": 0,
                    "context_precision": 0,
                    "context_recall": 0,
                    "answer_relevancy": 0,
                },
                "analysis": "❌ 检索失败：未找到相关上下文",
            })
            continue

        contexts = [r["text"][:800] for r in retrieval]

        # 用 LLM 生成回答
        answer = _generate_answer(llm, question, contexts)

        # LLM-as-Judge 评估
        metrics = _judge_quality(llm, question, contexts, answer)

        details.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "metrics": metrics,
        })

        time.sleep(0.5)  # 避免限流

    # 计算汇总指标
    summary = _summarize_metrics(details)

    result = {
        "metrics": summary,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "collection": collection_name,
    }

    # 输出报告
    if output_file:
        _save_report(result, output_file)
    else:
        _print_report(result)

    return result


def _generate_answer(llm, question: str, contexts: List[str]) -> str:
    """用 LLM 基于上下文生成回答"""
    prompt = (
        "基于以下检索到的上下文，回答用户的问题。\n"
        "如果上下文不足以回答问题，请如实说明。\n"
        "请用中文回答，简洁准确，引用具体信息。\n\n"
        "## 上下文\n"
        + "\n---\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
        + f"\n\n## 问题\n{question}\n\n## 回答"
    )
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        return f"（生成失败: {e}）"


def _judge_quality(
    llm,
    question: str,
    contexts: List[str],
    answer: str,
) -> dict:
    """LLM-as-Judge：对 RAG 质量进行多维评分"""
    prompt = (
        "你是一个 RAG 系统质量评估专家。请对以下 RAG 结果进行评分（0-10分），"
        "输出 JSON 格式。\n\n"
        "## 评估维度\n"
        "1. faithfulness（忠实度）：回答中的事实是否都能在上下文中找到支撑？\n"
        "2. context_precision（上下文精度）：检索到的上下文是否与问题相关？\n"
        "3. context_recall（上下文召回）：上下文是否覆盖了回答所需的关键信息？\n"
        "4. answer_relevancy（回答相关性）：回答是否直接回应了问题？\n\n"
        f"## 用户问题\n{question}\n\n"
        f"## 检索到的上下文\n" + "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        + f"\n\n## LLM 回答\n{answer}\n\n"
        "## 输出格式（只返回 JSON）\n"
        '{"faithfulness": 8, "context_precision": 7, "context_recall": 6, '
        '"answer_relevancy": 9, "analysis": "简要分析"}'
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        # 提取 JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "faithfulness": min(10, max(0, data.get("faithfulness", 5))),
                "context_precision": min(10, max(0, data.get("context_precision", 5))),
                "context_recall": min(10, max(0, data.get("context_recall", 5))),
                "answer_relevancy": min(10, max(0, data.get("answer_relevancy", 5))),
                "analysis": data.get("analysis", ""),
            }
    except Exception as e:
        logger.warning(f"Judge 解析失败: {e}")

    return {
        "faithfulness": 5,
        "context_precision": 5,
        "context_recall": 5,
        "answer_relevancy": 5,
        "analysis": "评估解析失败",
    }


def _summarize_metrics(details: List[dict]) -> dict:
    """汇总所有问题的指标"""
    dims = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    summary = {}
    for dim in dims:
        scores = [d["metrics"][dim] for d in details if dim in d["metrics"]]
        if scores:
            summary[dim] = {
                "avg": round(sum(scores) / len(scores), 2),
                "min": min(scores),
                "max": max(scores),
                "scores": scores,
            }
        else:
            summary[dim] = {"avg": 0, "min": 0, "max": 0, "scores": []}
    summary["overall"] = round(
        sum(s["avg"] for s in summary.values()) / len(summary), 2
    ) if summary else 0
    return summary


def _print_report(result: dict) -> None:
    """打印评估报告到控制台"""
    if "error" in result:
        print(f"❌ 评估失败: {result['error']}")
        return

    metrics = result["metrics"]
    print()
    print("=" * 60)
    print("  📊 RAG 质量评估报告")
    print(f"  时间: {result.get('timestamp', '')[:19]}")
    print(f"  试题数: {result['total_questions']}")
    print("=" * 60)

    print(f"\n  📈 总体评分: {metrics['overall']} / 10")
    print()
    for dim, data in metrics.items():
        if dim == "overall":
            continue
        print(f"  {_dim_label(dim)}: {data['avg']}  (min={data['min']}, max={data['max']})")
    print()

    for d in result["details"]:
        m = d["metrics"]
        avg_m = (m["faithfulness"] + m["context_precision"]
                 + m["context_recall"] + m["answer_relevancy"]) / 4
        bar = "█" * int(avg_m) + "░" * (10 - int(avg_m))
        print(f"  [{bar}] {d['question'][:50]}")
        if m.get("analysis"):
            print(f"       {m['analysis'][:80]}")

    print()


def _save_report(result: dict, file_path: str) -> None:
    """保存评估报告"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Markdown 格式
    lines = [
        "# 📊 RAG 质量评估报告\n",
        f"**时间**: {result.get('timestamp', '')[:19]}  |  "
        f"**试题数**: {result['total_questions']}  |  "
        f"**知识库**: {result.get('collection', '')}\n",
        "## 总体评分\n",
        f"**{result['metrics']['overall']} / 10**\n",
    ]
    for dim, data in result["metrics"].items():
        if dim == "overall":
            continue
        lines.append(f"- **{_dim_label(dim)}**: {data['avg']} (min={data['min']}, max={data['max']})")
    lines.append("\n## 逐题评估\n")
    for d in result["details"]:
        m = d["metrics"]
        lines.append(f"### {d['question']}\n")
        lines.append(f"- Faithfulness: {m['faithfulness']}")
        lines.append(f"- Context Precision: {m['context_precision']}")
        lines.append(f"- Context Recall: {m['context_recall']}")
        lines.append(f"- Answer Relevancy: {m['answer_relevancy']}")
        if m.get("analysis"):
            lines.append(f"- 分析: {m['analysis']}")
        lines.append(f"\n回答: {d['answer'][:200]}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"评估报告已保存: {path}")


def _dim_label(key: str) -> str:
    labels = {
        "faithfulness": "🧠 Faithfulness（忠实度）",
        "context_precision": "🎯 Context Precision（上下文精度）",
        "context_recall": "📚 Context Recall（上下文召回）",
        "answer_relevancy": "💬 Answer Relevancy（回答相关性）",
    }
    return labels.get(key, key)