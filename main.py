"""
通用智能体 (General-Purpose Agent)
==================================
基于 LangChain，支持联网搜索、文件读写、对话记忆。
适配中转站 API（OpenAI 兼容格式）。
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Windows 终端 Unicode 编码修复 ────────────────────
# 用户遇到 'utf-8' codec can't encode character '\udc9f': surrogates not allowed
# 原因：LLM 返回了 Windows 控制台无法处理的 Unicode 字符
# 解决：将 stdout/stderr 编码改为 utf-8，用 replace 替代无法编码的字符
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    # 方法一：TextIOWrapper（拦截 bytes→str 编码）
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream and hasattr(stream, "buffer"):
            setattr(sys, name, io.TextIOWrapper(
                stream.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            ))
    # 方法二：直接替换 write 方法（兜底 LangChain 内部 print）
    def _make_safe_writer(orig_write):
        def _safe_write(text):
            if isinstance(text, str):
                text = text.encode("utf-8", errors="replace").decode("utf-8")
            return orig_write(text)
        return _safe_write
    sys.stdout.write = _make_safe_writer(sys.stdout.write)
    sys.stderr.write = _make_safe_writer(sys.stderr.write)

from tools import web_search, read_file, write_file
from tools.rag import knowledge_search, graph_search, set_llm, ingest_file, ingest_directory, get_kb_stats, run_evaluation, build_graph, graph_stats

tools = [web_search, read_file, write_file, knowledge_search, graph_search]

# ── 模型初始化（中转站 / OpenAI 兼容） ──────────────
api_key = os.getenv("OPENAI_API_KEY", "")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

if not api_key or api_key == "sk-your-relay-api-key":
    print("❌ 未配置 API 密钥！")
    print("请编辑 .env 文件，填入中转站提供的密钥和地址。")
    print()
    print("必备字段:")
    print("  OPENAI_API_BASE  = 中转站接口地址 (如 https://api.xxx.com/v1)")
    print("  OPENAI_API_KEY   = 中转站给你的密钥")
    print("  OPENAI_MODEL_NAME = 模型名 (如 gpt-4o-mini / claude-3.5-sonnet)")
    sys.exit(1)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    api_key=api_key,
    base_url=api_base,
)

# 注入 LLM 到 RAG 模块（用于 reranker）
set_llm(llm)

print(f"  Model: {model_name}")
print(f"  Base : {api_base}")


# ── 记忆 ─────────────────────────────────────────────
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


# ── Agent ────────────────────────────────────────────
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个全能的通用智能体助手。\n"
        "你拥有以下能力：\n"
        "  - 联网搜索：获取实时信息\n"
        "  - 文件读写：读取和保存本地文件\n"
        "  - 知识库检索：查询本地文档\n\n"
        "【重要规则 - 防死循环】\n"
        "1. 每次调用工具后，如果结果不理想，最多再尝试 1 次\n"
        "2. 如果连续 2 次工具调用仍无满意结果，直接告诉用户没找到\n"
        "3. 不要反复调用同一个工具、同一个查询超过 2 次\n"
        "4. 知识库中没有的东西，就说没有，不要换着关键词反复搜\n\n"
        f"当前时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n\n"
        "请用中文回答，简洁清晰。在需要时主动使用工具。"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ── 死循环检测器 ────────────────────────────────
from langchain_core.callbacks import BaseCallbackHandler

class _LoopDetector(BaseCallbackHandler):
    """检测 Agent 死循环：同一工具相同查询 ≥2 次则终止"""
    def __init__(self):
        self._history = []
    def on_tool_start(self, serialized, input_str, **kwargs):
        sig = f"{serialized.get('name','')}:{input_str[:80]}"
        self._history.append(sig)
        if len(self._history) >= 2 and len(set(self._history[-2:])) == 1:
            raise Exception(f"检测到死循环: 反复调用同一工具，已强制终止")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="generate",
    callbacks=[_LoopDetector()],
)


# ── 命令行交互 ───────────────────────────────────────
def print_banner():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║     [AI] 通用智能体 v2.0                  ║")
    print("║     能力: 搜索 | RAG知识库 | 文件读写    ║")
    print("╚══════════════════════════════════════════╝")
    print("输入 /help 查看命令，/exit 退出")
    print()


def print_help():
    print()
    print("可用命令:")
    print("  /help         - 显示帮助")
    print("  /clear        - 清除对话历史")
    print("  /memory       - 查看当前记忆")
    print("  /tools        - 列出可用工具")
    print("  /ingest <路径>  - 导入文档到知识库（支持 .txt .md .pdf .py 等）")
    print("  /ingest_force <路径> - 强制重新导入（忽略 Hash 缓存）")
    print("  /ingest_all <目录> - 批量导入目录下所有支持格式文档")
    print("  /kb           - 查看知识库状态")
    print("  /kb_eval      - 对知识库进行质量评估（RAGAS 风格，5 道题）")
    print("  /kb_eval <N>  - 自定义评估题数（用预设题库前 N 题）")
    print("  /kb_clear     - 清空知识库")
    print("  /graph_build  - 构建知识图谱（从已有文档抽取实体关系）")
    print("  /graph_stats  - 查看知识图谱统计")
    print("  /exit         - 退出程序")
    print("  直接输入问题即可开始对话")
    print()


def main():
    print_banner()
    while True:
        try:
            user_input = input("\n[你] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[再见！]")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("[再见！]")
            break
        elif user_input == "/help":
            print_help()
            continue
        elif user_input == "/clear":
            memory.clear()
            print("对话历史已清除")
            continue
        elif user_input == "/memory":
            msgs = memory.chat_memory.messages
            print(f"当前记忆 ({len(msgs)} 条消息):")
            for m in msgs:
                role = "你" if m.type == "human" else "AI"
                content = m.content[:120]
                print(f"  {role}: {content}{'...' if len(m.content) > 120 else ''}")
            continue
        elif user_input == "/tools":
            print("可用工具:")
            for t in tools:
                print(f"  - {t.name}: {t.description.split(chr(10))[0]}")
            continue

        elif user_input.startswith("/ingest_all"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("用法: /ingest_all <目录路径>")
                continue
            dir_path = parts[1].strip()
            if not os.path.isdir(dir_path):
                print(f"目录不存在: {dir_path}")
                continue
            print(f"正在批量导入（增量模式）: {dir_path} ...")
            results = ingest_directory(dir_path, force=False)
            ok = sum(1 for r in results if r["status"] == "ok")
            unch = sum(1 for r in results if r["status"] == "unchanged")
            err = sum(1 for r in results if r["status"] == "error")
            skip = sum(1 for r in results if r["status"] == "skip")
            total_chunks = sum(r["chunks"] for r in results if r["status"] == "ok")
            print(f"导入完成: {ok} 新增 / {unch} 未变化 / {skip} 跳过 / {err} 失败, 共 {total_chunks} chunk")
            for r in results:
                if r["status"] == "error":
                    print(f"  ❌ {r['file']}: {r['message']}")
            continue

        elif user_input.startswith("/ingest_force"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("用法: /ingest_force <文件或目录路径>")
                continue
            path = parts[1].strip()
            if os.path.isdir(path):
                print(f"正在强制重新导入: {path} ...")
                results = ingest_directory(path, force=True)
                ok = sum(1 for r in results if r["status"] == "ok")
                err = sum(1 for r in results if r["status"] == "error")
                total_chunks = sum(r["chunks"] for r in results if r["status"] == "ok")
                print(f"强制导入完成: {ok} 成功 / {err} 失败, 共 {total_chunks} chunk")
            else:
                print(f"正在强制重新导入: {path} ...")
                result = ingest_file(path, force=True)
                print(f"{'✅' if result['status'] == 'ok' else '❌'} {result['message']}")
            continue

        elif user_input.startswith("/ingest"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("用法: /ingest <文件路径>")
                continue
            file_path = parts[1].strip()
            print(f"正在导入: {file_path} ...")
            result = ingest_file(file_path)
            if result["status"] == "ok":
                print(f"✅ {result['message']}")
            elif result["status"] == "skip":
                print(f"⏭️ {result['message']}")
            else:
                print(f"❌ 导入失败: {result['message']}")
            continue

        elif user_input == "/kb":
            stats = get_kb_stats()
            if "error" in stats:
                print(f"📚 知识库: {stats['error']}")
            else:
                print(f"📚 知识库统计:")
                print(f"   文档数:    {stats['document_count']}")
                print(f"   Parent块:  {stats['parent_chunks']}")
                print(f"   Child块:   {stats['child_chunks']}")
                print(f"   总chunk数: {stats['total_chunks']}")
                if stats["sources"]:
                    print(f"   来源文件:")
                    for s in stats["sources"]:
                        print(f"     - {s}")
            continue

        elif user_input == "/kb_clear":
            from tools.rag import reset_collection
            reset_collection()
            print("知识库已清空")
            continue

        elif user_input.startswith("/kb_eval"):
            parts = user_input.split(maxsplit=1)
            n_questions = 5
            if len(parts) > 1:
                flag = parts[1].strip()
                if flag.isdigit():
                    n_questions = int(flag)
                elif flag.lower() == "all":
                    from tools.rag.eval import PRESET_QUESTIONS
                    n_questions = len(PRESET_QUESTIONS)
            print(f"📊 正在运行 RAG 质量评估（{n_questions} 题，将依次调用 LLM 评分）...")
            print("   请耐心等待，每次评估需要约 30 秒 ...")
            result = run_evaluation(questions=None)
            if "error" in result:
                print(f"❌ 评估失败: {result['error']}")
            continue

        elif user_input == "/graph_build":
            print("🕸️ 正在构建知识图谱（将逐 chunk 抽取实体关系，可能需要几分钟）...")
            result = build_graph(force=True)
            if "error" in result:
                print(f"❌ 构建失败: {result['error']}")
            else:
                print(
                    f"✅ 知识图谱构建完成: "
                    f"{result['entity_count']} 实体, "
                    f"{result['edge_count']} 关系, "
                    f"{result['community_count']} 社区"
                )
                if result.get("cached"):
                    print("（使用缓存数据）")
            continue

        elif user_input == "/graph_stats":
            stats = graph_stats()
            if "status" in stats and stats["status"].startswith("未构建"):
                print(f"📊 {stats['status']}")
            else:
                print(f"📊 知识图谱统计:")
                print(f"   实体数: {stats.get('entity_count', 0)}")
                print(f"   关系数: {stats.get('edge_count', 0)}")
                print(f"   社区数: {stats.get('community_count', 0)}")
                print(f"   密度:  {stats.get('density', 0)}")
                communities = stats.get("communities", {})
                for cid, cdata in communities.items():
                    if isinstance(cdata, dict):
                        summary = cdata.get("summary", "")
                        size = cdata.get("size", 0)
                        print(f"   社区[{cid}]: {size} 实体 - {summary[:60]}...")
            continue

        try:
            print("\n[思考中...]")
            response = agent_executor.invoke({"input": user_input})
            print(f"\n[智能体] > {response['output']}")
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    main()