"""
通用智能体 Web UI — Streamlit 界面
====================================
保留 CLI 全部功能，增加可视化的文件上传、对话、知识库管理。
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# 必须在任何其他导入前设置环境变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 页面配置
st.set_page_config(
    page_title="通用智能体 v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 初始化 session state ──────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ── 延迟初始化（避免 Streamlit 热重载重复执行） ──
@st.cache_resource
def init_agent():
    """初始化 LLM、Agent、RAG 模块"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    if not api_key or api_key == "sk-your-api-key-here":
        return None, "❌ 未配置 API 密钥！请编辑 .env 文件"

    from langchain_openai import ChatOpenAI
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
    from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_core.callbacks import BaseCallbackHandler

    from tools import web_search, read_file, write_file, ocr_image
    from tools.rag import knowledge_search, graph_search, set_llm

    llm = ChatOpenAI(
        model=model_name, temperature=0.7,
        api_key=api_key, base_url=api_base,
    )
    set_llm(llm)

    tools = [web_search, read_file, write_file, ocr_image, knowledge_search, graph_search]

    # 死循环检测
    class _LoopDetector(BaseCallbackHandler):
        def __init__(self):
            self._history = []
        def on_tool_start(self, serialized, input_str, **kwargs):
            sig = f"{serialized.get('name','')}:{input_str[:80]}"
            self._history.append(sig)
            if len(self._history) >= 2 and len(set(self._history[-2:])) == 1:
                raise Exception(f"检测到死循环: 反复调用同一工具，已强制终止")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
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
         "请用中文回答，简洁清晰。在需要时主动使用工具。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True,
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory,
        verbose=False, handle_parsing_errors=True,
        max_iterations=8, early_stopping_method="generate",
        callbacks=[_LoopDetector()],
    )

    return agent_executor, f"✅ 已连接: {model_name}"


# ── 侧边栏 ─────────────────────────────────────────
with st.sidebar:
    st.title("🤖 通用智能体")
    st.caption("v2.0 · Agentic RAG · 6 工具")

    # 初始化
    if not st.session_state.initialized:
        with st.spinner("正在初始化..."):
            executor, msg = init_agent()
            st.session_state.agent_executor = executor
            st.session_state.initialized = True
            if executor:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # 文件上传
    st.subheader("📄 导入文档")
    uploaded_files = st.file_uploader(
        "选择文件（PDF/图片/TXT/MD）",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md", "py", "csv", "json"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        from tools.rag import ingest_file
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(f.name).suffix
            ) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            with st.spinner(f"正在导入 {f.name}..."):
                result = ingest_file(tmp_path)
                if result["status"] == "ok":
                    st.success(f"✅ {f.name}: {result['chunks']} chunk")
                elif result["status"] == "unchanged":
                    st.info(f"⏭️ {f.name}: 未变化")
                else:
                    st.error(f"❌ {f.name}: {result['message']}")
            os.unlink(tmp_path)

    st.divider()

    # 知识库管理
    st.subheader("📚 知识库")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 查看统计", use_container_width=True):
            from tools.rag import get_kb_stats
            stats = get_kb_stats()
            if "error" in stats:
                st.warning(stats["error"])
            else:
                st.write(f"文档数: {stats['document_count']}")
                st.write(f"总 chunk: {stats['total_chunks']}")
                if stats["sources"]:
                    with st.expander("来源文件"):
                        for s in stats["sources"]:
                            st.text(s)
    with col2:
        if st.button("🗑️ 清空", use_container_width=True):
            from tools.rag import reset_collection
            reset_collection()
            st.success("已清空")

    st.divider()

    # 图谱管理
    st.subheader("🕸️ 知识图谱")
    if st.button("🔨 构建图谱", use_container_width=True):
        from tools.rag import build_graph
        with st.spinner("构建中..."):
            result = build_graph(force=True)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"{result['entity_count']} 实体, {result['edge_count']} 关系")

    if st.button("📊 图谱统计", use_container_width=True):
        from tools.rag import graph_stats
        stats = graph_stats()
        if "status" in stats and "未构建" in str(stats.get("status", "")):
            st.warning(stats["status"])
        else:
            st.write(f"实体: {stats.get('entity_count', 0)}")
            st.write(f"关系: {stats.get('edge_count', 0)}")
            st.write(f"社区: {stats.get('community_count', 0)}")

    st.divider()

    # 评估
    if st.button("📊 RAG 质量评估", use_container_width=True):
        from tools.rag import run_evaluation
        with st.spinner("评估中（约 30 秒/题）..."):
            result = run_evaluation()
        if "error" in result:
            st.error(result["error"])
        else:
            m = result["metrics"]
            st.metric("总体评分", f"{m['overall']}/10")
            for dim, data in m.items():
                if dim != "overall":
                    st.metric(
                        {"faithfulness": "忠实度", "context_precision": "精度",
                         "context_recall": "召回", "answer_relevancy": "相关性"}
                        .get(dim, dim),
                        f"{data['avg']}/10",
                    )

    st.divider()
    st.caption(f"当前时间: {datetime.now().strftime('%H:%M:%S')}")


# ── 主聊天区域 ─────────────────────────────────────
st.title("💬 通用智能体对话")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander("🔍 查看详情", expanded=False):
                st.text(msg["details"])

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 检查是否初始化
    if not st.session_state.agent_executor:
        with st.chat_message("assistant"):
            st.error("Agent 未初始化，请检查 .env 配置")
        st.stop()

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                now_str = datetime.now().strftime("%Y年%m月%d日 %H:%M")
                time_tagged = f"[当前时间: {now_str}] {prompt}"
                response = st.session_state.agent_executor.invoke(
                    {"input": time_tagged}
                )
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })
            except Exception as e:
                st.error(f"出错了: {e}")