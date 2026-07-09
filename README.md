# 🤖 通用智能体 (General-Purpose Agent)

基于 LangChain 构建的通用 AI 智能体，支持：

- **联网搜索** — 通过 DuckDuckGo 获取实时信息
- **文件读写** — 创建、读取、编辑本地文件
- **对话记忆** — 记住上下文，连续对话
- **多模型** — 支持 OpenAI GPT 和 Anthropic Claude

## 快速开始

### 1. 安装依赖

```bash
cd D:\my-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API 密钥

编辑 `.env` 文件，填入密钥（二选一即可）：

| 平台 | 环境变量 | 获取地址 |
|------|----------|----------|
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com |

### 3. 启动

```bash
python main.py
```

## 使用命令

| 命令 | 作用 |
|------|------|
| `/help` | 显示帮助 |
| `/clear` | 清除对话历史 |
| `/memory` | 查看记忆内容 |
| `/tools` | 列出可用工具 |
| `/exit` | 退出程序 |

直接输入问题即可与智能体对话。

## 项目结构

```
D:\my-agent\
├── main.py           # 主程序入口
├── tools/            # 自定义工具
│   ├── __init__.py
│   ├── web_search.py # 联网搜索
│   └── file_tools.py # 文件读写
├── memory/           # 记忆存储目录
├── .env              # API 密钥配置
├── .gitignore
├── requirements.txt
└── README.md
```