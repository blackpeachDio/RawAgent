---
name: rawagent-project
description: >-
  Develops and maintains the RawAgent Python repo (LangGraph ReAct, RAG/Chroma,
  DashScope, memory queues, MCP, skillkit). Use when editing this workspace,
  running scripts, or when the user mentions RawAgent, rag/, agent/, memory/,
  or project-specific config paths.
---

# RawAgent 项目协作

## 范围

- **Agent**：`agent/react_agent.py`、`agent/react_graph_build.py`；图由 `compile_react_agent` 统一构建；内建工具在根包 **`tools/`**（勿使用已删除的 `agent.tools`）。
- **RAG**：`rag/`（`retrieval_pipeline`、`online_query`）；嵌入模型名见 `config/rag.yml`。
- **配置**：`utils/config_utils.py` 合并 `agent.yml`+`memory.yml`、`chroma.yml`+`chroma_memory.yml`；密钥优先环境变量 `DASHSCOPE_API_KEY`。
- **路径**：用 `utils.path_utils` 的 `get_repo_root`、`resolve_repo_path`、`get_config_path`；不要用已废弃的「相对 utils 的随意拼接」。
- **自检**：`python scripts/verify_paths.py`（轻量）；完整配置键用 `python scripts/verify_config.py`（需依赖与 `api.yml`/密钥）。

## 习惯

- 改行为前确认用户是否要**保持语义不变**（尤其 RAG/记忆/DB）。
- 终端在 Windows 下用 PowerShell 语法；优先在项目 venv 中运行（需 `yaml`、`langchain` 等与 `requirements.txt` 一致）。
- 新增依赖需用户明确同意；勿擅自加 PyPI 依赖。

## 输出

- 说明改动时用**完整句**；引用代码用 Cursor 要求的代码引用格式（带路径与行号）。
