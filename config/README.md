# 配置文件说明

运行时均由 `utils/config_utils.py` 加载并暴露为模块级变量（如 `agent_conf`、`chroma_conf`），**业务代码仍通过原有 `*_conf` 访问，键名不变**。

| 文件 | 职责 |
|------|------|
| `api.yml` | DashScope 等 API 密钥（可用环境变量 `DASHSCOPE_API_KEY` 覆盖） |
| `agent.yml` | Agent 运行时：对话条数、recursion、reflection、外部 CSV 路径等 |
| `memory.yml` | 长期记忆注入/抽取、事实存储、队列、多值画像等；与 `agent.yml` **合并**为 `agent_conf` |
| `rag.yml` | 聊天/嵌入模型名、温度、轻量 turbo 模型名 |
| `chroma.yml` | RAG 向量库、混合检索、rerank、改写、离线索引等 |
| `chroma_memory.yml` | 用户长期记忆向量库（合并入 `chroma_conf`，与 `chroma.yml` 键不冲突） |
| `prompts.yml` | 各类提示词文件路径 |
| `skills.yml` | `raw_agent_skillkit` 技能开关与根目录 |
| `mcp.json` | MCP 服务器列表 |
| `eval_rag.yml` | 离线 RAG 评测（可选） |

日志目录：项目根目录 `logs/`（见 `utils/log_utils.py`）。

自检：项目根执行 `python scripts/verify_config.py`，确认合并后的配置键存在。

路径：`utils.path_utils.get_repo_root()` 为仓库根；`get_abs_path("../...")` 仍以 `utils/` 为基准，与历史行为一致。
