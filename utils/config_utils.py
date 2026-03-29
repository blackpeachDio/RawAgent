"""
yaml
k: v
"""
import os

import yaml

from utils.path_utils import get_abs_path


def load_rag_config(config_path: str = get_abs_path("../config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path("../config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path("../config/prompts.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("../config/agent.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_mcp_config(config_path: str = get_abs_path("../config/mcp.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def load_api_config(config_path: str = get_abs_path("../config/api.yml"), encoding: str = "utf-8"):
    """支持 config/api.yml；Docker/CI 可仅用环境变量 DASHSCOPE_API_KEY（会覆盖文件中的值）。"""
    data: dict = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding=encoding) as f:
            data = yaml.load(f, Loader=yaml.FullLoader) or {}
    env_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if env_key:
        data["dashscope_api_key"] = env_key
    if not data.get("dashscope_api_key"):
        raise ValueError(
            "缺少 dashscope_api_key：请在 config/api.yml 中配置，或设置环境变量 DASHSCOPE_API_KEY。"
        )
    return data


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
mcp_conf = load_mcp_config()
api_conf = load_api_config()

# if __name__ == '__main__':
#     print(rag_conf["chat_model_name"])
