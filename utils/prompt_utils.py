from utils.config_utils import prompts_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path


def load_system_prompts():
    try:
        system_prompt_path = resolve_repo_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = resolve_repo_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = resolve_repo_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_judge_prompts():
    try:
        judge_prompt_path = resolve_repo_path(prompts_conf["judge_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_judge_prompts]在yaml配置项中没有judge_prompt_path配置项")
        raise e

    try:
        return open(judge_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_judge_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_reflect_critique_prompts() -> str:
    """在线自检；占位符 <<<USER_QUESTION>>>、<<<DRAFT_ANSWER>>>。"""
    try:
        path_key = prompts_conf.get(
            "reflect_critique_prompt_path",
            "../prompts/reflect_critique_prompt.txt",
        )
        path = resolve_repo_path(path_key)
    except Exception as e:
        logger.error("[load_reflect_critique_prompts] 路径解析失败: %s", e)
        raise
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_reflect_critique_prompts] 读取失败: %s", e)
        raise


def load_mem_extract_prompts():
    try:
        prompt_path = prompts_conf.get("mem_extract_prompt_path", "../prompts/mem_extract_prompt.txt")
        path = resolve_repo_path(prompt_path)
    except KeyError:
        path = resolve_repo_path("../prompts/mem_extract_prompt.txt")
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_mem_extract_prompts] 解析记忆提取提示词出错: %s", e)
        raise


def load_memory_extract_llm_gate_prompt() -> str:
    try:
        prompt_path = prompts_conf.get(
            "memory_extract_llm_gate_prompt_path",
            "../prompts/memory_extract_llm_gate.txt",
        )
        path = resolve_repo_path(prompt_path)
    except KeyError:
        path = resolve_repo_path("../prompts/memory_extract_llm_gate.txt")
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_memory_extract_llm_gate_prompt] 读取失败: %s", e)
        raise


def load_memory_inject_classifier_prompt() -> str:
    """memory_inject_mode=auto 且 strategy=llm 时，判定是否注入事实/向量记忆的短提示词；占位符：<<<USER_QUERY>>>。"""
    try:
        prompt_path = prompts_conf.get(
            "memory_inject_classifier_prompt_path",
            "../prompts/memory_inject_classifier.txt",
        )
        path = resolve_repo_path(prompt_path)
    except KeyError:
        path = resolve_repo_path("../prompts/memory_inject_classifier.txt")
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_memory_inject_classifier_prompt] 解析出错: %s", e)
        raise


def load_mem_inject_prompts() -> str:
    """拼接「用户记忆」与主 system 的模板，占位符：{memory}、{base}。"""
    try:
        prompt_path = prompts_conf.get("mem_inject_prompt_path", "../prompts/mem_inject_prompt.txt")
        path = resolve_repo_path(prompt_path)
    except KeyError:
        path = resolve_repo_path("../prompts/mem_inject_prompt.txt")
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_mem_inject_prompts] 解析出错: %s", e)
        raise


_mem_inject_template: str | None = None


def format_memory_system_prompt(memory: str, base: str) -> str:
    """将记忆与主提示词按模板拼成完整 system 文本。"""
    global _mem_inject_template
    if _mem_inject_template is None:
        _mem_inject_template = load_mem_inject_prompts()
    return _mem_inject_template.format(memory=memory, base=base)


if __name__ == '__main__':
    print(load_report_prompts())
