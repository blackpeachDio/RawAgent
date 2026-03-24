from utils.config_utils import prompts_conf
from utils.path_utils import get_abs_path
from utils.log_utils import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
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
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
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
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
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
        judge_prompt_path = get_abs_path(prompts_conf["judge_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_judge_prompts]在yaml配置项中没有judge_prompt_path配置项")
        raise e

    try:
        return open(judge_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_judge_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_mem_extract_prompts():
    try:
        prompt_path = prompts_conf.get("mem_extract_prompt_path", "../prompts/mem_extract_prompt.txt")
        path = get_abs_path(prompt_path)
    except KeyError:
        path = get_abs_path("../prompts/mem_extract_prompt.txt")
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error("[load_mem_extract_prompts] 解析记忆提取提示词出错: %s", e)
        raise


if __name__ == '__main__':
    print(load_report_prompts())

