from typing import Callable
from utils.prompt_utils import format_memory_system_prompt, load_report_prompts, load_system_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.log_utils import logger
from utils.token_utils import count_agent_llm_input_tokens


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    try:
        n_in = count_agent_llm_input_tokens(state, runtime)
        logger.info("[agent_llm] 输入 token 估算（cl100k_base 近似，供成本参考）: %d", n_in)
    except Exception as e:
        logger.warning("[agent_llm] token 估算失败: %s", e)

    # full, max_chars = get_prompt_log_config()
    #
    # try:
    #     msg_list = state.get("messages") or []
    #     truncate_fn = lambda s: maybe_truncate(s, full=full, max_chars=max_chars)
    #     prompt_text = format_messages_as_prompt_text(msg_list, truncate_fn=truncate_fn)
    #     log_truncated_block(logger, "[PROMPT_BEGIN]", "[PROMPT_END]", prompt_text)
    # except Exception as e:
    #     logger.warning("[log_before_model]打印 prompt 失败：%s", str(e))

    return None


@dynamic_prompt
def build_system_prompt(request: ModelRequest):
    """根据 context 动态构建 system prompt：报告/主流程切换 + 用户记忆注入。"""
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        base = load_report_prompts()
    else:
        base = load_system_prompts()

    # 注入长期记忆（摘要、画像等），供模型参考（模板见 prompts/mem_inject_prompt.txt）
    memory = request.runtime.context.get("memory", "").strip()
    if memory:
        return format_memory_system_prompt(memory, base)
    return base
