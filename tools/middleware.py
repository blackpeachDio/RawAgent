import time
from typing import Callable

from utils.latency_trace import note_before_model, note_llm_api_wall, note_tool_done
from utils.prompt_utils import (
    append_original_query_anchor,
    format_memory_system_prompt,
    load_report_prompts,
    load_system_prompts,
)
from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse, after_model,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.token_utils import count_agent_llm_input_tokens_from_model_request


def _tool_call_id(request: ToolCallRequest) -> str | None:
    tc = request.tool_call
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


def _tool_error_content_for_model(request: ToolCallRequest, exc: Exception) -> str:
    """结构化错误文本，便于模型区分参数/超时/网络等并重试。"""
    tc = request.tool_call
    name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
    err_type = type(exc).__name__
    msg = (str(exc) or "").strip() or "（无详细信息）"
    low = msg.lower()
    if "timeout" in low or isinstance(exc, TimeoutError):
        hint = "倾向：超时类问题，可稍后重试、缩小请求范围或换用其它工具。"
    elif err_type in ("ValidationError", "ValueError", "TypeError", "KeyError", "JSONDecodeError"):
        hint = "倾向：参数或返回解析问题，请对照工具定义修正入参后再调。"
    elif any(s in low for s in ("connection", "resolve", "network", "ssl", "errno")):
        hint = "倾向：网络连接类问题，可稍后重试。"
    elif "429" in msg or "rate" in low:
        hint = "倾向：限流或配额，请稍后重试。"
    else:
        hint = "请根据错误类型判断是否修正参数或稍后重试；勿向用户编造工具已成功。"
    return (
        f"[工具执行失败]\n"
        f"工具名: {name}\n"
        f"错误类型: {err_type}\n"
        f"错误详情: {msg}\n"
        f"处理建议: {hint}"
    )


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    t_tool = time.perf_counter()
    try:
        result = handler(request)
        note_tool_done(str(request.tool_call["name"]), time.perf_counter() - t_tool)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        name = request.tool_call.get("name", "") if isinstance(request.tool_call, dict) else getattr(request.tool_call, "name", "")
        note_tool_done(str(name or "unknown"), time.perf_counter() - t_tool)
        logger.error(f"工具{name}调用失败，原因：{str(e)}", exc_info=True)
        tid = _tool_call_id(request)
        if tid is None:
            raise
        body = _tool_error_content_for_model(request, e)
        return ToolMessage(content=body, tool_call_id=tid)


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    note_before_model()
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug("[before_model] messages_count=%d", len(state["messages"]))
    return None

@after_model
def after_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 模型调用之后（调试用，默认不刷屏）
    logger.debug("[after_model] messages_count=%d", len(state["messages"]))
    return None

@wrap_model_call
def log_wrap_model_tokens(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """在真正调模型前可选估算 token；大 tools schema 时估算很慢，见 agent.yml。"""
    if bool(agent_conf.get("agent_llm_token_estimate_enabled", True)):
        try:
            n_in = count_agent_llm_input_tokens_from_model_request(request)
            logger.info(
                "[agent_llm] 输入 token 估算（含 system、messages、tools 等，cl100k_base 近似，供成本参考）: %d",
                n_in,
            )
        except Exception as e:
            logger.warning("[agent_llm] token 估算失败: %s", e)
    t0 = time.perf_counter()
    resp = handler(request)
    note_llm_api_wall(time.perf_counter() - t0)
    return resp


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
        base = format_memory_system_prompt(memory, base)

    # 本轮原始用户问题：每次调 LLM 前写入 system，降低多步工具循环中的目标漂移
    oq = request.runtime.context.get("original_query")
    if bool(agent_conf.get("agent_anchor_original_query", True)):
        base = append_original_query_anchor(base, oq if isinstance(oq, str) else "")
    return base
