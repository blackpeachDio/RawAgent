"""
内建 LangChain 工具定义；对外 ``from tools import rag_summarize, ...``。
"""
import datetime
import os

from langchain_core.tools import tool
from langgraph.types import interrupt

from rag.online_query import RagSummarizeService
from utils.config_utils import agent_conf
from utils.log_utils import logger
from utils.path_utils import resolve_repo_path
from .impl.tool_get_weather import get_weather_impl

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010", ]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return get_weather_impl(city)


USER_MAP = {
    "张三": {"id": "U1001", "location": "深圳"},
    "李四": {"id": "U1002", "location": "合肥"},
    "王五": {"id": "U1003", "location": "杭州"},
    "赵六": {"id": "U1004", "location": "北京"},
    "钱七": {"id": "U1005", "location": "上海"},
}


# ======================
# 工具 1：根据姓名获取城市
# ======================
@tool(description="获取用户所在城市的名称，根据用户姓名查询，以纯字符串形式返回")
def get_user_location(user_name: str) -> str:
    """根据用户姓名获取所在城市"""
    if user_name not in USER_MAP:
        return "未知城市"
    return USER_MAP[user_name]["location"]


# ======================
# 工具 2：根据姓名获取用户ID
# ======================
@tool(description="获取用户的ID，根据用户姓名查询，以纯字符串形式返回")
def get_user_id(user_name: str) -> str:
    """根据用户姓名获取用户ID"""
    if user_name not in USER_MAP:
        return "未知用户ID"
    return USER_MAP[user_name]["id"]


# ======================
# 工具 3：获取真实当前月份
# ======================
@tool(description="获取当前月份，以纯字符串形式返回，如：'2025年07月'")
def get_current_month() -> str:
    """获取真实的当前月份"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月")  # 例子：2025年07月


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = resolve_repo_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


@tool(
    description=(
            "当用户问题中关键信息缺失、含糊或无法安全调用其它工具时调用。"
            "questions 须为完整中文、直接展示给用户的追问说明；"
            "missing_slots 为逗号分隔的缺失要点摘要（可选）。"
            "调用后对话将暂停直到用户在前端补充；禁止编造用户未提供的信息。"
    )
)
def request_user_clarification(
        reason: str,
        questions: str,
        missing_slots: str = "",
) -> str:
    """人机回环：挂起当前 run，用户补充后 `interrupt` 返回用户文本。"""
    slots = [s.strip() for s in (missing_slots or "").split(",") if s.strip()]
    payload = {
        "kind": "hitl",
        "reason": (reason or "").strip(),
        "questions": (questions or "").strip(),
        "missing_slots": slots,
    }
    user_reply = interrupt(payload)
    if isinstance(user_reply, dict):
        user_reply = user_reply.get("text") or user_reply.get("content") or ""
    user_reply = (str(user_reply) if user_reply is not None else "").strip()
    return (
            "[用户已补充以下信息，请据此继续推理或作答；勿重复已回答过的追问。]\n"
            + user_reply
    )
