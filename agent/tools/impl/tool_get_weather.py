import json
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from utils.log_utils import logger

_WEATHER_DESC_ZH = {
    "Clear": "晴",
    "Partly cloudy": "多云",
    "Cloudy": "阴",
    "Overcast": "阴",
    "Mist": "雾",
    "Fog": "雾",
    "Light rain": "小雨",
    "Moderate rain": "中雨",
    "Heavy rain": "大雨",
    "Light snow": "小雪",
    "Moderate snow": "中雪",
}


def _fetch_weather_from_wttr(city: str) -> str | None:
    """从 wttr.in 获取实时天气（免费，无需 API key）。"""
    try:
        city_encoded = quote(city.strip())
        url = f"https://wttr.in/{city_encoded}?format=j1"
        req = Request(url, headers={"User-Agent": "curl/7.64.1"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cc = data.get("current_condition", [{}])[0]
        temp = cc.get("temp_C", "-")
        feels = cc.get("FeelsLikeC", temp)
        humidity = cc.get("humidity", "-")
        desc_en = (cc.get("weatherDesc", [{}])[0].get("value", "") if cc.get("weatherDesc") else "")
        desc = _WEATHER_DESC_ZH.get(desc_en, desc_en or "未知")
        wind = cc.get("windspeedKmph", "-")
        wind_dir = cc.get("winddir16Point", "")
        precip = cc.get("precipMM", "0")
        return (
            f"城市{city}：{desc}，气温{temp}℃（体感{feels}℃），"
            f"空气湿度{humidity}%，{wind_dir}风{wind}km/h，"
            f"降水{precip}mm"
        )
    except (URLError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("[get_weather] wttr.in 获取失败: %s", e)
        return None


def get_weather_impl(city: str) -> str:
    result = _fetch_weather_from_wttr(city)
    if result:
        return result
    return f"暂无法获取{city}的实时天气，请稍后重试或检查城市名称是否正确。"
