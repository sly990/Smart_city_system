from mcp.server.fastmcp import FastMCP
import os
import requests
import logging

logging.basicConfig(
    filename='weather_mcp.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    force=True
)

mcp = FastMCP("SmartCityWeatherServer", stateless_http=True, json_response=True)

SENIVERSE_API_KEY = os.getenv("SENIVERSE_API_KEY", "SoDoN4dB2HPb_CUXz")

@mcp.tool()
def get_realtime_weather(location: str) -> str:
    logging.info(f"收到天气查询请求: {location}")
    try:
        url = "https://api.seniverse.com/v3/weather/now.json"
        params = {
            "key": SENIVERSE_API_KEY,
            "location": location,
            "language": "zh-Hans",
            "unit": "c"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "results" in data and data["results"]:
            now = data["results"][0]["now"]
            loc = data["results"][0]["location"]
            text = f"【{loc['name']}实时天气】天气现象：{now['text']}，当前温度：{now['temperature']}℃。"
            logging.info(text)
            return text

        return f"未找到 {location} 的天气数据，请检查城市名称。"
    except Exception as e:
        logging.exception("获取天气信息失败")
        return f"获取天气信息失败: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 Weather MCP Server on http://127.0.0.1:8001/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1", port=8001)