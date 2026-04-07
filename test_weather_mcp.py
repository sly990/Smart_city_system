# test_mcp.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test():
    params = StdioServerParameters(command="python", args=["weather_mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_realtime_weather", arguments={"location": "广州"})
            print(f"MCP 测试结果: {result}")

asyncio.run(test())


#python test_weather_mcp.py