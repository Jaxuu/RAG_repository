import asyncio
import os

import httpx2
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


load_dotenv(override=True)


async def main():
    url = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"

    api_key = os.getenv("MCP_DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError("MCP_DASHSCOPE_API_KEY 未读取到")

    print("API Key 是否读取到：", bool(api_key))
    print("MCP URL：", url)

    async with httpx2.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/event-stream",
        },
        timeout=httpx2.Timeout(
            60.0,
            read=300.0,
        ),
        follow_redirects=True,
    ) as http_client:

        async with streamable_http_client(
            url,
            http_client=http_client,
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                print("开始 initialize...")

                result = await session.initialize()

                print("\n初始化成功：")
                print(result)

                print("\n开始获取工具列表...")

                tools = await session.list_tools()

                print("\n工具列表：")

                for tool in tools.tools:
                    print(tool.name)


if __name__ == "__main__":
    asyncio.run(main())