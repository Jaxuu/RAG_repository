import sqlite3
import json
import logging
from mcp.server.fastmcp import FastMCP

# 初始化 ERP 数据库 MCP Server
mcp = FastMCP("ERP-SQLite-Server", port=8001)

# 指向我们刚刚用脚本生成的数据库文件
DB_PATH = r"D:\Project\llm-project\RAG_repository\repository\test\db_storage\test.db"


@mcp.tool()
def query_erp_database(sql_query: str) -> str:
    """
    检索企业内部 ERP 数据库。
    重要提示：
    - 表名为: inventory
    - 包含字段: part_number (型号), category (分类), stock_quantity (库存数量), unit_price (单价), warehouse_location (仓库位置), last_updated (更新时间)

    Args:
        sql_query: 合法的 SQLite SQL 查询语句。例如 "SELECT stock_quantity, unit_price FROM inventory WHERE part_number = 'HDR-60-24'"
    """
    logging.info(f"收到 SQL 查询: {sql_query}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        # 获取列名
        columns = [description[0] for description in cursor.description]
        conn.close()

        # 组装成 JSON 返回给大模型
        return json.dumps({"columns": columns, "data": rows}, ensure_ascii=False)
    except Exception as e:
        error_msg = f"SQL执行失败: {str(e)}"
        logging.error(error_msg)
        return error_msg


if __name__ == "__main__":
    # 工业标准 SSE 架构，监听 8001 端口 (避开 RAG 的 8080 端口)
    mcp.run(transport='sse')