"""
MarkdownTableLinearizer 表格线性化解析工具
专为 RAG (检索增强生成) 架构设计，解决大模型和向量模型对复杂二维表格的“注意力失焦”与“格式降智”问题。

【核心处理流程与原因】：

1. 双轨制拦截与标签清洗 (Recognition & Cleaning)
    - HTML 表格优先 (BeautifulSoup)：
        工业文档转换时常残留复杂的 <table border="1"> 等 HTML 标签。正则匹配极易被换行和嵌套标签打断。
        本工具直接利用 BS4 提取 DOM 树，处理后使用 `replace_with` 将结果原位替换为纯文本，
        最后通过 `get_text()` 彻底抹除 <tr> <td> 等会干扰大模型注意力的非语义噪音。
    - Markdown 表格兜底 (Regex)：
        处理完 HTML 后，利用正则精准捕获原生 Markdown 表格 (如 |---|---| 格式)。

2. 结构矩阵化与降维 (Matrixification)
    无论来源是 HTML 还是 Markdown，统一将其映射为标准二维数组（List[List[str]]，即 Grid）。
    - 应对复杂合并：针对 HTML 的 rowspan 和 colspan（合并单元格），通过在 Grid 中向下、向右自动
      填充占位符并复制文本，将复杂的“立体交叉表”强制拍平为绝对对齐的标准二维矩阵。

3. 语义解耦与自然语言展平 (Semantic Decoupling Linearization) -> `_grid_to_text` 方法
    这是大幅提升 Ragas 评测中 Context Precision (上下文精确度) 的核心。彻底摒弃了旧版使用方括号
    "[表头: 值] [表头: 值]" 堆砌同一行的做法，改为以下绝对策略：

    - 策略 1（双列 KV 表）：
        直接降维为键值对，输出格式：`- Key: Value`。

    - 策略 2（多列表格 - 自然语言锚点法）：
        【痛点】：早期将不同型号的参数挤在同一行，会导致模型发生“上下文串扰 (Context Crosstalk)”，
                 错把 A 的参数安在 B 身上，导致严重幻觉和低召回。
        【解法】：将表格第一列强制视为“主键/型号锚点”。以锚点为核心，向上追溯表头，结合单元格数值，
                 重组为高内聚的自然语言短句。
        【输出形态】：`- 关于【EDR-75-24】：DC VOLTAGE为24V，RATED POWER为76.8W。`
        【收益】：强行绑定了“实体对象(锚点)”与“属性值”，极大地迎合了向量模型(BGE-M3)和基座大模型的
                 自然语言预训练偏好，彻底消除注意力偏移。
"""

import re
from typing import List
from bs4 import BeautifulSoup

class MarkdownTableLinearizer:
    """
    解决：HTML复杂合并单元格、无表头KV表、左上角空置交叉表、原生MD表
    """

    MD_TABLE_PATTERN = re.compile(
        r'((?:^[ \t]*\|.*\|[ \t]*\n)'
        r'(?:^[ \t]*\|[ \t]*[-:]+[-| :]*\|[ \t]*\n)'
        r'(?:^[ \t]*\|.*\|[ \t]*(?:\n|$))*)',
        re.MULTILINE
    )

    @classmethod
    def process(cls, content: str) -> str:
        if not content:
            return content

        # 1. 彻底解决残留 <table> 标签的问题：直接丢给 bs4
        if "<table" in content.lower():
            soup = BeautifulSoup(content, "html.parser")
            for table in soup.find_all("table"):
                linearized_text = cls._process_single_html_table(table)
                # 替换掉原有的 table 节点为纯文本
                table.replace_with(linearized_text)
            content = soup.get_text(separator="\n", strip=True)

        # 2. 处理纯 Markdown 格式的表格
        if "|" in content:
            content = cls.MD_TABLE_PATTERN.sub(cls._replace_md_table, content)

        return content

    @classmethod
    def _process_single_html_table(cls, table) -> str:
        rows = table.find_all("tr")
        if not rows: return ""

        # 1. 纯 KV 表（只有两列）的快速降维
        is_kv_table = all(len(row.find_all(['td', 'th'])) == 2 for row in rows[:3])
        if is_kv_table:
            res = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) == 2:
                    k = cells[0].get_text(separator=" ", strip=True)
                    v = cells[1].get_text(separator=" ", strip=True)
                    if k and v:
                        res.append(f"- 【{k}】：{v}")
            return "\n".join(res)

        # 2. 复杂交叉表：还原 rowspan/colspan 生成标准二维网格
        grid = []
        for _ in range(len(rows)):
            grid.append([])

        has_th = len(table.find_all("th")) > 0

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                # 跳过已被 rowspan/colspan 占用的格子
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                text = cell.get_text(separator=" ", strip=True)

                for r in range(row_idx, row_idx + rowspan):
                    while len(grid) <= r: grid.append([])
                    while len(grid[r]) < col_idx + colspan: grid[r].append(None)
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = text
                col_idx += colspan

        return cls._grid_to_text(grid, is_md=False, has_th=has_th)

    @classmethod
    def _replace_md_table(cls, match) -> str:
        md_text = match.group(0).strip()
        lines = md_text.split('\n')
        grid = []
        for line in lines:
            if re.match(r'^[ \t]*\|[ \t\-|:]+\|[ \t]*$', line): continue
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            grid.append(cells)
        return cls._grid_to_text(grid, is_md=True, has_th=False)

    @classmethod
    def _grid_to_text(cls, grid: List[List[str]], is_md: bool, has_th: bool) -> str:
        if not grid or not grid[0]: return ""

        cols_count = max(len(r) for r in grid)
        for r in grid:
            while len(r) < cols_count: r.append("")

        res = []

        # 【策略 1：两列表格，绝对的 KV 关系】保持不变
        if cols_count == 2:
            for r in grid:
                if r[0] and r[0] not in ['-', '---']:
                    res.append(f"- {r[0]}: {r[1]}")
            return "\n\n" + "\n".join(res) + "\n\n"

        # 【策略 2 优化：多列表格语义解耦化】
        # 提取表头（支持多行表头拼接，这里默认第0行为主表头）
        headers = [h.strip() for h in grid[0]]

        for r in grid[1:]:
            # 跳过分割线 (---|---) 或空行
            if not any(r) or (r[0] and set(r[0].strip()).issubset({'-', ':', ' '})):
                continue

            # 视第一列为主键/型号锚点（例如 "EDR-75-24"）
            row_anchor = r[0] if r[0] and r[0] not in ('-', '/') else "通用"

            # 遍历当前行的每一列，拒绝长行平铺，改为生成独立的子属性描述
            row_statements = []
            for c in range(1, cols_count):  # 从第1列开始，第0列作为锚点
                val = r[c]
                if not val or val in ('-', '/', '\\', '无', 'N/A'):
                    continue

                header_name = headers[c] if c < len(headers) and headers[c] else f"指标{c + 1}"
                # 过滤掉表头中的冗余特殊符号
                header_name = re.sub(r'[\n\r\[\]]', ' ', header_name).strip()

                # 【核心改进】：生成高内聚的短句，自带型号锚点，彻底消除上下文串扰
                row_statements.append(f"{header_name}为{val}")

            if row_statements:
                # 组合成形如："- EDR-75-24 的 DC VOLTAGE 为 24V，RATED CURRENT 为 3.2A，RATED POWER 为 76.8W"
                statement_str = "，".join(row_statements)
                res.append(f"- 关于【{row_anchor}】：{statement_str}。")

        return "\n\n" + "\n".join(res) + "\n\n"