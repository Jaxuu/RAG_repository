import uuid
import re
from typing import Tuple, List, Dict, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage

from repository.processor.query_processor.base import BaseNode
from repository.processor.query_processor.state import QueryGraphState
from repository.utils.client.ai_clients import AIClients
from repository.utils.client.storage_clients import StorageClients
from repository.prompts.query_prompt import TEXT2CYPHER_SYSTEM_PROMPT


class KgSearchNode(BaseNode):
    name = "kg_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        user_query = state.get('rewritten_query') or state.get('original_query')

        if not user_query:
            return {"kg_search_chunks": []}

        # 1. 动态生成 Cypher (Text2Cypher)
        cypher_query = self._generate_cypher(user_query)
        if not cypher_query:
            self.logger.warning("大模型未能生成有效的 Cypher 语句，跳过图谱检索。")
            return {"kg_search_chunks": []}

        self.logger.info(f"生成动态 Cypher 查询: \n{cypher_query}")

        # 2. 执行 Cypher 查询获取原始数据
        kg_results = self._execute_cypher(cypher_query)
        if not kg_results:
            self.logger.info("图谱检索未命中任何关联数据。")
            return {"kg_search_chunks": []}

        # 3. 将原始结果转化为自然语言 Chunk (解决 JSON 大模型不易读的问题)
        chunk_content = self._textualize_results(user_query, kg_results)

        # 4. 伪装成标准的 RAG 召回切片
        chunk = {
            "distance": 1.0,  # 图谱数据的置信度极高，赋满分
            "entity": {
                "chunk_id": f"kg_{uuid.uuid4().hex[:8]}",
                "content": chunk_content,
                "title": "知识图谱精准解答",
                "source": "knowledge_graph",
                "item_name": "图谱多跳聚合结果",
                "parent_id": None
            }
        }

        self.logger.info(f"知识图谱转化 Chunk 完成，长度 {len(chunk_content)} 字符。")
        return {"kg_search_chunks": [chunk]}

    def _generate_cypher(self, user_query: str) -> str:
        """调用 LLM 根据 Schema 生成 Cypher 语句"""
        try:
            llm_client = AIClients.get_llm_client(response_format=False)

            user_prompt = f"用户问题：{user_query}\n请输出 Cypher 语句："

            response = llm_client.invoke([
                SystemMessage(content=TEXT2CYPHER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            # 清理可能的 markdown 标记
            cypher = response.content.strip()
            cypher = re.sub(r"^```(?:cypher|sql)?\s*", "", cypher)
            cypher = re.sub(r"\s*```$", "", cypher)

            # 简单的安全校验（只允许读操作）
            if "DELETE " in cypher.upper() or "SET " in cypher.upper() or "REMOVE " in cypher.upper() or "MERGE " in cypher.upper():
                self.logger.error("检测到危险操作，已拦截！")
                return ""

            return cypher
        except Exception as e:
            self.logger.error(f"Text2Cypher 生成失败: {e}")
            return ""

    def _execute_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """在 Neo4j 中执行查询并返回字典列表"""
        try:
            neo4j_client = StorageClients.get_neo4j_client()
            with neo4j_client.session() as session:
                result = session.run(cypher)
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Neo4j 语句执行失败: {e}")
            return []

    def _textualize_results(self, user_query: str, kg_results: List[Dict[str, Any]]) -> str:
        """
        利用大模型将干瘪的 JSON 结构，转化为极具高语义密度的自然语言，
        从而让下游的 Reranker (BGE-M3) 能够完美打分！
        """
        try:
            llm_client = AIClients.get_llm_client(response_format=False)

            sys_prompt = "你是一个数据翻译官。请把下面的 JSON 数据（来源于图数据库），转化为一段自然流畅、信息密集的中文说明文本，用以回答用户的疑问。如果数据中包含多个产品对比，请条理清晰地罗列。"
            user_prompt = f"【用户问题】：{user_query}\n【图谱查询结果 (JSON)】：\n{kg_results}\n\n请直接输出翻译后的自然语言说明（不加废话）："

            response = llm_client.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])
            return "【知识图谱精准检索结果】\n" + response.content.strip()
        except Exception as e:
            self.logger.error(f"图谱结果文本化翻译失败，降级返回 JSON 字符串: {e}")
            return f"【知识图谱原始数据】\n{kg_results}"