import json
import concurrent.futures
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from repository.processor.import_processor.base import BaseNode
from repository.processor.import_processor.state import ImportGraphState
from repository.utils.client.ai_clients import AIClients
from repository.utils.client.storage_clients import StorageClients
from repository.prompts.import_prompt import KG_EXTRACTION_SYSTEM_PROMPT, KG_EXTRACTION_USER_PROMPT_TEMPLATE

class KgExtractionNode(BaseNode):
    name = "kg_extraction_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 拿到核心实体与【父切片】集合
        item_name = state.get('item_name')
        parent_chunks = state.get('parent_chunks', [])

        if not item_name or not parent_chunks:
            self.logger.warning("缺失 item_name 或 parent_chunks，跳过图谱抽取。")
            state['kg_triplets'] = {}
            return state

        # 2. 核心修改：切片动态批处理 (Chunk Batching)
        # 将 500-800 字符的小父切片，合并为适合大模型单次阅读的大块（约 6000 字符）
        BATCH_MAX_CHARS = 8000
        batched_texts = []
        current_batch = []
        current_len = 0

        for chunk in parent_chunks:
            content = chunk.get('content', '').strip()
            if len(content) < 20:
                continue

            # 如果当前批次加上新切片超出了限制，则封箱当前批次
            if current_len + len(content) > BATCH_MAX_CHARS and current_batch:
                batched_texts.append("\n\n".join(current_batch))
                current_batch = [content]
                current_len = len(content)
            else:
                current_batch.append(content)
                current_len += len(content)

        # 兜底最后一个批次
        if current_batch:
            batched_texts.append("\n\n".join(current_batch))

        self.logger.info(
            f"开始为 [{item_name}] 抽取图谱。原父切片数: {len(parent_chunks)}，合并后批次数: {len(batched_texts)}")

        # 3. 线程池并发处理【合并后的大批次】
        merged_properties = {}
        merged_relations = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._extract_from_chunk, item_name, batch_text)
                for batch_text in batched_texts
            ]

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    merged_properties.update(res.get("properties", {}))
                    merged_relations.extend(res.get("relations", []))

        # 4. 写入 Neo4j 属性图
        if merged_properties or merged_relations:
            self._save_to_neo4j(item_name, state.get('item_features', {}), merged_properties, merged_relations)
        else:
            self.logger.info(f"未从 {item_name} 的文档中提取到结构化参数。")

        # 5. 更新状态
        state['kg_triplets'] = {"properties": merged_properties, "relations": merged_relations}
        return state

    def _extract_from_chunk(self, item_name: str, context: str) -> Dict[str, Any]:
        """抽取图谱属性"""
        try:
            llm_client = AIClients.get_llm_client(response_format=True)

            user_prompt = KG_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                item_name=item_name,
                context=context
            )

            response = llm_client.invoke([
                SystemMessage(content=KG_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            res_str = response.content.strip('` \n')
            if res_str.startswith("json"):
                res_str = res_str[4:]

            extracted_data = json.loads(res_str)
            return extracted_data
        except Exception as e:
            self.logger.warning(f"单个父切片图谱抽取失败，已跳过: {e}")
            return {}

    def _save_to_neo4j(self, item_name: str, item_features: dict, properties: dict, relations: list):
        """将合并后的完整参数与关系写入 Neo4j"""
        # 获取基础结构化特征，兜底处理防止 None
        brand = item_features.get('brand') or '未知品牌'
        category = item_features.get('category') or '未知分类'
        model = item_features.get('model') or '未知型号'

        # 清洗 Properties：确保全是字符串，因为 Neo4j 节点属性不支持嵌套 Dict 或 None
        clean_properties = {}
        for k, v in properties.items():
            if isinstance(v, (str, int, float, bool)):
                clean_properties[k] = str(v)
            elif isinstance(v, list):
                clean_properties[k] = ", ".join([str(i) for i in v])

        # 【核心升级】：构建标准工业图谱拓扑
        cypher_query = """
        // 1. 匹配或创建 品牌(Brand) 和 品类(Category) 的独立节点
        MERGE (b:Brand {name: $brand})
        MERGE (c:Category {name: $category})

        // 2. 匹配或创建 核心产品(Product) 节点
        MERGE (p:Product {name: $item_name})
        ON CREATE SET p.model = $model

        // 3. 建立 核心拓扑关系
        MERGE (p)-[:BELONGS_TO_BRAND]->(b)
        MERGE (p)-[:IS_A_CATEGORY]->(c)

        // 4. 动态追加所有抽取到的产品参数到 Product 节点的 Property 中
        SET p += $properties

        // 5. 展开并写入 LLM 提取的外部实体关系 (Accessories, Series, etc.)
        WITH p
        UNWIND $relations AS rel
        WITH p, rel WHERE rel.target_name IS NOT NULL AND rel.target_name <> ""
        // 为了兼容未来各种意想不到的实体，统一用 :Entity 标签
        MERGE (t:Entity {name: rel.target_name})
        MERGE (p)-[r:RELATED_TO]->(t)
        // 把大模型提取的具体关系描述（如"可搭配"、"前置型号"）存入关系类型的属性中
        SET r.type = rel.relation_type
        """

        try:
            driver = StorageClients.get_neo4j_client()
            with driver.session() as session:
                session.run(cypher_query,
                            item_name=item_name,
                            brand=brand,
                            category=category,
                            model=model,
                            properties=clean_properties,
                            relations=relations)
            self.logger.info(
                f"成功将产品 [{item_name}] 写入 Neo4j 拓扑网络，"
                f"绑定品牌[{brand}]与品类[{category}]，包含 {len(clean_properties)} 个属性和 {len(relations)} 个动态关系。"
            )
        except Exception as e:
            self.logger.error(f"存入 Neo4j 失败: {e}")