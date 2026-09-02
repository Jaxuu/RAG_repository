from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
from repository.processor.import_processor.base import BaseNode, setup_logging, T
from repository.processor.import_processor.state import ImportGraphState
from repository.processor.import_processor.exceptions import StateFieldError, ValidationError
from repository.utils.client.ai_clients import AIClients
from repository.utils.client.storage_clients import StorageClients
from repository.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        主要职责：
        1. 负责利用LLM提取商品的具体型号（名）
        2. 嵌入商品具体型号（名）
        3. 存储到Milvus中（MySQL：模糊查询的时候不会考虑语义）
        """

        # 1. 参数校验 (获取父块和子块)
        file_title, parent_chunks, child_chunks, item_name_chunk_k = self._validate_state(state)

        # 2. 构建上下文 (使用父块来构建上下文，信息更完整)
        item_name_context = self._prepare_llm_context(parent_chunks, item_name_chunk_k)

        # 3. 调用LLM模型 提取商品名
        item_features = self._recognition_item_name(item_name_context, file_title)

        # 4. 向量化(嵌入模型：bge(bge-m3))：混合向量[稠密：相似性匹配、稀疏：精确匹配]
        item_name, dense_vector, sparse_vector = self._embedding_item_name(item_features)

        # 5. 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_features, item_name)

        # 6. 回填(更新LLM提取到的item_name到父块和子块)
        self._fill_item_name(state, parent_chunks, child_chunks, item_features, item_name)

        return state

    def _validate_state(self, state) -> Tuple[str, List, List, int]:
        """
        校验并获取状态信息，适配父子切片结构
        """
        # 1. 获取文档标题
        file_title = state.get('file_title')
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name='file_title', expected_type=str)

        # 2. 获取父块和子块
        parent_chunks = state.get('parent_chunks')
        if not parent_chunks or not isinstance(parent_chunks, list):
            raise StateFieldError(node_name=self.name, field_name='parent_chunks', expected_type=list)

        child_chunks = state.get('child_chunks')
        if not child_chunks or not isinstance(child_chunks, list):
            raise StateFieldError(node_name=self.name, field_name='child_chunks', expected_type=list)

        # 3. 获取 item_name_chunk_k
        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="商品名识别使用的切片数不合法")

        return file_title, parent_chunks, child_chunks, item_name_chunk_k

    def _prepare_llm_context(self, chunks: List[Dict], item_name_chunk_k: int) -> str:
        """
        准备商品名识别的上下文
        """
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get('content')
            splice_context = f"【切片】- f{index}- {content}"
            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> dict:
        # (代码保持不变)
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败, 降级: {str(e)}")
            return {"brand": None, "model": file_title, "category": None}

        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            llm_result_str = llm_response.content.strip('` \n')
            if llm_result_str.startswith("json"):
                llm_result_str = llm_result_str[4:]

            result_dict = json.loads(llm_result_str)
            self.logger.info(f"LLM为文档：{file_title} 提取的结构化特征：{result_dict}")
            return result_dict

        except Exception as e:
            self.logger.error(f"LLM提取特征失败，降级使用文件标题: {str(e)}")
            return {"brand": None, "model": file_title, "category": None}

    def _embedding_item_name(self, item_features: dict) -> Tuple[
        Optional[str], Optional[List], Optional[Dict[str, Any]]]:
        # (代码保持不变)
        brand = item_features.get("brand") or ""
        model = item_features.get("model") or ""
        category = item_features.get("category") or ""

        item_name = f"{brand} {model} {category}".strip()
        if not item_name:
            return None, None, None

        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE_M3嵌入模型客户端创建失败: {str(e)}")
            return None, None, None
        try:
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            dense_vector = vector_result.get('dense')[0].tolist()
            sparse_csr = vector_result.get('sparse')
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            weight = sparse_csr.data[start_index:end_index].tolist()

            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f"计算出来的稠密向量的维度: {len(dense_vector)}")
            return item_name, dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3嵌入模型计算{item_name}向量失败 原因：{str(e)}")
            return None, None, None

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_features: dict,
                       item_name: str):
        # (代码保持不变)
        if not dense_vector or not sparse_vector:
            return

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败,原因：{str(e)}")
            return

        item_name_collection_name = self.config.item_name_collection

        try:
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            item_name_data_row = {
                "file_title": file_title,
                "brand": item_features.get("brand") or "",
                "model": item_features.get("model") or "",
                "category": item_features.get("category") or "",
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            inserted_result = milvus_client.insert(collection_name=item_name_collection_name,
                                                   data=[item_name_data_row])
        except Exception as e:
            self.logger.error(f"商品插入失败 {str(e)}")

        self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")

    def _create_item_name_collection(self, item_name_collection_name: str, milvus_client: MilvusClient):
        # (代码保持不变)
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)

        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="model", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        milvus_client.create_collection(collection_name=item_name_collection_name,
                                        schema=schema, index_params=index_params)

        self.logger.info(f"创建{item_name_collection_name}集合成功")

    def _fill_item_name(self, state: ImportGraphState, parent_chunks: List[Dict], child_chunks: List[Dict],
                        item_features: dict, item_name: str):
        """
        回填 item_name 到父块和子块
        """
        # 给父块回填
        for chunk in parent_chunks:
            chunk['item_name'] = item_name

        # 给子块回填
        for chunk in child_chunks:
            chunk['item_name'] = item_name

        state['parent_chunks'] = parent_chunks
        state['child_chunks'] = child_chunks

        state['item_features'] = item_features
        state['item_name'] = item_name


if __name__ == '__main__':
    setup_logging()

    temp_dir = Path(r"D:\Project\llm-project\RAG_repository\repository\processor\import_processor\temp")

    # 注意：测试脚本这里也需要对应修改读取的文件，假设上游备份的名字叫 chunks_parent_child.json
    chunk_json_path = temp_dir / "chunks_parent_child.json"
    output_path = temp_dir / "chunks_item_name_parent_child.json"

    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 从之前上游备份的 json 中获取 parent 和 child 数组
    state = {
        "file_title": "万用表的使用",
        "parent_chunks": chunk_content.get("parent_chunks", []),
        "child_chunks": chunk_content.get("child_chunks", [])
    }

    node = ItemNameRecognitionNode()
    result = node.process(state)

    print(f"商品名: {result.get('item_name')}")
    print(f"父块数量: {len(result.get('parent_chunks', []))}")
    print(f"子块数量: {len(result.get('child_chunks', []))}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"item_name:{result.get('item_name')}生成完成，结果已保存至:\n{output_path}")