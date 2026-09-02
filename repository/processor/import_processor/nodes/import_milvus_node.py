from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Sequence
from pymilvus import MilvusClient, DataType
from repository.processor.import_processor.base import BaseNode, setup_logging
from repository.processor.import_processor.state import ImportGraphState
from repository.processor.import_processor.exceptions import StateFieldError, ValidationError, MilvusError
from repository.utils.client.storage_clients import StorageClients


# ================= 标量字段定义 =================
@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 子切片通用的标量字段 (新增 parent_id)
_CHILD_SCALAR_FIELDS: Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="parent_id", datatype=DataType.VARCHAR, max_length=128),  # 【核心新增】：指向父块
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)

# 父切片的标量字段 (极其精简，用于溯源提取长文本)
_PARENT_SCALAR_FIELDS: Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder():
    """负责处理和Milvus字段约束相关的逻辑"""

    @staticmethod
    def build_child_schema(milvus_client: MilvusClient, dim: int):
        """创建子切片集合的 schema (带向量)"""
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        # 子切片主键：保持自增不变 (或者是切分时生成的 string UUID 也可以，这里为了最小改动保留自增)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)

        # 向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 标量字段
        for spec in _CHILD_SCALAR_FIELDS:
            kwargs = {"field_name": spec.field_name, "datatype": spec.datatype}
            if spec.max_length:
                kwargs['max_length'] = spec.max_length
            schema.add_field(**kwargs)
        return schema

    @staticmethod
    def build_parent_schema(milvus_client: MilvusClient):
        """创建父切片集合的 schema (纯文本，无向量)"""
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 【关键】：父切片的主键必须是我们自己生成的那个 UUID (字符串)，因为子切片要通过 parent_id 找它
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)

        # Milvus 的底层铁律：只要一个 Collection 被创建，它的 Schema 里就必须包含至少一个向量字段（Float Vector 或 Sparse Float Vector），并且该向量字段必须为其配置对应的索引参数。
        # 塞入一个维度为 2 的极小哑向量，永远存 [0.0, 0.0]，永不使用它做检索
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=2)

        for spec in _PARENT_SCALAR_FIELDS:
            kwargs = {"field_name": spec.field_name, "datatype": spec.datatype}
            if spec.max_length:
                kwargs['max_length'] = spec.max_length
            schema.add_field(**kwargs)
        return schema


class _MilvusIndexBuilder:
    """负责处理和Milvus索引相关的逻辑"""

    @staticmethod
    def build_child_index_params(milvus_client: MilvusClient):
        """子切片需要建立复杂的混合向量索引"""
        index = milvus_client.prepare_index_params()
        index.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX",
                        metric_type="COSINE")
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return index

    @staticmethod
    def build_parent_index_params(milvus_client: MilvusClient):
        """父切片由于只做主键精确查询，Milvus 默认会对主键建索引，这里给父切片的哑向量补上最基础的索引"""
        index = milvus_client.prepare_index_params()
        index.add_index(field_name="dense_vector", index_name="parent_dense_index", index_type="AUTOINDEX",
                        metric_type="COSINE")
        return index


class _MilvusInserter:
    """负责处理和Milvus插入数据相关的逻辑"""

    def __init__(self, milvus_client: MilvusClient):
        self._milvus_client = milvus_client

    def insert_rows(self, collection_name: str, data: List[Dict[str, Any]]):
        if not data:
            return []
        # 1. 插入
        inserted_result = self._milvus_client.insert(collection_name=collection_name, data=data)
        # 2. 返回主键列表
        return inserted_result.get('ids', [])


class ImportMilvusNode(BaseNode):
    """充当门面（门面模式）"""
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 获取Milvus客户端
        self.log_step("init", "获取Milvus客户端")
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"MilVus客户端创建失败,异常原因{str(e)}")
            raise MilvusError(message=f"MilVus客户端创建失败,异常原因{str(e)}", node_name=self.name)

        # 2. 获取集合名称
        # 假设你在 config 中已经配置了这两个变量，如果没有，请去 config.py 中添加！
        child_collection = getattr(self.config, 'child_chunks_collection', 'child_chunks')
        parent_collection = getattr(self.config, 'parent_chunks_collection', 'parent_chunks')

        # 3. 校验并处理子切片 (Child Chunks)
        self.log_step("process_child", "处理子切片入库...")
        validated_children, dim = self._validate_child_state(state)
        self._create_child_collection(child_collection, milvus_client, dim)
        _inserter = _MilvusInserter(milvus_client)
        _inserter.insert_rows(child_collection, validated_children)

        # 4. 校验并处理父切片 (Parent Chunks)
        self.log_step("process_parent", "处理父切片入库...")
        validated_parents = self._validate_parent_state(state)
        self._create_parent_collection(parent_collection, milvus_client)
        _inserter.insert_rows(parent_collection, validated_parents)

        self.logger.info("父子切片分离入库全部完成！")
        return state

    def _validate_child_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        """专门校验包含向量的子切片"""
        child_chunks = state.get("child_chunks")
        if not child_chunks or not isinstance(child_chunks, list):
            raise StateFieldError("待入库的 child_chunks 为空或类型无效", self.name)

        validated_chunks = []
        for i, chunk in enumerate(child_chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(f"child_chunks[{i}] 类型无效", self.name)

            # 必须要有 parent_id 才能认祖归宗
            if not chunk.get("parent_id"):
                raise ValidationError(f"child_chunks[{i}] 缺失致命的 parent_id", self.name)

            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                # 如果我们在切分时没有给子块生成 chunk_id，这里要跳过，因为建表时我们用了 VARCHAR
                if not chunk.get("chunk_id"):
                    import uuid
                    chunk['chunk_id'] = str(uuid.uuid4())
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"child_chunks[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError("所有 child_chunk 均无有效向量，无法入库", self.name)

        dim = len(validated_chunks[0]["dense_vector"])
        return validated_chunks, dim

    def _validate_parent_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        """专门校验不需要向量的长文本父切片"""
        parent_chunks = state.get("parent_chunks")
        if not parent_chunks or not isinstance(parent_chunks, list):
            raise StateFieldError("待入库的 parent_chunks 为空或类型无效", self.name)

        validated_chunks = []
        for i, chunk in enumerate(parent_chunks):
            if not isinstance(chunk, dict):
                continue
            # 父块最重要的是自己的 ID
            if not chunk.get("chunk_id"):
                self.logger.warning(f"parent_chunks[{i}] 缺失 chunk_id，将被忽略。")
                continue

            # 复制一份干净的字典，并强制塞入一个 2 维的哑向量满足 Milvus 建表校验
            clean_chunk = {k: v for k, v in chunk.items() if k not in ["sparse_vector"]}
            clean_chunk["dense_vector"] = [0.0, 0.0]
            validated_chunks.append(clean_chunk)

        return validated_chunks

    def _create_child_collection(self, collection_name: str, milvus_client: MilvusClient, dim: int):
        if milvus_client.has_collection(collection_name):
            self.logger.info(f"子切片集合 {collection_name} 已存在")
            return
        schema = _MilvusSchemaBuilder.build_child_schema(milvus_client, dim)
        index_params = _MilvusIndexBuilder.build_child_index_params(milvus_client)
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    def _create_parent_collection(self, collection_name: str, milvus_client: MilvusClient):
        if milvus_client.has_collection(collection_name):
            self.logger.info(f"父切片集合 {collection_name} 已存在")
            return
        schema = _MilvusSchemaBuilder.build_parent_schema(milvus_client)
        index_params = _MilvusIndexBuilder.build_parent_index_params(milvus_client)
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def _cli_main() -> None:
    import json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(r"D:\Project\llm-project\RAG_repository\repository\processor\import_processor\temp")

    # 模拟读取经过了 embedding 之后的父子 JSON 文件
    input_path = temp_dir / "chunks_vector_parent_child.json"
    output_path = temp_dir / "chunks_final_success.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 装载状态机
    state: ImportGraphState = {
        "parent_chunks": content.get('parent_chunks'),
        "child_chunks": content.get('child_chunks')
    }

    # 为了本地测试不报错，临时给 config 注入两个表名（实际中你应该在 config.py 中定义它们）
    node = ImportMilvusNode()
    if not hasattr(node.config, 'child_chunks_collection'):
        node.config.child_chunks_collection = "child_chunks_v1"
    if not hasattr(node.config, 'parent_chunks_collection'):
        node.config.parent_chunks_collection = "parent_chunks_v1"

    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"父子切片入库完成，结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()