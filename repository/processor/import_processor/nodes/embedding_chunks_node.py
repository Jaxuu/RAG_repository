from typing import List, Dict, Any, Tuple
from pathlib import Path
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from repository.processor.import_processor.base import BaseNode, setup_logging, T
from repository.processor.import_processor.state import ImportGraphState
from repository.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from repository.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验 state (现在要拿 parent 和 child 两个集合)
        self.log_step("step1", "校验父子切片的数据结构")
        parent_chunks, child_chunks = self._validate_state(state)

        # 2. 获取嵌入模型
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)

        # 3. 批量嵌入 (核心修改：只对 child_chunks 进行向量化)
        batch_size = self.config.embedding_batch_size
        total_children = len(child_chunks)
        final_child_chunks = []

        self.logger.info(f"开始对子切片进行向量化，父切片跳过。子切片总数: {total_children}")

        for index in range(0, total_children, batch_size):
            batch_chunks = child_chunks[index:index + batch_size]
            batch_end = index + len(batch_chunks)
            self.logger.info(f"子切片嵌入批次 [{index + 1}-{batch_end}] / {total_children}")

            current_chunks = self._embed_chunks(batch_chunks, embed_model)
            final_child_chunks.extend(current_chunks)

        # 4. 更新 state (父切片原样返回，子切片带上了向量)
        state['parent_chunks'] = parent_chunks
        state['child_chunks'] = final_child_chunks

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # 1. 获取并校验父块
        parent_chunks = state.get('parent_chunks')
        if not parent_chunks or not isinstance(parent_chunks, list):
            raise StateFieldError(node_name=self.name, field_name="parent_chunks", expected_type=list)

        # 2. 获取并校验子块
        child_chunks = state.get('child_chunks')
        if not child_chunks or not isinstance(child_chunks, list):
            raise StateFieldError(node_name=self.name, field_name="child_chunks", expected_type=list)

        # 这里你可以加一些针对单个 chunk 是否是 dict 的循环校验，如果嫌啰嗦也可以省略
        return parent_chunks, child_chunks

    def _embed_chunks(self, batch_chunks: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction) -> List[
        Dict[str, Any]]:
        """批量嵌入子块"""
        # 注意：这里我们依然使用 item_name 和子块的 content 拼接作为检索标的
        embedding_documents = [f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}" for chunk in batch_chunks]

        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}", node_name=self.name)

        if not embed_vector:
            raise EmbeddingError(message="嵌入结果不存在")

        sparse_csr = embed_vector.get('sparse')
        for i, chunk in enumerate(batch_chunks):
            chunk['dense_vector'] = embed_vector.get('dense')[i].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(sparse_csr, i)

        return batch_chunks

    def _extract_sparse_vector(self, sparse_csr, index: int):
        """从稀疏矩阵中提取当前chunk对象的稀疏向量"""
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        weight = sparse_csr.data[start_index:end_index].tolist()
        return dict(zip(token_id, weight))


if __name__ == '__main__':
    import json

    setup_logging()

    base_dir = Path(r"D:\Project\llm-project\RAG_repository\repository\processor\import_processor\temp")

    # 模拟读取经过了 item_name_recognition_node 之后的父子 JSON 文件
    input_path = base_dir / "chunks_item_name_parent_child.json"
    output_path = base_dir / "chunks_vector_parent_child.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    # 传入父子数据
    result_state = node.process({
        "parent_chunks": chunks_data.get('parent_chunks'),
        "child_chunks": chunks_data.get('child_chunks')
    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")