"""
节点名称：item_name_confirmed_node (意图识别与路由节点)

核心逻辑说明（新架构）：
1. 结构化意图提取：利用 LLM 将用户的自然语言查询转换为包含明确特征（品牌、型号、类别）的 JSON 数组，支持单次提问中包含【多个产品】的对比。
2. 领域安全拦截：通过 `is_domain_relevant` 字段，前置拦截闲聊、天气等非专业领域问题，节省 Token 与检索资源。
3. 官方型号对齐（归一化）：对于提取出的具体型号（如 "350-24"），到 Milvus 的商品名库中进行碰撞，将其修正为官方标准名（如 "LRS-350-24"）。
   * 采用【顺延占位机制】，避免两个相似的口语化产品被映射到同一个官方标准名上。
4. 三路路由决策（Routing）：
   - 无关问题，或完全没有提取到有效特征，直接结束。
   - 有具体型号且在本地知识库中找到了高置信度的映射，走本地高精度检索。
   - 无具体型号的走泛化查询
"""

import logging, re, json
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from repository.processor.query_processor.base import BaseNode
from repository.processor.query_processor.state import QueryGraphState
from repository.utils.client.ai_clients import AIClients
from repository.utils.client.storage_clients import StorageClients
from repository.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE, ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE
from repository.utils.embedding_util import generate_bge_m3_hybrid_vectors
from repository.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from repository.processor.query_processor.config import get_config
from repository.utils.mongo_history_util import get_recent_messages


class _ItemNameExtractor:

    def extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """保持签名不变"""
        llm_result = {
            "item_names": [],
            "rewritten_query": original_query,
            "is_domain_relevant": True,
            "query_features_list": []
        }

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"LLM客户端获取失败 原因:{str(e)}")
            return llm_result

        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context.strip() if history_context else "暂无历史上下文",
            query=original_query)

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE),
                HumanMessage(content=item_name_user_prompt)
            ])
        except Exception as e:
            logger.error(f"LLM调用失败,原因：{str(e)}")
            return llm_result

        parsed_result = self._clean_and_parse(llm_response.content)
        if not parsed_result:
            return llm_result

        # 解析多商品特征
        products = parsed_result.get("products", [])
        if not isinstance(products, list):
            products = []

        item_names = []
        for p in products:
            if not isinstance(p, dict): continue
            brand = p.get("brand")
            model = p.get("model")
            category = p.get("category")

            # 为每个产品单独拼装一个检索长词，即使没有model也会拼装（供下游Web搜索备用）
            combined_name = f"{brand or ''} {model or ''} {category or ''}".strip()
            if combined_name:
                item_names.append(combined_name)

        llm_result['item_names'] = item_names
        llm_result['rewritten_query'] = parsed_result.get('rewritten_query', original_query)
        llm_result['is_domain_relevant'] = parsed_result.get('is_domain_relevant', True)
        llm_result['query_features_list'] = products
        return llm_result

    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        """保持签名不变"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(content)
        except JSONDecodeError as e:
            logger.error(f"llm输出结果{llm_response_content} 反序列化失败 原因:{str(e)}")
            return {}


class _ItemNameAligner:
    def __init__(self):
        self._config = get_config()

    def search_and_align(self, search_item_names: List[str]) -> List[str]:
        """返回 confirmed (List[str])"""
        search_result: List[Dict[str, Any]] = self._search_vector(search_item_names)
        if not search_result:
            return []
        confirmed = self._align(search_result)
        return confirmed

    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """保持签名不变"""
        final_search_result = []
        try:
            milvus_client = StorageClients.get_milvus_client()
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"客户端获取失败 原因:{str(e)}")
            return final_search_result

        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=item_names)
        except Exception as e:
            logger.error(f"生成混合向量失败 原因:{str(e)} ")
            return final_search_result

        for index, item_name in enumerate(item_names):
            hybrid_requests = create_hybrid_search_requests(
                hybrid_vector_result['dense'][index],
                hybrid_vector_result['sparse'][index],
                limit=self._config.item_name_search_limit_per_req,
            )

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=hybrid_requests,
                ranker_weights=(self._config.item_name_dense_weight, self._config.item_name_sparse_weight),
                norm_score=True,
                limit=self._config.item_name_search_limit,
                output_fields=['item_name']
            )

            matches = [{"score": res['distance'], "item_name": res['entity'].get('item_name', '')} for res in
                       (hybrid_search_result[0] if hybrid_search_result else [])]

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })
        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> List[str]:
        """【顺延占位版】：遍历所有的搜索结果，遇到碰撞自动顺延取 Top2/Top3"""
        confirmed = []
        # 兼容旧配置并引入新的单一置信度
        confidence_threshold = self._config.item_name_confidence

        for item_sea_res in search_result:
            item_name_matches = item_sea_res.get('matches', [])
            if not item_name_matches:
                continue

            # 按分数降序排列
            item_name_matches_sorted = sorted(item_name_matches, key=lambda x: x['score'], reverse=True)

            # 遍历当前提取产品的候选列表，寻找【达标】且【未被占用】的型号
            for match in item_name_matches_sorted:
                # 只有分数大于置信度的才有资格被选中
                if match.get('score') > confidence_threshold:
                    picked = match.get('item_name')
                    # 如果该型号有效且没有被前面的产品认领走
                    if picked and picked not in confirmed:
                        confirmed.append(picked)
                        break  # 占坑成功，立刻跳出内层循环，去处理下一个实体产品
                else:
                    # 因为是降序排列，如果当前这个分数已经不够了，后面的肯定更低，直接抛弃
                    break

        return confirmed


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state.get('original_query', '')
        session_id = state.get('session_id')

        history_context = get_recent_messages(session_id=session_id, limit=10)
        formatted_history = " ".join([f"角色:{h.get('role', '')},内容:{h.get('text', '')}" for h in history_context])

        llm_result: Dict[str, Any] = self._extractor.extract_item_name(original_query, formatted_history)

        # 局部变量提取
        is_relevant = llm_result.get('is_domain_relevant', True)
        products = llm_result.get('query_features_list', [])
        item_names = llm_result.get('item_names', [])
        rewritten_query = llm_result.get('rewritten_query')

        # 【新增过滤】：只有包含 model 的产品，才会被装填进向量检索列表
        search_item_names = []
        for p in products:
            if p.get('model'):
                brand = p.get("brand")
                model = p.get("model")
                category = p.get("category")
                name_str = f"{brand or ''} {model or ''} {category or ''}".strip()
                if name_str:
                    search_item_names.append(name_str)

        # 仅对存在 model 的产品进行向量数据库撞库对齐
        if search_item_names:
            confirmed = self._aligner.search_and_align(search_item_names)
        else:
            confirmed = []

        # 核心决策分发（去除了 options 参数）
        self._decide(confirmed, state, rewritten_query, item_names, is_relevant, products)

        state['history'] = history_context
        return state

    def _decide(self, confirmed: List[str], state: QueryGraphState,
                rewritten_query: str, item_names: List[str],
                is_relevant: bool, products: List[Dict[str, Any]]):
        """【精简决策分支】：执行明确的三路路由"""

        has_any_feature = False
        has_any_model = False
        for p in products:
            if p.get('brand') or p.get('model') or p.get('category'):
                has_any_feature = True
            if p.get('model'):
                has_any_model = True

        # === 规则 1：不相关，或者 (品牌、型号、类型) 都没有 -> 直接 End ===
        if not is_relevant or not has_any_feature:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称或需求，请提供更准确的信息。"
            return

        # === 规则 3：提取到了model且库内对齐分数高于置信度 -> 走高精检索===
        if has_any_model and confirmed:
            state['item_names'] = confirmed  # 放入成功对齐后的官方名称列表
            state['rewritten_query'] = rewritten_query
            state["query_type"] = "precise"
            return

        # === 规则 2：有model但分数低(confirmed为空)，或者没model但有brand/category -> 泛化查询 ===
        state['item_names'] = item_names  # 放入原始拼接词
        state['rewritten_query'] = rewritten_query
        state["query_type"] = "fuzzy"

if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        # "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)
