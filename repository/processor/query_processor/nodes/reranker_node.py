from typing import Tuple, List, Dict, Any
import math
import re
from FlagEmbedding import FlagReranker
from repository.processor.query_processor.base import BaseNode, T
from repository.processor.query_processor.state import QueryGraphState
from repository.utils.client.ai_clients import AIClients
from repository.utils.client.storage_clients import StorageClients


class RerankerNode(BaseNode):
    name = "reranker_node"

    @staticmethod
    def _sigmoid(score: float) -> float:
        """sigmoid归一化，将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """

        Args:
            state:

        Returns:

        """

        # 1. 获取用户问题
        user_query = state.get('rewritten_query') or state.get('original_query')

        # 2. 获取两路检索结果(本地检索结果、远程检索结果)
        rerank_outputs: List[Dict[str, Any]] = self._collect_rerank_inputs(state)

        # 3. 利用Reranker进行精排
        refine_docs: List[Dict[str, Any]] = self._refine_rank(user_query, rerank_outputs)

        # 4. 动态配额与断崖截断 (对子切片执行淘汰机制)
        reranked_child_docs = self.filter_reranked_docs_by_entity(
            refine_docs,
            state.get('item_names', []),
            self.config.rerank_min_top_k,
            self.config.rerank_max_top_k
        )

        # 5. 拿幸存的高分子切片，去换取完整的父切片大段落
        final_parent_docs = self._convert_to_parent_chunks_post_rerank(reranked_child_docs)

        state['reranked_docs'] = final_parent_docs

        self.logger.info(f"Reranker最终输出给大模型的长上下文切片个数: {len(final_parent_docs)}")

        return state

    def _collect_rerank_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """
        获取两路检索结果(本地检索结果、远程检索结果)
        Args:
            state:

        Returns:

        """
        final_docs = []
        # 1. 获取本地检索结果
        rrf_chunks = state.get('rrf_chunks') or []
        for chunk in rrf_chunks:
            # 1. 判断chunk
            if not chunk or not isinstance(chunk, dict):
                continue

            # 2. 获取chunk的信息
            # 2.1 获取chunk中content
            content = chunk.get('content', '')
            if not content:
                continue

            #  Markdown 图片链接
            content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
            # 清洗 HTML 图片标签 (可选)
            content = re.sub(r'<img.*?>', '', content)
            content = content.strip()

            # 【重要】把多余的换行符和空格压缩掉
            content = re.sub(r'\n+', '\n', content).strip()

            # 如果清洗完图片后，文本过短（比如只剩换行符或一个孤零零的无意义短标题），直接过滤掉
            if len(content) < 15:
                continue

            # 2.2 获取chunk中的title、chunk_id、parent_id
            title = chunk.get('title', '')
            chunk_id = chunk.get('chunk_id')
            parent_id = chunk.get('parent_id')
            item_name = chunk.get('item_name', '')

            # 3. 格式化文档(格式化本地)
            formated_local_doc = self._format_doc(
                content=content,
                chunk_id=chunk_id,
                title=title,
                source="local",
                parent_id=parent_id,  # 传入 parent_id
                item_name=item_name
            )

            final_docs.append(formated_local_doc)

        # 2. 获取远程检索结果
        web_search_docs = state.get('web_search_docs') or []
        for doc in web_search_docs:

            # 1. 判断doc
            if not doc or not isinstance(doc, dict):
                continue

            # 2. 获取content
            content = doc.get('snippet', '')
            # 2.3 获取title
            title = doc.get('title', '')
            # 2.4 获取url
            url = doc.get('url', '')

            # 3. 格式化文档(格式化远程)
            formated_web_doc = self._format_doc(content=content, title=title, url=url, source="web")
            final_docs.append(formated_web_doc)

        self.logger.info(f"获取Reranker阶段需要的切片（子切片+Web）个数{len(final_docs)}")
        return final_docs

    def _format_doc(self, content: str, chunk_id: str = None, title: str = "", url: str = "", source: str = "", parent_id: str = None, item_name:str = ""):
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source,
            "parent_id": parent_id,
            "item_name": item_name
        }

    def _refine_rank(self, user_query: str, rerank_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        reranker模型进行打分&排序【精排】
        Args:
            user_query:  用户的查询
            rerank_outputs: 本地和远程融合后的检索结果

        Returns:
            Dict[str,Any]:{"score","","other":"..."}

        """
        if not rerank_outputs:
            return []

        # 1. 获取重排序模型
        try:
            rerank_client: FlagReranker = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f"获取BGE-M3-Reranker模型失败 原因:{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

        # 3. 构建Q->D的pair对,把设备名注入到文本开头,拉高参数表格的语义得分   标记
        query_doc_pairs = [(user_query, d.get('content')) for d in rerank_outputs]
        query_doc_pairs = []
        for d in rerank_outputs:
            item_name = d.get('item_name', '')
            content = d.get('content', '')
            # 拼接格式："设备：xxx \n 内容：xxx"
            enriched_content = f"【设备型号】{item_name}\n{content}" if item_name else content
            query_doc_pairs.append((user_query, enriched_content))

        try:
            # 4. 计算(注意：BGE-M3重排序模型计算出来的得分可以很大也可以很小(负无穷大,正无穷大))
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)

            # 5.组合最终结果
            doc_score = [{**d, "score": self._sigmoid(float(s))} for d, s in zip(rerank_outputs, rerank_scores)]

            # 6. 排序
            sorted_doc_score = sorted(doc_score, key=lambda x: x['score'], reverse=True)

            # 7. 阈值过滤
            filtered_doc_score = [
                doc for doc in sorted_doc_score
                if doc['score'] > self.config.rerank_min_score_threshold
            ]
            self.logger.info(f"获取Reranker后满足条件的切片结果个数{len(filtered_doc_score)}")

            # 8. 返回
            return filtered_doc_score
        except Exception as e:
            self.logger.error(f"BGE-M3重排序模型计算分数失败 原因：{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

    def _convert_to_parent_chunks_post_rerank(self, reranked_child_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在重排和截断全部完成之后，用胜出的子切片去换取父切片原文。
        """

        local_docs = [d for d in reranked_child_docs if d.get('source') == 'local']
        web_docs = [d for d in reranked_child_docs if d.get('source') == 'web']

        # 建立 parent_id 到子切片(包含score)的映射，方便后面继承分数
        parent_to_best_child = {}
        for doc in local_docs:
            pid = doc.get("parent_id")
            if pid and pid not in parent_to_best_child:
                parent_to_best_child[pid] = doc

        # 提取 parent_id
        parent_ids = list(parent_to_best_child.keys())

        final_docs = []
        final_docs.extend(web_docs)  # Web文档无父子概念，直接放行

        if not parent_ids:
            return final_docs

        try:
            milvus_client = StorageClients.get_milvus_client()
            parent_collection = getattr(self.config, 'parent_chunks_collection', 'parent_chunks')

            formatted_ids = ",".join([f"'{pid}'" for pid in parent_ids])
            expr = f"chunk_id in [{formatted_ids}]"

            self.logger.info(f"正在溯源，用 {len(local_docs)} 个子切片召回 {len(parent_ids)} 个唯一父切片...")
            parent_results = milvus_client.query(
                collection_name=parent_collection,
                filter=expr,
                output_fields=["chunk_id", "content", "title", "file_title", "item_name"]
            )

            # 组装完整的父切片，【关键：继承最高分子切片的分数和原始设备名】
            for p_chunk in parent_results:
                pid = p_chunk.get("chunk_id")
                best_child = parent_to_best_child.get(pid, {})

                # 清洗刚从数据库捞出来的父切片文本    标记
                raw_content = p_chunk.get("content", "")
                clean_content = re.sub(r'!\[.*?\]\(.*?\)', '', raw_content)
                clean_content = re.sub(r'<img.*?>', '', clean_content)
                clean_content = re.sub(r'\n+', '\n', clean_content).strip()

                final_docs.append({
                    "content": clean_content,
                    "chunk_id": pid,
                    "title": p_chunk.get("title", ""),
                    "source": "local_parent",
                    "item_name": p_chunk.get("item_name", "") or best_child.get("item_name", ""),
                    "score": best_child.get("score", 0.0)
                })

            # 2. 必须重排一次，因为 Milvus 返回的 parent_results 是无序的
            final_docs = sorted(final_docs, key=lambda x: x.get('score', 0), reverse=True)
        except Exception as e:
            self.logger.error(f"召回父切片失败，降级使用短子块: {e}")
            final_docs.extend(local_docs)

        return final_docs

    def filter_reranked_docs_by_entity(self, reranked_docs: list[dict], confirmed_items: list[str],
                                       rerank_min_top_k: int = 3, rerank_max_top_k: int = 8) -> list[dict]:
        """
        当存在多个设备对比时，对每个设备分别执行断崖截断，保证每个设备都有平等的、动态的上下文配额。
        对于 Web 召回的结果，通过文本字面匹配将其归类到对应设备中。
        """
        if len(confirmed_items) <= 1:
            # 单设备查询或未识别出设备：保持原有的全局断崖截断逻辑
            self.logger.info(f"仅单设备查询，执行全局断崖截断")
            return self._cliff_cutoff(reranked_docs, rerank_min_top_k, rerank_max_top_k)

        self.logger.info(f"存在 {len(confirmed_items)} 个设备对比，对每个设备独立执行动态断崖截断")

        # 1. 准备分桶 (保持原有的降序顺序)
        item_buckets = {item: [] for item in confirmed_items}

        # 2. 将文档发牌到对应的设备桶中
        for doc in reranked_docs:
            for item in confirmed_items:
                # 1. 强元数据匹配：取自带的 item_name
                doc_item = doc.get("item_name") or doc.get("metadata", {}).get("item_name")

                # 2. 提取核心型号
                item_parts = item.split()
                model_candidate = max(item_parts, key=len) if item_parts else item

                # 提取字母数字混合的核心前缀（忽略 - / 等符号）
                # 比如把 HF46F-24-HS1 变成 HF46F
                match = re.match(r'([A-Za-z]+[0-9]+[A-Za-z]*)', model_candidate)
                core_model = match.group(1) if match else model_candidate[:5]  # 兜底取前5个字符

                # 清理被检查文本中的常见分隔符，方便比对
                clean_content = re.sub(r'[-/_\s]', '', doc.get("content", ""))
                clean_core = re.sub(r'[-/_\s]', '', core_model)

                # 3. 构建多重匹配条件（满足其一即可入桶）
                cond_exact = (doc_item == item)
                cond_full = (item in doc.get("content", "") or item in doc.get("title", ""))

                # 只要核心型号长度>=4，且在清理掉符号的内容中能找到，就算命中
                cond_core = (len(clean_core) >= 4 and clean_core in clean_content)

                if cond_exact or cond_full or cond_core:
                    item_buckets[item].append(doc)

        # 3. 对每个桶独立执行断崖截断，并使用 dict 去重
        # (因为一篇对比评测的 Web 文章可能同时进入了 A 桶和 B 桶，避免给 LLM 喂重复数据)
        final_docs_dict = {}

        for item, docs in item_buckets.items():
            if not docs:
                continue
            # 因为原列表 reranked_docs 是按 score 降序的，所以分发到 bucket 里的 docs 也是天然降序的，可以直接断崖
            self.logger.info(f"对设备 [{item}] 的 {len(docs)} 个候选文档进行截断:")
            cutoff_docs = self._cliff_cutoff(docs, rerank_min_top_k, rerank_max_top_k)
            for d in cutoff_docs:
                final_docs_dict[id(d)] = d  # 用 Python 对象的内存地址作为唯一键，极其安全且防重


        # 4. 组装并重新进行一次全局降序排序
        final_docs = sorted(list(final_docs_dict.values()), key=lambda x: x.get('score', 0), reverse=True)

        # 5. 哪怕每个产品都拿到了自己的配额，为了防止大模型超载，做一个硬上限控制,比如最多允许输入 10 个长切片（10000字左右）
        safe_limit = self.config.final_rerank_max_top_k
        if len(final_docs) > safe_limit:
            self.logger.warning(f"触发全局截断机制：总切片数 {len(final_docs)} 超载，强制截断至 {safe_limit} 个")
            final_docs = final_docs[:safe_limit]

        self.logger.info(f"断崖阶段后的子切片的总个数{len(final_docs)}")

        return final_docs

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]], rerank_min_top_k: int, rerank_max_top_k: int) -> List[
        Dict[str, Any]]:
        """
          动态top_k: 文档分数归一化后只需一个断崖阈值(rerank_gap_threshold)
          从头开始寻找最大断崖点，再用 min_top_k 兜底，把最大断崖之前的文档保留下来
        """
        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cut_off = upper_bound
        max_gap = 0

        # 从第0个间隔开始遍历，找全局最大断崖
        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = refine_docs[i].get('score')
            next_score = refine_docs[i + 1].get('score')

            if current_score is None or next_score is None:
                continue

            gap = current_score - next_score

            if gap >= self.config.rerank_gap_threshold and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                self.logger.info(f"位置{i + 1}发生最大断崖，断崖差值: {gap:.4f}")
                if gap >= self.config.rerank_max_gap:
                    break

        # 兜底：不管断崖在哪，至少保留 lower_bound 个
        cut_off = max(cut_off, lower_bound)

        cutoff_docs = refine_docs[:cut_off]
        self.logger.info(f"断崖阶段后的子切片个数{len(cutoff_docs)}")
        return cutoff_docs

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)
    print(result['reranked_docs'])
