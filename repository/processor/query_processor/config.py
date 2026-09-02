"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== answer_output配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "8"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    # 多个产品分别断崖后融合个数的上限
    final_rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("FINAL_RERANKER_MAX_TOP_K", "16"))
    )
    # 断崖阈值起作用的最低gap值
    rerank_gap_threshold: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_THRESHOLD", "0.04"))
    )
    # 如果断崖阈值大于max_gap直接break不再找最大gap
    rerank_max_gap: float = field(
        default_factory=lambda: float(os.getenv("RERANK_MAX_GAP", "0.10"))
    )
    rerank_min_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("RERANK_MIN_SCORE_THRESHOLD", "0.3"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "40"))
    )
    rrf_hybrid_search_weight_precise: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYBRID_SEARCH_WEIGHT_PRECISE", "1.0"))
    )
    rrf_hyde_search_weight_precise: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYDE_SEARCH_WEIGHT_PRECISE", "0.4"))
    )
    rrf_hybrid_search_weight_fuzzy: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYBRID_SEARCH_WEIGHT_FUZZY", "0.5"))
    )
    rrf_hyde_search_weight_fuzzy: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYDE_SEARCH_WEIGHT_FUZZY", "0.5"))
    )

    # ==================== Web检索配置 ====================
    web_search_limit: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_LIMIT", "4"))
    )

    # ==================== Hyde检索配置 ====================
    hyde_search_limit_per_req: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT_PER_REQ", "30"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "30"))
    )
    hyde_search_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("HYDE_SEARCH_DENSE_WEIGHT", "0.7"))
    )
    hyde_search_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("HYDE_SEARCH_SPARSE_WEIGHT_PRECISE", "0.3"))
    )

    # ==================== hybrid检索配置 ====================
    hybrid_search_limit_per_req: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_SEARCH_LIMIT_PER_REQ", "30"))
    )
    hybrid_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_SEARCH_LIMIT", "30"))
    )
    hybrid_search_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_SEARCH_DENSE_WEIGHT", "0.3"))
    )
    hybrid_search_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_SEARCH_SPARSE_WEIGHT", "0.7"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_CONFIDENCE", "0.6"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.2"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.8"))
    )
    item_name_search_limit_per_req: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_SEARCH_LIMIT_PER_REQ", "5"))
    )
    item_name_search_limit: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_SEARCH_LIMIT", "5"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    child_chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHILD_CHUNKS_COLLECTION", "")
    )
    parent_chunks_collection: str = field(
        default_factory=lambda: os.getenv("PARENT_CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )
    mcp_dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_API_KEY", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config