"""
RAGAS 断点续传评测脚本 (Resume Evaluation)
用于解决 API 欠费、限流导致的中断问题。
直接读取 CSV 中的 retrieved_contexts 和 response，只对缺失分数的行调用裁判大模型。
"""

import sys
from unittest.mock import MagicMock

# 垫片：拦截废弃 VertexAI 模块，防止 ragas 导入报错
if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
import os
import json
import asyncio
import threading
from typing import List
import pandas as pd
import numpy as np
from ast import literal_eval
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

try:
    from ragas.run_config import RunConfig
except ImportError:
    RunConfig = None

# ==================== 1. 环境配置 ====================
load_dotenv(override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE",
                                                     "https://dashscope.aliyuncs.com/compatible-mode/v1")
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "BAAI/bge-m3")
BGE_DEVICE = os.getenv("BGE_DEVICE", "cuda:0")

# 你的裁判模型 (这里配置你的 DeepSeek-V4-Pro)
judge_chat_model = ChatOpenAI(
    model=os.getenv("JUDGE_LLM_MODEL", "qwen-plus"),
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
    max_retries=5,
    timeout=180.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)
judge_llm = LangchainLLMWrapper(judge_chat_model)


# 线程安全 BGE-M3 包装器
class ThreadSafeBgeEmbeddings:
    def __init__(self, raw_embeddings):
        self._raw = raw_embeddings
        self._lock = threading.Lock()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with self._lock:
            return self._raw.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        with self._lock:
            return self._raw.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


raw_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=BGE_M3_PATH,
    model_kwargs={'device': BGE_DEVICE},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
)
judge_embeddings = LangchainEmbeddingsWrapper(ThreadSafeBgeEmbeddings(raw_bge_embeddings))

# 初始化指标
try:
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall
    )

    eval_metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm)
    ]
except ImportError:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall
    )

    eval_metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm)
    ]


# ==================== 2. 断点续传核心逻辑 ====================
def resume_evaluation(csv_path: str):
    print(f"📁 正在读取中断的文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 清理掉之前生成的汇总行 (如果存在)
    df = df[df['user_input'] != '[GLOBAL_AVERAGE_SUMMARY]'].copy()

    # 找到哪些行需要重新评测 (特征：faithfulness 为 NaN 或空白)
    missing_mask = df['faithfulness'].isna() | (df['faithfulness'] == '')
    df_missing = df[missing_mask].copy()

    if df_missing.empty:
        print("✅ 所有数据均已评测完毕，无需续传！")
        return

    print(f"🔍 发现 {len(df_missing)} 条未评测的数据 (如第 {df_missing.index[0] + 1} 条往后)，准备调用大模型评测...")

    # 构建 Ragas 需要的数据集结构
    # 注意：CSV 中读取出来的 retrieved_contexts 是字符串格式的列表 "['xxx', 'yyy']"，需要用 literal_eval 还原为真实的 Python List
    try:
        contexts_list = [literal_eval(ctx) if isinstance(ctx, str) else ctx for ctx in df_missing['retrieved_contexts']]
    except Exception as e:
        print("⚠️ 警告：无法解析 retrieved_contexts，尝试直接包装为列表。")
        contexts_list = [[ctx] for ctx in df_missing['retrieved_contexts']]

    dataset_dict = {
        "user_input": df_missing['user_input'].tolist(),
        "question": df_missing['user_input'].tolist(),
        "retrieved_contexts": contexts_list,
        "contexts": contexts_list,
        "response": df_missing['response'].tolist(),
        "answer": df_missing['response'].tolist(),
        "reference": df_missing['reference'].tolist(),
        "ground_truth": df_missing['reference'].tolist()
    }

    dataset = Dataset.from_dict(dataset_dict)

    # 启动 Ragas 评测 (这只会消耗少量金额，因为只有几十条)
    eval_kwargs = {
        "dataset": dataset,
        "metrics": eval_metrics,
        "llm": judge_llm,
        "embeddings": judge_embeddings,
    }
    if RunConfig is not None:
        eval_kwargs["run_config"] = RunConfig(max_workers=2, timeout=300, max_retries=5,max_wait=60)

    print("🚀 开始断点续传评测...")
    results = evaluate(**eval_kwargs)
    new_results_df = results.to_pandas()

    # ==================== 3. 结果合并与重新计算 ====================
    print("🔄 正在将新数据回填并重新计算全局平均分...")

    # 将新拿到的分数填回到原始大表中
    for index, new_row in new_results_df.iterrows():
        # df_missing 的 index 对应在原始 df 的绝对位置
        original_idx = df_missing.index[index]
        df.loc[original_idx, 'faithfulness'] = new_row.get('faithfulness', np.nan)
        df.loc[original_idx, 'answer_relevancy'] = new_row.get('answer_relevancy', np.nan)
        df.loc[original_idx, 'llm_context_precision_with_reference'] = new_row.get(
            'llm_context_precision_with_reference', np.nan)
        df.loc[original_idx, 'context_recall'] = new_row.get('context_recall', np.nan)

    # 仅计算 4 个核心 Ragas 指标的平均数
    metric_cols = ['faithfulness', 'answer_relevancy', 'llm_context_precision_with_reference', 'context_recall']
    avg_scores = df[metric_cols].mean(numeric_only=True).fillna(0.0).to_dict()

    # 写回到 JSON 摘要
    summary_path = os.path.join(os.path.dirname(csv_path), "ragas_summary_fixed.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(df),
            "average_scores": {k: round(float(v), 4) for k, v in avg_scores.items()}
        }, f, ensure_ascii=False, indent=2)

    # 追加汇总行
    avg_row = {col: "" for col in df.columns}
    avg_row['user_input'] = "[GLOBAL_AVERAGE_SUMMARY]"
    for col in metric_cols:
        avg_row[col] = round(float(avg_scores.get(col, 0.0)), 4)

    df_final = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # 另存为一个新的文件，防止覆盖出错
    final_csv_path = csv_path.replace(".csv", "_fixed.csv")
    df_final.to_csv(final_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n🎉 完美！修复后的最终表格已保存至: {final_csv_path}")
    print(f"📊 新的平均分摘要已保存至: {summary_path}")


if __name__ == "__main__":
    # 指向你中断的那个 CSV 文件
    interrupted_csv_file = "ragas_evaluation_report.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, interrupted_csv_file)

    resume_evaluation(target_path)