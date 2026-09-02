"""
RAGAS 评测系统核心指标说明 (Ragas Metrics Guide)

本脚本使用 Ragas 框架对 RAG 系统的检索 (Retrieval) 与生成 (Generation) 质量进行量化评估。
评估包含以下四个核心维度，得分区间均为 [0.0, 1.0]，分数越高代表表现越好。

1. 忠实度 (Faithfulness) - 评估生成质量 (防幻觉)
   - 物理含义：大模型生成的回答是否完全基于检索到的上下文，是否存在“胡编乱造”。
   - 计算原理：
     a. 使用 Judge LLM 将生成的回答 (Answer) 拆解为多个独立的事实陈述 (Claims)。
     b. 使用 Judge LLM 交叉比对，判断这些陈述是否都能从检索到的切片 (Contexts) 中找到确凿依据。
     c. 得分计算：[受上下文支持的陈述数量] / [回答中的总陈述数量]。

2. 回答相关度 (Answer Relevancy) - 评估生成质量 (防答非所问)
   - 物理含义：生成的回答与用户提出的原始问题是否紧密相关，是否直击痛点。
   - 计算原理：
     a. 不直接对比问题和答案的文本，而是让 Judge LLM 根据生成的回答 (Answer) 逆向推导出多个潜在的“原问题”。
     b. 使用 Embedding 模型计算这些逆向生成的问题与真实用户输入 (Question) 之间的余弦相似度。
     c. 相似度越高，说明回答越没有偏离主题，惩罚了长篇大论但毫无信息量的车轱辘话。

3. 上下文召回率 (Context Recall / LLMContextRecall) - 评估检索质量 (防漏检)
   - 物理含义：检索出的切片是否完整覆盖了正确解答该问题所需的所有知识点。
   - 计算原理：
     a. 使用 Judge LLM 将标准参考答案 (Ground Truth) 拆解为多条独立的核心事实语句。
     b. 逐一检查每一条标准事实是否都能在检索出的切片 (Contexts) 中找到对应内容。
     c. 得分计算：[在上下文中找到依据的标准事实数] / [标准答案的总事实数]。
     d. 如果此项得分低，说明混合检索系统没有把包含关键答案的文档捞出来。

4. 上下文精确率 (Context Precision / LLMContextPrecisionWithReference) - 评估排序质量 (防噪音干扰)
   - 物理含义：在检索出的所有切片中，真正有用的切片是否被 Reranker 排在了最前面。
   - 计算原理：
     a. 对比检索切片 (Contexts) 与标准答案 (Ground Truth)，判断每个切片是否相关（Relevant = 1 或 0）。
     b. 采用类似 MAP@K (Mean Average Precision) 的衰减算法进行计分。
     c. 如果相关切片排在第 1 位，得分极高；如果相关切片被挤到了第 5 位，或者混入了大量无关切片，得分会显著下降。

结果不好,我看了原因有三点，
一是检索到的上下文老是会有图片信息，这很大程度拉低了相关度;

二是这种模糊问题，参考答案和我的都符合，但是因为不同，所以结果就很差;

三是没检索到，比如第二个问题回答的是"宏发 HF46F/24-HS1 功率继电器的线圈工作电压为 24VDC，宏发 HF32FV-16-12-HLTF(590) 继电器的线圈工作电压在参考内容中未提及。"

针对这三种情况:
一我觉得可以在reranker开始阶段过滤一下，如果是图片的信息就过滤掉。
二我觉得样本里不要问这种推荐的问题了。
三我不知道怎么解决，是不是文档里确实是没有这个信息？你有什么建议？
"""

import sys
from unittest.mock import MagicMock

# 1. 垫片：拦截废弃 VertexAI 模块
if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

import os
import json
import asyncio
import threading
from typing import List
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

try:
    from ragas.run_config import RunConfig
except ImportError:
    RunConfig = None

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 引入项目查询图
from repository.processor.query_processor.main_graph import query_app

# ==================== 2. 环境配置与模型初始化 ====================
load_dotenv(override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "BAAI/bge-m3")
BGE_DEVICE = os.getenv("BGE_DEVICE", "cuda:0")

# 裁判 LLM
judge_chat_model = ChatOpenAI(
    model=os.getenv("JUDGE_LLM_MODEL", "qwen-plus"),
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
    max_retries=5,
    timeout=180.0,  # 延长超时时间
    model_kwargs={"response_format": {"type": "json_object"}}
)
judge_llm = LangchainLLMWrapper(judge_chat_model)


# 线程安全 BGE-M3 包装器：彻底解决 Windows/CUDA 在 asyncio 线程池中的资源竞争与死锁
class ThreadSafeBgeEmbeddings(Embeddings):
    def __init__(self, raw_embeddings: Embeddings):
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

# ==================== 3. 初始化 Ragas 评测指标 ====================
try:
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall
    )
    eval_metrics = [
        Faithfulness(llm=judge_llm),
        # 增加 strictness=1 参数，强制 n=1，绕过 API 限制
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
        # 这里的异常捕获分支也同步加上 strictness=1
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm)
    ]

# ==================== 4. 数据采集 ====================
def collect_rag_samples(test_cases_path: str) -> Dataset:
    if not os.path.exists(test_cases_path):
        raise FileNotFoundError(f"未找到测试集文件: {test_cases_path}")

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    user_inputs = []
    references = []
    responses = []
    retrieved_contexts = []

    print(f"🚀 开始采集评测数据，共 {len(test_cases)} 组测试用例...")

    for idx, item in enumerate(test_cases, 1):
        query = item["user_input"]
        ref = item["reference"]
        print(f"[{idx}/{len(test_cases)}] 正在执行查询: {query[:30]}...")

        state = query_app.invoke({
            "original_query": query,
            "session_id": f"eval-session-{idx}",  # 每次评测独立隔离，杜绝历史串扰
            "task_id": f"eval-task-{idx}",
            "is_stream": False
        })

        retrieved_docs = [
            chunk.get("content", "")
            for chunk in state.get("reranked_docs", [])
            if chunk.get("content")
        ]

        user_inputs.append(query)
        references.append(ref)
        responses.append(state.get("answer", ""))
        retrieved_contexts.append(retrieved_docs)

    return Dataset.from_dict({
        "user_input": user_inputs,
        "question": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "contexts": retrieved_contexts,
        "response": responses,
        "answer": responses,
        "reference": references,
        "ground_truth": references
    })


# ==================== 5. 纯表格与 JSON 数据结果保存 ====================
def save_evaluation_results(df: pd.DataFrame, output_dir: str):
    precision_col = next((c for c in df.columns if 'precision' in c), 'llm_context_precision_with_reference')
    recall_col = next((c for c in df.columns if 'recall' in c), 'context_recall')
    faith_col = next((c for c in df.columns if 'faithfulness' in c), 'faithfulness')
    rel_col = next((c for c in df.columns if 'relevan' in c), 'answer_relevancy')

    metric_cols = [c for c in [faith_col, rel_col, precision_col, recall_col] if c in df.columns]

    # 均值计算
    avg_scores = df[metric_cols].mean(numeric_only=True).fillna(0.0).to_dict()

    # 1. 导出 summary.json
    summary_path = os.path.join(output_dir, "ragas_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(df),
            "average_scores": {k: round(float(v), 4) for k, v in avg_scores.items()}
        }, f, ensure_ascii=False, indent=2)

    # 2. 追加 [GLOBAL_AVERAGE_SUMMARY] 汇总行并保存 CSV 明细
    avg_row = {col: "" for col in df.columns}
    avg_row['user_input'] = "[GLOBAL_AVERAGE_SUMMARY]"
    for col in metric_cols:
        avg_row[col] = round(float(avg_scores.get(col, 0.0)), 4)

    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    report_csv_path = os.path.join(output_dir, "ragas_evaluation_report.csv")
    df_with_avg.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 评测报告 (含均分汇总行) 已保存至: {report_csv_path}")
    print(f"✅ 评测摘要 JSON 已保存至: {summary_path}")


# ==================== 6. 主执行入口 ====================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_name = os.getenv("RAGAS_TESTSET", "test_cases_temp.json")
    test_cases_path = os.path.join(current_dir, json_file_name)

    dataset = collect_rag_samples(test_cases_path)

    print("\n🔍 正在调用 Judge LLM 与本地 BGE-M3 计算得分...")

    eval_kwargs = {
        "dataset": dataset,
        "metrics": eval_metrics,
        "llm": judge_llm,
        "embeddings": judge_embeddings,
    }

    if RunConfig is not None:
        eval_kwargs["run_config"] = RunConfig(
            max_workers=2,
            timeout=300,   # 延长超时阈值，兼容多用例大并发
            max_retries=5,
            max_wait=60
        )

    results = evaluate(**eval_kwargs)

    print("\n================== 🎯 RAGAS 总体评测报告 ==================")
    print(results)
    print("============================================================")

    df = results.to_pandas()
    save_evaluation_results(df, current_dir)


if __name__ == "__main__":
    main()