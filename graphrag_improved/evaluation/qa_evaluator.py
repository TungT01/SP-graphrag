"""
evaluation/qa_evaluator.py
---------------------------
端到端 QA 评估模块。

流程：
  1. 用 retriever 检索相关上下文
  2. 将上下文 + 问题发送给 LLM 生成答案
  3. 用 Exact Match / Token F1 / ROUGE-L 评估生成答案 vs 标准答案

这是验证核心 claim 的关键模块：
  - [A+VS]  标准 Leiden  + 向量检索 + LLM 生成 → 微观精度基线
  - [B3+VS] 约束 Leiden + 向量检索 + LLM 生成 → 微观精度实验组
  A+VS vs B3+VS 的差异 = 物理结构约束的纯贡献
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .evaluator import compute_exact_match, compute_token_f1, compute_rouge_l


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """单条 QA 评估结果。"""
    question: str
    gold_answer: str
    predicted_answer: str
    context_used: str
    exact_match: float
    token_f1: float
    rouge_l: float
    latency_ms: float = 0.0


@dataclass
class QAMetrics:
    """QA 评估汇总指标。"""
    exact_match: float = 0.0
    token_f1: float = 0.0
    rouge_l: float = 0.0
    num_queries: int = 0
    avg_context_chars: float = 0.0
    avg_latency_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"端到端 QA 评估（{self.num_queries} 条）\n"
            f"  Exact Match  : {self.exact_match:.4f}\n"
            f"  Token F1     : {self.token_f1:.4f}\n"
            f"  ROUGE-L      : {self.rouge_l:.4f}\n"
            f"  平均上下文长度: {self.avg_context_chars:.0f} chars\n"
            f"  平均延迟      : {self.avg_latency_ms:.0f} ms"
        )

    def to_dict(self) -> dict:
        return {
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "rouge_l": round(self.rouge_l, 4),
            "num_queries": self.num_queries,
            "avg_context_chars": round(self.avg_context_chars, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# LLM 生成答案
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question based ONLY on the provided context. "
    "Give a concise, direct answer. "
    "If the answer is not in the context, say 'Not found in context'."
)

_QA_USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (be concise and direct):"
)


def _generate_answer(
    client,
    provider: str,
    model: str,
    question: str,
    context: str,
    max_tokens: int = 128,
) -> Tuple[str, float]:
    """调用 LLM 生成答案，返回 (answer, latency_ms)。"""
    t0 = time.time()
    prompt = _QA_USER_TEMPLATE.format(
        context=context[:3000],  # 截断避免超出上下文窗口
        question=question,
    )
    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_QA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip()
        else:  # openai / openai_compatible
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content.strip()
        latency_ms = (time.time() - t0) * 1000
        return answer, latency_ms
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        print(f"    [QA] LLM 调用失败: {e}")
        return "", latency_ms


# ---------------------------------------------------------------------------
# QA 缓存工具
# ---------------------------------------------------------------------------

def _make_qa_cache_key(question: str, context: str, model: str) -> str:
    """缓存 key：基于 question + context + model，任一变化则 miss。"""
    raw = f"{question}|||{context[:3000]}|||{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_qa_cache(cache_path: Optional[str]) -> Dict[str, dict]:
    if not cache_path:
        return {}
    p = Path(cache_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  [QA缓存] 文件损坏，已忽略并重建：{p.name}")
            p.unlink()
    return {}


def _save_qa_cache(cache: Dict[str, dict], cache_path: Optional[str]) -> None:
    if not cache_path:
        return
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(p)  # 原子替换，进程被杀时旧文件保持完整


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate_qa_end_to_end(
    qa_pairs: List,           # List[EvalQAPair]
    retriever,                # URetriever
    llm_config,               # LlmConfig
    max_context_chars: int = 3000,
    concurrency: int = 5,
    verbose: bool = True,
    cache_path: Optional[str] = None,
) -> QAMetrics:
    """
    端到端 QA 评估：检索 → 生成 → 评分。

    Parameters
    ----------
    qa_pairs : List[EvalQAPair]
        问答对，每个含 question, answer, context_ids
    retriever : URetriever
        已构建的检索器
    llm_config : LlmConfig
        LLM 配置（provider, model, api_key, base_url）
    max_context_chars : int
        传给 LLM 的最大上下文字符数
    concurrency : int
        并发生成线程数
    verbose : bool

    Returns
    -------
    QAMetrics
    """
    from graphrag_improved.summarization.summarizer import _make_llm_client

    if not qa_pairs:
        return QAMetrics()

    valid_qa = [qa for qa in qa_pairs if qa.answer and qa.answer.strip()]
    if not valid_qa:
        return QAMetrics()

    if verbose:
        print(f"  [QA评估] {len(valid_qa)} 条问题，模型={llm_config.provider}/{llm_config.model}")
        print(f"  [QA评估] 并发={concurrency}，上下文上限={max_context_chars} chars")

    # 加载缓存
    qa_cache = _load_qa_cache(cache_path)
    cache_lock = threading.Lock()
    cache_hits = 0
    if verbose and cache_path:
        print(f"  [QA缓存] 加载 {len(qa_cache)} 条缓存（{Path(cache_path).name}）")

    client = _make_llm_client(llm_config)
    provider = llm_config.provider.lower()

    def _process_one(qa) -> QAResult:
        nonlocal cache_hits
        # 检索上下文
        result = retriever.retrieve(qa.question)
        context = result.merged_context or ""
        if not context:
            context = "\n".join(
                hit.text for hit in result.bottom_up_hits[:5]
            )
        context = context[:max_context_chars]

        # 查缓存
        key = _make_qa_cache_key(qa.question, context, llm_config.model)
        cached = qa_cache.get(key)
        if cached:
            with cache_lock:
                cache_hits += 1
            predicted = cached["predicted_answer"]
            latency = cached.get("latency_ms", 0.0)
        else:
            # 调用 LLM
            predicted, latency = _generate_answer(
                client, provider, llm_config.model,
                qa.question, context, max_tokens=128
            )
            with cache_lock:
                qa_cache[key] = {"predicted_answer": predicted, "latency_ms": latency}

        # 评分
        em = compute_exact_match(predicted, qa.answer)
        f1 = compute_token_f1(predicted, qa.answer)
        rl = compute_rouge_l(predicted, qa.answer)

        return QAResult(
            question=qa.question,
            gold_answer=qa.answer,
            predicted_answer=predicted,
            context_used=context,
            exact_match=em,
            token_f1=f1,
            rouge_l=rl,
            latency_ms=latency,
        )

    results: List[QAResult] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_process_one, qa): qa for qa in valid_qa}
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                print(f"    [QA评估] 任务失败: {e}")

            if verbose and len(results) % 50 == 0:
                elapsed = time.time() - t0
                rate = len(results) / elapsed if elapsed > 0 else 0
                print(f"  [QA评估] {len(results)}/{len(valid_qa)}  ({rate:.1f} 条/s)")

            # 每 100 条保存一次缓存（先在锁内拷贝快照，避免并发写入冲突）
            if len(results) % 100 == 0:
                with cache_lock:
                    snapshot = dict(qa_cache)
                _save_qa_cache(snapshot, cache_path)

    # 最终保存缓存（executor 已退出，所有线程已完成，仍用快照保险）
    with cache_lock:
        snapshot = dict(qa_cache)
    _save_qa_cache(snapshot, cache_path)
    if verbose and cache_path:
        api_calls = len(results) - cache_hits
        print(f"  [QA缓存] 命中 {cache_hits} 条，新增 API 调用 {api_calls} 条，已保存 → {cache_path}")

    if not results:
        return QAMetrics()

    n = len(results)
    metrics = QAMetrics(
        exact_match=sum(r.exact_match for r in results) / n,
        token_f1=sum(r.token_f1 for r in results) / n,
        rouge_l=sum(r.rouge_l for r in results) / n,
        num_queries=n,
        avg_context_chars=sum(len(r.context_used) for r in results) / n,
        avg_latency_ms=sum(r.latency_ms for r in results) / n,
    )

    if verbose:
        print(f"\n{metrics.summary()}")

    return metrics
