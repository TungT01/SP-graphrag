"""
evaluation/summary_quality_evaluator.py
-----------------------------------------
摘要质量直接度量模块（方向 C）。

目的：直接比较标准 Leiden（A）和约束 Leiden（B3）社区摘要的质量，
验证"物理纯净社区 → 更连贯摘要"这一传导链条的第一步。

三个维度：
  1. 主题聚焦度（LLM打分 1-5）：摘要是否围绕单一清晰的主题
  2. 实体覆盖率（自动计算）：摘要中提到的实体占社区实体总数的比例
  3. 跨文档混杂率（自动计算）：社区内 doc_id 数量（越少越纯净）

对照设计：
  - 从两种社区中各抽取 N 对"规模匹配"的社区（实体数相近）
  - 完全相同的评分标准，LLM 不知道社区来自哪种系统（盲评）
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CommunitySample:
    """单个社区的样本信息。"""
    community_id: int
    level: int
    system: str            # "standard" 或 "constrained"
    entity_ids: List[str]
    doc_ids: List[str]
    structural_entropy: float
    summary: str
    # 自动计算
    num_entities: int = 0
    num_docs: int = 0
    entity_coverage: float = 0.0   # 摘要中提到的实体比例
    focus_score: float = 0.0       # LLM 主题聚焦度打分（1-5）
    focus_reason: str = ""         # LLM 给出的理由


@dataclass
class SummaryQualityMetrics:
    """摘要质量对比汇总指标。"""
    system: str
    n_samples: int
    avg_focus_score: float = 0.0
    avg_entity_coverage: float = 0.0
    avg_num_docs: float = 0.0
    avg_structural_entropy: float = 0.0
    pct_single_doc: float = 0.0    # 来自单一文档的社区比例

    def summary_str(self) -> str:
        return (
            f"[{self.system}] n={self.n_samples}\n"
            f"  主题聚焦度（1-5）: {self.avg_focus_score:.2f}\n"
            f"  实体覆盖率       : {self.avg_entity_coverage:.2%}\n"
            f"  平均来源文档数   : {self.avg_num_docs:.2f}\n"
            f"  单文档社区比例   : {self.pct_single_doc:.1%}\n"
            f"  平均结构熵       : {self.avg_structural_entropy:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "n_samples": self.n_samples,
            "avg_focus_score": round(self.avg_focus_score, 3),
            "avg_entity_coverage": round(self.avg_entity_coverage, 4),
            "avg_num_docs": round(self.avg_num_docs, 3),
            "avg_structural_entropy": round(self.avg_structural_entropy, 4),
            "pct_single_doc": round(self.pct_single_doc, 4),
        }


# ---------------------------------------------------------------------------
# 采样工具
# ---------------------------------------------------------------------------

def sample_matched_communities(
    comm_df_a: pd.DataFrame,
    comm_df_b: pd.DataFrame,
    summary_cache_a: Dict[str, str],
    summary_cache_b: Dict[str, str],
    n_samples: int = 100,
    min_level: int = 2,
    min_entities: int = 3,
    max_entities: int = 15,
    seed: int = 42,
) -> Tuple[List[CommunitySample], List[CommunitySample]]:
    """
    从两个系统中各采样 N 个规模匹配的社区。

    匹配规则：
    - level >= min_level（低层社区太小，摘要无意义）
    - min_entities <= 实体数 <= max_entities（过滤极端大小）
    - 有非空摘要

    Parameters
    ----------
    comm_df_a : 标准 Leiden 社区表
    comm_df_b : 约束 Leiden 社区表
    summary_cache_a : 标准 Leiden 摘要缓存 {cache_key: summary}
    summary_cache_b : 约束 Leiden 摘要缓存
    n_samples : 每个系统采样数量
    """
    from graphrag_improved.summarization.summarizer import _community_cache_key

    rng = random.Random(seed)

    def _get_valid(comm_df, cache, system_name):
        valid = []
        for _, row in comm_df.iterrows():
            if int(row["level"]) < min_level:
                continue
            entity_ids = row.get("entity_ids", []) or []
            if not (min_entities <= len(entity_ids) <= max_entities):
                continue
            # 查摘要
            key = _community_cache_key(row)
            summary = cache.get(key, "")
            if not summary or len(summary) < 20:
                continue
            doc_ids = row.get("doc_ids", []) or []
            valid.append(CommunitySample(
                community_id=int(row["community_id"]),
                level=int(row["level"]),
                system=system_name,
                entity_ids=entity_ids,
                doc_ids=doc_ids,
                structural_entropy=float(row.get("structural_entropy", 0.0)),
                summary=summary,
                num_entities=len(entity_ids),
                num_docs=len(set(doc_ids)),
            ))
        return valid

    valid_a = _get_valid(comm_df_a, summary_cache_a, "standard")
    valid_b = _get_valid(comm_df_b, summary_cache_b, "constrained")

    # 各采样 n_samples 个
    samples_a = rng.sample(valid_a, min(n_samples, len(valid_a)))
    samples_b = rng.sample(valid_b, min(n_samples, len(valid_b)))

    return samples_a, samples_b


# ---------------------------------------------------------------------------
# 自动指标计算
# ---------------------------------------------------------------------------

def compute_entity_coverage(summary: str, entity_ids: List[str]) -> float:
    """
    计算摘要中提到的实体占比。

    将 node_id 转换为可读实体名（取最后一段），检查是否出现在摘要中。
    """
    if not entity_ids or not summary:
        return 0.0
    summary_lower = summary.lower()
    mentioned = 0
    for eid in entity_ids:
        # node_id 格式：{sent_id}-{entity_name}，取最后一段
        name = str(eid).rsplit("-", 1)[-1].replace("_", " ").lower()
        if len(name) >= 3 and name in summary_lower:
            mentioned += 1
    return mentioned / len(entity_ids)


# ---------------------------------------------------------------------------
# LLM 主题聚焦度打分
# ---------------------------------------------------------------------------

_FOCUS_SYSTEM = (
    "You are evaluating the thematic coherence of a knowledge graph community summary. "
    "Rate the summary on a scale of 1-5:\n"
    "5 = Perfectly focused on one clear, specific topic\n"
    "4 = Mostly focused, minor tangents\n"
    "3 = Some focus but multiple loosely related topics\n"
    "2 = Scattered across multiple unrelated topics\n"
    "1 = No coherent theme, random collection of facts\n"
    "Respond with JSON only: {\"score\": <1-5>, \"reason\": \"<one sentence>\"}"
)

_FOCUS_USER = "Summary to evaluate:\n\n{summary}"


def score_focus(
    client,
    provider: str,
    model: str,
    summary: str,
    max_tokens: int = 80,
) -> Tuple[float, str]:
    """调用 LLM 对摘要主题聚焦度打分，返回 (score, reason)。"""
    try:
        prompt = _FOCUS_USER.format(summary=summary[:600])
        if provider == "anthropic":
            response = client.messages.create(
                model=model, max_tokens=max_tokens,
                system=_FOCUS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _FOCUS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()

        # 解析 JSON
        data = json.loads(text)
        score = float(data.get("score", 3))
        reason = str(data.get("reason", ""))
        return max(1.0, min(5.0, score)), reason

    except Exception as e:
        # 降级：尝试从文本中提取数字
        numbers = re.findall(r'\b([1-5])\b', text if 'text' in dir() else "")
        if numbers:
            return float(numbers[0]), f"parse_fallback: {str(e)[:50]}"
        return 3.0, f"error: {str(e)[:80]}"


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate_summary_quality(
    samples_a: List[CommunitySample],
    samples_b: List[CommunitySample],
    llm_config,
    concurrency: int = 10,
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> Tuple[SummaryQualityMetrics, SummaryQualityMetrics]:
    """
    对两组社区样本进行摘要质量评估。

    Parameters
    ----------
    samples_a : 标准 Leiden 样本
    samples_b : 约束 Leiden 样本
    llm_config : LlmConfig（provider, model, api_key）
    concurrency : 并发 LLM 调用数
    output_path : 结果保存路径（JSON）

    Returns
    -------
    Tuple[SummaryQualityMetrics, SummaryQualityMetrics]
        (标准Leiden指标, 约束Leiden指标)
    """
    from graphrag_improved.summarization.summarizer import _make_llm_client

    all_samples = samples_a + samples_b
    if verbose:
        print(f"  [摘要质量] 评估 {len(all_samples)} 个社区")
        print(f"    标准Leiden: {len(samples_a)} 个，约束Leiden: {len(samples_b)} 个")
        print(f"    模型: {llm_config.provider}/{llm_config.model}，并发: {concurrency}")

    # 先计算自动指标（无需 API）
    for s in all_samples:
        s.entity_coverage = compute_entity_coverage(s.summary, s.entity_ids)

    # LLM 主题聚焦度打分（并发）
    client = _make_llm_client(llm_config)
    provider = llm_config.provider.lower()
    t0 = time.time()

    def _score_one(sample: CommunitySample) -> CommunitySample:
        score, reason = score_focus(client, provider, llm_config.model, sample.summary)
        sample.focus_score = score
        sample.focus_reason = reason
        return sample

    scored = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_score_one, s): s for s in all_samples}
        for future in as_completed(futures):
            try:
                scored.append(future.result())
            except Exception as e:
                s = futures[future]
                s.focus_score = 3.0
                s.focus_reason = f"error: {e}"
                scored.append(s)
            if verbose and len(scored) % 20 == 0:
                elapsed = time.time() - t0
                print(f"    进度: {len(scored)}/{len(all_samples)}  ({elapsed:.0f}s)")

    if verbose:
        print(f"  [摘要质量] 评分完成，耗时 {time.time()-t0:.0f}s")

    # 分组汇总
    def _aggregate(samples: List[CommunitySample], system: str) -> SummaryQualityMetrics:
        n = len(samples)
        if n == 0:
            return SummaryQualityMetrics(system=system, n_samples=0)
        return SummaryQualityMetrics(
            system=system,
            n_samples=n,
            avg_focus_score=sum(s.focus_score for s in samples) / n,
            avg_entity_coverage=sum(s.entity_coverage for s in samples) / n,
            avg_num_docs=sum(s.num_docs for s in samples) / n,
            avg_structural_entropy=sum(s.structural_entropy for s in samples) / n,
            pct_single_doc=sum(1 for s in samples if s.num_docs <= 1) / n,
        )

    scored_a = [s for s in scored if s.system == "standard"]
    scored_b = [s for s in scored if s.system == "constrained"]
    metrics_a = _aggregate(scored_a, "standard (λ=0)")
    metrics_b = _aggregate(scored_b, "constrained (λ=0.003)")

    if verbose:
        print()
        print(metrics_a.summary_str())
        print()
        print(metrics_b.summary_str())
        print()
        diff_focus = metrics_b.avg_focus_score - metrics_a.avg_focus_score
        diff_cov = metrics_b.avg_entity_coverage - metrics_a.avg_entity_coverage
        print(f"  差值（约束 - 标准）:")
        print(f"    主题聚焦度: {diff_focus:+.2f}")
        print(f"    实体覆盖率: {diff_cov:+.2%}")
        print(f"    单文档比例: {metrics_b.pct_single_doc - metrics_a.pct_single_doc:+.1%}")

    # 保存详细结果
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "metrics_standard": metrics_a.to_dict(),
            "metrics_constrained": metrics_b.to_dict(),
            "samples_standard": [
                {"community_id": s.community_id, "level": s.level,
                 "num_entities": s.num_entities, "num_docs": s.num_docs,
                 "structural_entropy": s.structural_entropy,
                 "focus_score": s.focus_score, "focus_reason": s.focus_reason,
                 "entity_coverage": s.entity_coverage,
                 "summary": s.summary[:200]}
                for s in scored_a
            ],
            "samples_constrained": [
                {"community_id": s.community_id, "level": s.level,
                 "num_entities": s.num_entities, "num_docs": s.num_docs,
                 "structural_entropy": s.structural_entropy,
                 "focus_score": s.focus_score, "focus_reason": s.focus_reason,
                 "entity_coverage": s.entity_coverage,
                 "summary": s.summary[:200]}
                for s in scored_b
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  结果已保存: {output_path}")

    return metrics_a, metrics_b
