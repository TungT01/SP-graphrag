"""
experiments/run_summary_quality_eval.py
-----------------------------------------
方向 C：摘要质量直接度量实验。

对比标准 Leiden（A）和约束 Leiden（B3）的社区摘要质量：
  1. 主题聚焦度（LLM 打分 1-5）
  2. 实体覆盖率（自动计算）
  3. 跨文档混杂率（自动计算，doc_ids 数量）

用法：
  python3 -m experiments.run_summary_quality_eval \\
      --api-key YOUR_KIMI_KEY \\
      --provider kimi \\
      --n-samples 100 \\
      --output-dir experiments/results_summary_quality
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.edge_scheduler import EdgeSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection
from graphrag_improved.evaluation.summary_quality_evaluator import (
    evaluate_summary_quality, sample_matched_communities
)
from graphrag_improved.experiments.data_loader import load_multihop_dataset
from graphrag_improved.experiments.run_multihop_eval import build_pipeline_text_units
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.pipeline_config import ExtractionConfig, LlmConfig
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="摘要质量直接度量实验（方向 C）")
    parser.add_argument("--data-dir", default="../data/multihop_rag")
    parser.add_argument("--cache-dir", default="experiments/results_v11_lambda/cache",
                        help="实体抽取缓存目录（复用已有的）")
    parser.add_argument("--summary-cache-a", default="summary_cache/summaries_leiden_standard.json")
    parser.add_argument("--summary-cache-b", default="summary_cache/summaries_leiden_constrained_003.json")
    parser.add_argument("--n-qa", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="每个系统采样的社区数量（默认 200）")
    parser.add_argument("--min-entities", type=int, default=3,
                        help="社区最小实体数（默认 3）")
    parser.add_argument("--max-entities", type=int, default=100,
                        help="社区最大实体数（默认 100，覆盖中大型社区）")
    parser.add_argument("--output-dir", default="experiments/results_summary_quality")
    parser.add_argument("--provider", default="kimi",
                        choices=["kimi", "anthropic", "openai", "openai_compatible"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    # 构建 LlmConfig
    provider = args.provider.lower()
    base_url = ""
    model = args.llm_model
    if provider == "kimi":
        provider = "openai_compatible"
        base_url = "https://api.moonshot.cn/v1"
        model = model or "moonshot-v1-8k"
        api_key = args.api_key or os.environ.get("MOONSHOT_API_KEY", "")
    elif provider == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    else:
        model = model or "gpt-4o-mini"
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    llm_config = LlmConfig(
        enabled=True, provider=provider, model=model,
        api_key=api_key, base_url=base_url,
        max_tokens=80, batch_size=args.concurrency,
    )

    print(f"\n{'='*60}")
    print(f"  摘要质量直接度量实验（方向 C）")
    print(f"{'='*60}")
    print(f"  模型: {provider}/{model}")
    print(f"  采样数: 每组 {args.n_samples} 个社区")

    # 加载实体抽取缓存
    cache_dir = Path(args.cache_dir)
    entities_df = pd.read_parquet(cache_dir / f"entities_n{args.n_qa}.parquet")
    rels_df = pd.read_parquet(cache_dir / f"relationships_n{args.n_qa}.parquet")
    print(f"\n[1/4] 加载实体缓存: {len(entities_df)} 实体，{len(rels_df)} 关系")

    # 构建两种社区
    print("\n[2/4] 构建社区...")
    print("  标准 Leiden (λ=0)...")
    cfg_a = AnnealingConfig(lambda_init=0.0, lambda_min=0.0,
        schedule=AnnealingSchedule('exponential'), decay_rate=0.5, max_level=10)
    comm_df_a = run_constrained_community_detection(
        entities=entities_df, relationships=rels_df,
        annealing_config=cfg_a, anchor_granularity='sent',
        use_lcc=False, seed=42
    )
    print(f"    社区 {len(comm_df_a)} 个，层次 {comm_df_a['level'].nunique()} 层")

    print("  约束 Leiden (λ=0.003)...")
    cfg_b = AnnealingConfig(lambda_init=0.003, lambda_min=0.0,
        schedule=AnnealingSchedule('exponential'), decay_rate=0.5, max_level=10)
    es = EdgeSchedule.build(entities_df, include_cross_doc=False)
    comm_df_b = run_constrained_community_detection(
        entities=entities_df, relationships=rels_df,
        annealing_config=cfg_b, anchor_granularity='para',
        edge_schedule=es, use_lcc=False, seed=42
    )
    print(f"    社区 {len(comm_df_b)} 个，层次 {comm_df_b['level'].nunique()} 层")

    # 加载摘要缓存
    print("\n[3/4] 加载摘要缓存...")
    with open(args.summary_cache_a) as f:
        cache_a = json.load(f)
    with open(args.summary_cache_b) as f:
        cache_b = json.load(f)
    print(f"  标准Leiden摘要: {len(cache_a)} 条")
    print(f"  约束Leiden摘要: {sum(1 for v in cache_b.values() if v)} 条非空")

    # 采样
    samples_a, samples_b = sample_matched_communities(
        comm_df_a, comm_df_b, cache_a, cache_b,
        n_samples=args.n_samples, min_level=2,
        min_entities=args.min_entities,
        max_entities=args.max_entities,
    )
    print(f"  采样完成: 标准={len(samples_a)}, 约束={len(samples_b)}")

    # 评估
    print("\n[4/4] LLM 主题聚焦度评分...")
    out_path = str(Path(args.output_dir) / "summary_quality_results.json")
    metrics_a, metrics_b = evaluate_summary_quality(
        samples_a, samples_b, llm_config,
        concurrency=args.concurrency,
        verbose=True,
        output_path=out_path,
    )

    # 最终对比表
    print(f"\n{'='*60}")
    print("  摘要质量对比结果")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'标准Leiden(A)':<18} {'约束Leiden(B3)':<18} {'差值'}")
    print(f"  {'-'*68}")
    rows = [
        ("主题聚焦度(1-5)", metrics_a.avg_focus_score, metrics_b.avg_focus_score),
        ("实体覆盖率",      metrics_a.avg_entity_coverage, metrics_b.avg_entity_coverage),
        ("平均来源文档数",  metrics_a.avg_num_docs, metrics_b.avg_num_docs),
        ("单文档社区比例",  metrics_a.pct_single_doc, metrics_b.pct_single_doc),
        ("平均结构熵",     metrics_a.avg_structural_entropy, metrics_b.avg_structural_entropy),
    ]
    for name, va, vb in rows:
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<20} {va:<18.4f} {vb:<18.4f} {sign}{diff:.4f}")
    print(f"{'='*60}")
    print(f"\n  详细结果: {out_path}")


if __name__ == "__main__":
    main()
