"""
experiments/stats_community_distribution.py
--------------------------------------------
统计社区层次分布，用于估算摘要生成的 LLM 调用次数和费用。

运行方式：
  cd /Users/ttung/Desktop/个人学习
  python3 -m graphrag_improved.experiments.stats_community_distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CACHE_DIR   = Path(__file__).parent / "results" / "cache"
# 使用路径B（噪声过滤后）的缓存，对应 [3] Ours+A+B
CACHE_TAG_B = "n200_b"
CACHE_TAG   = "n200"

# 两组都统计：Baseline(λ=0) 和 Ours(λ=1000)
CONFIGS = [
    dict(name="[0] Baseline (λ=0)",  lambda_init=0.0,    intra_doc_merging=False, cache="orig"),
    dict(name="[1] Ours (λ=1000)",   lambda_init=1000.0, intra_doc_merging=False, cache="orig"),
    dict(name="[3] Ours+A+B",        lambda_init=1000.0, intra_doc_merging=True,  cache="b"),
]

LLM_THRESHOLD = 5   # 节点数 >= 此值的社区才调 LLM 生成摘要

# Kimi moonshot-v1-8k 价格（元/M tokens）
PRICE_INPUT  = 2.0
PRICE_OUTPUT = 10.0
# 每次 LLM 调用的估算 token 数
EST_INPUT_TOKENS  = 400   # system_prompt + 实体列表
EST_OUTPUT_TOKENS = 150   # 摘要输出


def load_cache(tag: str):
    ep = CACHE_DIR / f"entities_{tag}.parquet"
    rp = CACHE_DIR / f"relationships_{tag}.parquet"
    if not ep.exists():
        print(f"  [错误] 缓存不存在：{ep}")
        print(f"  请先运行一次完整实验以生成缓存：")
        print(f"    python3 -m graphrag_improved.experiments.run_multihop_eval --n-qa 200")
        return None, None
    entities_df      = pd.read_parquet(ep)
    relationships_df = pd.read_parquet(rp)
    print(f"  加载缓存 [{tag}]：实体 {len(entities_df)}，关系 {len(relationships_df)}")
    return entities_df, relationships_df


def stats_one(cfg: dict, entities_df, relationships_df):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"{'='*60}")

    from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
    annealing_config = AnnealingConfig(
        lambda_init=cfg["lambda_init"],
        lambda_min=0.0,
        schedule=AnnealingSchedule("exponential"),
        decay_rate=0.5,
        max_level=10,
    )

    print("  运行社区检测...")
    communities_df = run_constrained_community_detection(
        entities=entities_df,
        relationships=relationships_df,
        annealing_config=annealing_config,
        max_cluster_size=10,
        max_iterations=10,
        seed=42,
        use_lcc=False,
        intra_doc_merging=cfg["intra_doc_merging"],
        intra_doc_edge_weight=0.5,
    )

    if communities_df.empty:
        print("  [警告] 社区为空")
        return

    total = len(communities_df)
    levels = sorted(communities_df["level"].unique())
    print(f"\n  总社区数：{total}，层次：{levels}")

    # 计算每个社区的节点数
    communities_df = communities_df.copy()
    communities_df["node_count"] = communities_df["entity_ids"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # ── 按层级统计 ──────────────────────────────────────────────
    print(f"\n  {'层级':<8} {'社区数':>8} {'平均节点':>10} {'最大节点':>10} "
          f"{'≥{t}节点(LLM)':>14} {'<{t}节点(规则)':>14}".format(t=LLM_THRESHOLD))
    print(f"  {'-'*70}")

    total_llm   = 0
    total_rule  = 0

    for level in levels:
        sub = communities_df[communities_df["level"] == level]
        n_total  = len(sub)
        avg_size = sub["node_count"].mean()
        max_size = sub["node_count"].max()
        n_llm    = (sub["node_count"] >= LLM_THRESHOLD).sum()
        n_rule   = n_total - n_llm
        total_llm  += n_llm
        total_rule += n_rule
        print(f"  Level {level:<3} {n_total:>8,} {avg_size:>10.2f} {max_size:>10} "
              f"{n_llm:>14,} {n_rule:>14,}")

    print(f"  {'-'*70}")
    print(f"  {'合计':<8} {total:>8,} {'':>10} {'':>10} "
          f"{total_llm:>14,} {total_rule:>14,}")

    # ── 节点数分布直方图 ──────────────────────────────────────
    print(f"\n  节点数分布（所有层级合并）：")
    bins = [1, 2, 3, 5, 10, 20, 50, float("inf")]
    labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "≥50"]
    for i, label in enumerate(labels):
        lo = bins[i]
        hi = bins[i + 1]
        if hi == float("inf"):
            count = (communities_df["node_count"] >= lo).sum()
        else:
            count = ((communities_df["node_count"] >= lo) &
                     (communities_df["node_count"] < hi)).sum()
        bar = "█" * min(40, int(count / max(total, 1) * 200))
        print(f"    {label:>5} 节点: {count:>7,}  {bar}")

    # ── 费用估算 ──────────────────────────────────────────────
    print(f"\n  费用估算（LLM 阈值：节点数 ≥ {LLM_THRESHOLD}）：")
    print(f"    需要 LLM 的社区数：{total_llm:,}")
    print(f"    规则生成的社区数：{total_rule:,}（免费）")

    input_cost  = total_llm * EST_INPUT_TOKENS  / 1_000_000 * PRICE_INPUT
    output_cost = total_llm * EST_OUTPUT_TOKENS / 1_000_000 * PRICE_OUTPUT
    total_cost  = input_cost + output_cost

    print(f"    估算 input tokens：{total_llm * EST_INPUT_TOKENS:,}  → ¥{input_cost:.4f}")
    print(f"    估算 output tokens：{total_llm * EST_OUTPUT_TOKENS:,}  → ¥{output_cost:.4f}")
    print(f"    合计费用（moonshot-v1-8k）：¥{total_cost:.4f}")
    print(f"    Batch API 5折后：¥{total_cost * 0.5:.4f}")

    # ── 不同阈值的费用对比 ────────────────────────────────────
    print(f"\n  不同 LLM 阈值的费用对比：")
    print(f"    {'阈值':>6} {'LLM社区数':>10} {'费用(原价)':>12} {'费用(Batch)':>12}")
    print(f"    {'-'*44}")
    for thresh in [3, 5, 10, 20]:
        n = (communities_df["node_count"] >= thresh).sum()
        c = n * (EST_INPUT_TOKENS * PRICE_INPUT + EST_OUTPUT_TOKENS * PRICE_OUTPUT) / 1_000_000
        print(f"    {f'≥{thresh}':>6} {n:>10,} {f'¥{c:.4f}':>12} {f'¥{c*0.5:.4f}':>12}")

    return communities_df


def main():
    print("=" * 60)
    print("  社区层次分布统计 & 摘要生成费用估算")
    print("=" * 60)

    # 加载两份缓存
    print("\n[1/2] 加载实体抽取缓存...")
    ents_orig, rels_orig = load_cache(CACHE_TAG)
    ents_b,    rels_b    = load_cache(CACHE_TAG_B)

    if ents_orig is None:
        return

    # 统计三组
    for cfg in CONFIGS:
        tag = cfg["cache"]
        ents = ents_b if tag == "b" else ents_orig
        rels = rels_b if tag == "b" else rels_orig
        if ents is None:
            print(f"\n  跳过 {cfg['name']}（缓存不存在）")
            continue
        stats_one(cfg, ents, rels)

    print(f"\n{'='*60}")
    print("  统计完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
