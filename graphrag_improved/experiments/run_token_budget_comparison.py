"""
experiments/run_token_budget_comparison.py
-------------------------------------------
公平 token 预算对比实验：
在相同上下文长度限制下，比较 D+V（直接段落检索）和 B3+VS（社区摘要检索）的 QA 精度。

B3+VS 自然平均上下文 = 1744 chars（来自 v11b 实验）
D+V 通常填满 3000 chars cap，将其限制到 1744 chars 进行公平比较。

用法：
  python3 -m experiments.run_token_budget_comparison --api-key YOUR_KEY
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection
from graphrag_improved.retrieval.retriever import URetriever
from graphrag_improved.evaluation.qa_evaluator import evaluate_qa_end_to_end
from graphrag_improved.evaluation.evaluator import QAPair
from graphrag_improved.experiments.data_loader import load_multihop_dataset, corpus_to_text_units
from graphrag_improved.pipeline_config import LlmConfig


def main():
    parser = argparse.ArgumentParser(description="公平 token 预算对比：D+V vs B3+VS")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--n-qa", type=int, default=500)
    parser.add_argument("--context-budget", type=int, default=1744,
                        help="上下文长度限制（默认 1744，与 B3+VS 自然上下文相同）")
    parser.add_argument("--data-dir", default="../data/multihop_rag")
    parser.add_argument("--cache-dir", default="experiments/results_v11_lambda/cache")
    parser.add_argument("--output-dir", default="experiments/results_v11_lambda")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    llm_config = LlmConfig(
        enabled=True, provider="openai_compatible",
        model="deepseek-v4-flash", api_key=api_key,
        base_url="https://api.deepseek.com",
        max_tokens=128, batch_size=10,
    )

    print(f"\n{'='*60}")
    print(f"  公平 Token 预算对比实验")
    print(f"  上下文预算: {args.context_budget} chars")
    print(f"{'='*60}")

    # 加载数据
    dataset = load_multihop_dataset(args.data_dir)
    dataset = dataset.subset(args.n_qa, seed=42)
    retrieval_units = corpus_to_text_units(dataset.corpus)
    eval_qa = [QAPair(question=qa.query, answer=qa.answer, context_ids=qa.supporting_doc_ids)
               for qa in dataset.qa_pairs if qa.supporting_doc_ids]
    print(f"有效QA: {len(eval_qa)} 条")

    entities_df = pd.read_parquet(f"{args.cache_dir}/entities_n{args.n_qa}.parquet")
    rels_df     = pd.read_parquet(f"{args.cache_dir}/relationships_n{args.n_qa}.parquet")

    # D+V：直接段落向量检索，限制 token 预算
    print(f"\n构建 D+V 检索器...")
    cfg = AnnealingConfig(lambda_init=0.0, lambda_min=0.0,
        schedule=AnnealingSchedule("exponential"), decay_rate=0.5, max_level=10)
    comm_df = run_constrained_community_detection(
        entities=entities_df, relationships=rels_df,
        annealing_config=cfg, anchor_granularity="sent", use_lcc=False, seed=42)
    retriever = URetriever(communities_df=comm_df, text_units=retrieval_units,
        entities_df=entities_df, retrieval_mode="vector_bottomup_only")

    print(f"运行 QA 评估（max_context={args.context_budget} chars）...")
    t0 = time.time()
    metrics = evaluate_qa_end_to_end(
        qa_pairs=eval_qa, retriever=retriever, llm_config=llm_config,
        max_context_chars=args.context_budget, concurrency=10, verbose=True,
    )
    elapsed = time.time() - t0

    # 打印对比结果
    print(f"\n{'='*60}")
    print(f"  公平 Token 预算对比（{args.context_budget} chars 限制）")
    print(f"{'='*60}")
    print(f"  {'方法':<30} {'EM':>6}  {'Token F1':>8}  {'avg_context':>11}")
    print(f"  {'-'*58}")
    print(f"  {'B3+VS 社区摘要 (v11b, n=881)':<30} {'0.1237':>6}  {'0.1844':>8}  {'1,744 chars':>11}")
    print(f"  {'D+V  段落检索 @ budget':<30} {metrics.exact_match:>6.4f}  {metrics.token_f1:>8.4f}  {metrics.avg_context_chars:>10.0f}")
    print(f"{'='*60}")

    diff_em = metrics.exact_match - 0.1237
    print(f"\n  精度差（D+V - B3+VS）: EM {diff_em:+.4f} ({diff_em/0.1237*100:+.1f}%)")
    print(f"  token 预算: 完全相同 ({args.context_budget} chars)")
    print(f"  耗时: {elapsed:.0f}s")

    # 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{args.output_dir}/token_budget_comparison.json"
    result = {
        "context_budget": args.context_budget,
        "b3vs_reference": {"em": 0.1237, "f1": 0.1844, "context": 1744, "source": "v11b n=881"},
        "dv_at_budget": metrics.to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
