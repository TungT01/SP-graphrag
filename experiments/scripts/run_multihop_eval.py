"""
experiments/run_multihop_eval.py
---------------------------------
MultiHop-RAG 完整对照实验脚本（v5 渐进合并架构）。

六组消融实验设计（v5）：
  [0] Baseline          : λ=0, sent 锚点（纯模块度 Leiden，无任何改进）
  [1] 仅改锚点          : λ=1000, para 锚点（结构熵有意义，但图仍是断裂森林）
  [2] 仅分层加边        : λ=1000, sent 锚点 + EdgeSchedule（图连通，但熵无区分度）
  [3] 锚点+分层加边      : λ=1000, para 锚点 + EdgeSchedule（完整方案）
  [4] 完整方案+Path A    : λ=1000, para 锚点 + EdgeSchedule + 文档内消解边
  [5] 完整方案+跨文档    : λ=1000, para 锚点 + EdgeSchedule（含跨文档边）+ Path A

实验目的：
  - [0] vs [1]：验证锚点粒度对结构熵的影响（熵是否非零）
  - [0] vs [2]：验证分层加边对层次数的影响（图连通性）
  - [1] vs [3]：验证分层加边在有意义熵约束下的效果
  - [2] vs [3]：验证锚点粒度在有分层加边时的效果
  - [3] vs [4]：验证 Path A 的额外贡献
  - [3] vs [5]：验证跨文档边的贡献

公平性保证：
  所有组别的层数由图收敛性决定（v5 移除了 lambda < 1e-6 终止条件），
  不再被 lambda 阈值截断，确保不同 lambda 值的实验在相同条件下对比。

特性：
  - 自动检测 spaCy 可用性，不可用时降级为正则切句
  - 支持快速模式（--n-qa 50）和完整模式（--n-qa 200）
  - 实时打印进度，结果保存为 JSON + 控制台表格

用法：
  # 快速冒烟（50 条 QA）
  python3 -m graphrag_improved.experiments.run_multihop_eval --n-qa 50

  # 完整实验（200 条 QA）
  python3 -m graphrag_improved.experiments.run_multihop_eval --n-qa 200

  # 全量实验（2556 条 QA，耗时较长）
  python3 -m graphrag_improved.experiments.run_multihop_eval

  # 只跑前三组（快速验证）
  python3 -m graphrag_improved.experiments.run_multihop_eval --n-qa 50 --groups 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 路径设置
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.edge_scheduler import EdgeSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection
from graphrag_improved.data.ingestion import (
    Document, SentenceUnit, TextUnit,
    _make_para_id, _make_sent_id,
    document_to_text_units,
)
from graphrag_improved.evaluation.evaluator import (
    CommunityMetrics, Evaluator, QAPair as EvalQAPair, RetrievalMetrics,
)
from graphrag_improved.experiments.data_loader import (
    MultiHopDataset, corpus_to_text_units, load_multihop_dataset,
)
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.pipeline_config import ExtractionConfig
from graphrag_improved.retrieval.retriever import URetriever


# ---------------------------------------------------------------------------
# 环境检测
# ---------------------------------------------------------------------------

def _check_spacy() -> bool:
    """检查 spaCy 和 en_core_web_sm 是否可用。"""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


SPACY_AVAILABLE = _check_spacy()


# ---------------------------------------------------------------------------
# 数据准备：corpus → TextUnit（含 SentenceUnit）
# ---------------------------------------------------------------------------

def _split_sentences_batch(texts: List[str], verbose: bool = False) -> List[List[str]]:
    """
    批量 spaCy 切句（nlp.pipe 比逐篇快 3-5x）。
    返回与 texts 等长的句子列表的列表。
    """
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    results = []
    # 截断超长文本，避免 spaCy 内存溢出
    truncated = [t[:30000] for t in texts]
    total = len(truncated)
    t0 = time.time()
    for doc in nlp.pipe(truncated, batch_size=64):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        results.append(sents if sents else [truncated[len(results)][:500]])
        if verbose and len(results) % 500 == 0:
            print(f"    spaCy 切句进度：{len(results)}/{total}  ({time.time()-t0:.1f}s)")
    return results


def _split_sentences_fallback_batch(texts: List[str]) -> List[List[str]]:
    """正则批量切句（降级方案）。"""
    import re
    results = []
    for text in texts:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        sents = [s.strip() for s in parts if s.strip()]
        results.append(sents if sents else [text.strip()])
    return results


def build_pipeline_text_units(
    dataset: MultiHopDataset,
    relevant_doc_ids: Optional[set] = None,
    verbose: bool = True,
    use_regex: bool = False,
) -> List[TextUnit]:
    """
    将 MultiHop-RAG corpus 转换为 v3 格式的 TextUnit 列表。

    优化：
    1. relevant_doc_ids 不为 None 时，只处理 QA 涉及的文章（大幅减少文章数）
    2. 使用 nlp.pipe 批量处理（比逐篇快 3-5x）
    3. use_regex=True 时强制使用正则切句（全量 corpus 时推荐，速度快 10x）

    spaCy 可用时：使用 spaCy 依存句法切句（精确）
    spaCy 不可用时：使用正则切句（降级，仍能生成 SentenceUnit）
    """
    # 过滤：只处理相关文章
    if relevant_doc_ids is not None:
        docs_to_process = [d for d in dataset.corpus if d.doc_id in relevant_doc_ids]
    else:
        docs_to_process = dataset.corpus

    use_spacy = SPACY_AVAILABLE and not use_regex
    mode = "spaCy批量" if use_spacy else "正则（快速）"
    if verbose:
        print(f"  [数据] 构建 TextUnit（{len(docs_to_process)}/{len(dataset.corpus)} 篇文章，切句模式：{mode}）...")

    t0 = time.time()

    # 先做段落切分（不依赖 spaCy）
    from graphrag_improved.data.ingestion import _split_paragraphs
    import re as _re

    # 收集所有段落文本，用于批量切句
    all_para_texts: List[str] = []
    para_meta: List[tuple] = []  # (doc_id, doc_title, source, para_idx)

    for doc in docs_to_process:
        paragraphs = _split_paragraphs(doc.body)
        for para_idx, para_text in enumerate(paragraphs):
            all_para_texts.append(para_text)
            para_meta.append((doc.doc_id, doc.title, doc.source, para_idx))

    if verbose:
        print(f"    共 {len(all_para_texts)} 个段落，开始批量切句...")

    # 批量切句
    if use_spacy:
        try:
            all_sent_lists = _split_sentences_batch(all_para_texts, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"    [警告] spaCy 批量切句失败（{e}），降级为正则")
            all_sent_lists = _split_sentences_fallback_batch(all_para_texts)
    else:
        all_sent_lists = _split_sentences_fallback_batch(all_para_texts)

    # 组装 TextUnit
    units: List[TextUnit] = []
    for i, (para_text, (doc_id, doc_title, source, para_idx), sent_list) in enumerate(
        zip(all_para_texts, para_meta, all_sent_lists)
    ):
        para_id = _make_para_id(doc_id, para_idx)
        sentence_units: List[SentenceUnit] = []
        for sent_idx, sent_text in enumerate(sent_list):
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            sent_id = _make_sent_id(para_id, sent_idx)
            sentence_units.append(SentenceUnit(
                sent_id=sent_id,
                text=sent_text,
                doc_id=doc_id,
                para_id=para_id,
                sent_index=sent_idx,
                para_index=para_idx,
            ))
        if not sentence_units:
            continue
        units.append(TextUnit(
            chunk_id=para_id,
            text=para_text,
            doc_id=doc_id,
            doc_title=doc_title,
            chunk_index=para_idx,
            sentences=sentence_units,
            metadata={"source": source},
        ))

    total_sents = sum(len(u.sentences) for u in units)
    if verbose:
        print(f"  [数据] 完成：{len(units)} 个段落，{total_sents} 个句子  ({time.time()-t0:.1f}s)")

    return units


# ---------------------------------------------------------------------------
# 单次实验
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    lambda_init: float
    lambda_min: float = 0.0
    annealing_schedule: str = "exponential"
    decay_rate: float = 0.5
    max_cluster_size: int = 10
    max_iterations: int = 10
    seed: int = 42
    top_k_communities: int = 5
    top_k_chunks: int = 5
    # v5 新增：锚点粒度（"sent" = v4 兼容，"para" = v5 默认）
    anchor_granularity: str = "para"
    # v5 新增：是否使用分层加边
    use_edge_schedule: bool = False
    # v5 新增：EdgeSchedule 是否包含跨文档边
    edge_schedule_cross_doc: bool = True
    # 路径A：文档内实体消解
    intra_doc_merging: bool = False
    intra_doc_edge_weight: float = 0.5


@dataclass
class RunResult:
    config: RunConfig
    community_metrics: CommunityMetrics
    retrieval_metrics: RetrievalMetrics
    elapsed_seconds: float
    num_entities: int
    num_relationships: int
    num_communities: int
    num_levels: int


def run_one(
    cfg: RunConfig,
    text_units: List[TextUnit],
    retrieval_units: List[dict],
    eval_qa: List[EvalQAPair],
    entities_df,       # 共享的抽取结果（两个版本复用）
    relationships_df,
    verbose: bool = True,
) -> RunResult:
    """运行单次实验（只做社区检测 + 评估，抽取结果复用）。"""
    t0 = time.time()

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  实验：{cfg.name}  (λ={cfg.lambda_init}, anchor={cfg.anchor_granularity})")
        print(f"{'─'*55}")

    # 社区检测
    if verbose:
        print(f"  [1/2] 社区检测 (λ={cfg.lambda_init}, anchor={cfg.anchor_granularity})...")
        t1 = time.time()

    annealing_config = AnnealingConfig(
        lambda_init=cfg.lambda_init,
        lambda_min=cfg.lambda_min,
        schedule=AnnealingSchedule(cfg.annealing_schedule),
        decay_rate=cfg.decay_rate,
        max_level=10,
    )

    # v5 新增：构建 EdgeSchedule（若启用）
    edge_schedule = None
    if cfg.use_edge_schedule:
        edge_schedule = EdgeSchedule.build(
            entities_df,
            include_cross_doc=cfg.edge_schedule_cross_doc,
        )
        if verbose:
            print(f"        EdgeSchedule: {edge_schedule.summary()}")

    communities_df = run_constrained_community_detection(
        entities=entities_df,
        relationships=relationships_df,
        annealing_config=annealing_config,
        max_cluster_size=cfg.max_cluster_size,
        max_iterations=cfg.max_iterations,
        seed=cfg.seed,
        use_lcc=False,   # 处理所有连通分量（v3 图碎片化，不能只用 LCC）
        intra_doc_merging=cfg.intra_doc_merging,
        intra_doc_edge_weight=cfg.intra_doc_edge_weight,
        anchor_granularity=cfg.anchor_granularity,
        edge_schedule=edge_schedule,
    )
    num_levels = communities_df["level"].nunique() if not communities_df.empty else 0
    if verbose:
        print(f"        社区 {len(communities_df)} 个，层次 {num_levels} 层  ({time.time()-t1:.1f}s)")

    # 检索 + 评估
    if verbose:
        print(f"  [2/2] 检索评估（{len(eval_qa)} 条 QA）...")
        t2 = time.time()

    retriever = URetriever(
        communities_df=communities_df,
        text_units=retrieval_units,
        entities_df=entities_df,
        top_k_communities=cfg.top_k_communities,
        top_k_chunks=cfg.top_k_chunks,
    )
    evaluator = Evaluator()
    community_metrics = evaluator.evaluate_community_quality(communities_df, relationships_df)
    retrieval_metrics = evaluator.evaluate_retrieval(eval_qa, retriever, k_values=[1, 3, 5, 10])

    elapsed = time.time() - t0
    if verbose:
        print(f"        MRR={retrieval_metrics.mrr:.4f}  "
              f"P@5={retrieval_metrics.precision_at_k.get(5,0):.4f}  "
              f"NDCG@10={retrieval_metrics.ndcg_at_k.get(10,0):.4f}  "
              f"({time.time()-t2:.1f}s)")
        print(f"        Level0 纯净率={community_metrics.level0_purity_rate:.2%}  "
              f"平均结构熵={community_metrics.avg_structural_entropy:.4f}")
        print(f"  总耗时：{elapsed:.1f}s")

    return RunResult(
        config=cfg,
        community_metrics=community_metrics,
        retrieval_metrics=retrieval_metrics,
        elapsed_seconds=elapsed,
        num_entities=len(entities_df),
        num_relationships=len(relationships_df),
        num_communities=len(communities_df),
        num_levels=num_levels,
    )


# ---------------------------------------------------------------------------
# 主实验流程
# ---------------------------------------------------------------------------

def _load_or_extract(
    text_units: List[TextUnit],
    cache_dir: Path,
    cache_tag: str,
    verbose: bool = True,
):
    """加载缓存的抽取结果，或重新抽取并保存缓存。"""
    import pandas as pd
    entities_cache = cache_dir / f"entities_{cache_tag}.parquet"
    rels_cache     = cache_dir / f"relationships_{cache_tag}.parquet"

    if entities_cache.exists() and rels_cache.exists():
        if verbose:
            print(f"  [缓存] 加载 {cache_tag} 抽取结果...")
        entities_df     = pd.read_parquet(entities_cache)
        relationships_df = pd.read_parquet(rels_cache)
        if verbose:
            print(f"  实体 {len(entities_df)} 个，关系 {len(relationships_df)} 条  (from cache)")
    else:
        t0 = time.time()
        extraction_config = ExtractionConfig(
            backend="spacy",
            spacy_model="en_core_web_sm",
            min_entity_freq=1,
        )
        entities_df, relationships_df = extract(text_units, extraction_config)
        if verbose:
            print(f"  实体 {len(entities_df)} 个，关系 {len(relationships_df)} 条  ({time.time()-t0:.1f}s)")
        entities_df.to_parquet(entities_cache, index=False)
        relationships_df.to_parquet(rels_cache, index=False)
        if verbose:
            print(f"  [缓存] 已保存 {cache_tag}")

    return entities_df, relationships_df


def run_experiment(
    data_dir: str = "./data/multihop_rag",
    n_qa: Optional[int] = 200,
    output_dir: str = "./experiments/results",
    question_type: Optional[str] = None,
    verbose: bool = True,
    use_regex: bool = True,
    groups: Optional[List[int]] = None,
) -> List[RunResult]:
    """
    六组消融实验（v5 渐进合并架构）：

      [0] Baseline          : lambda=0, sent 锚点（纯模块度 Leiden）
      [1] 仅改锚点          : lambda=1000, para 锚点（结构熵有意义）
      [2] 仅分层加边        : lambda=1000, sent 锚点 + EdgeSchedule
      [3] 锚点+分层加边      : lambda=1000, para 锚点 + EdgeSchedule（完整方案）
      [4] 完整方案+Path A    : lambda=1000, para 锚点 + EdgeSchedule + 文档内消解边
      [5] 完整方案+跨文档    : lambda=1000, para 锚点 + EdgeSchedule（含跨文档边）+ Path A

    所有组别共享同一份抽取结果（只有社区检测阶段的参数不同）。

    Parameters
    ----------
    groups : List[int], optional
        只运行指定编号的实验组（如 [0, 1, 3]），None 表示运行全部六组
    """
    t_total = time.time()

    print("\n" + "="*65)
    print("  GraphRAG Improved — MultiHop-RAG 六组消融实验 (v5)")
    print("="*65)
    mode_str = "正则（快速）" if use_regex else ("spaCy" if SPACY_AVAILABLE else "正则降级")
    print(f"  切句模式：{mode_str}")
    if groups is not None:
        print(f"  运行组别：{groups}")

    # ── 1. 加载数据集 ────────────────────────────────────────────
    print("\n[1/5] 加载数据集...")
    dataset = load_multihop_dataset(data_dir)

    if question_type:
        dataset = dataset.filter_by_type(question_type)
        print(f"  过滤后：{dataset.num_qa} 条 {question_type} 类型问题")

    if n_qa is not None:
        dataset = dataset.subset(n_qa, seed=42)

    type_counts: Dict[str, int] = {}
    for qa in dataset.qa_pairs:
        type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1

    eval_qa = [
        EvalQAPair(
            question=qa.query,
            answer=qa.answer,
            context_ids=qa.supporting_doc_ids,
            metadata={"question_type": qa.question_type},
        )
        for qa in dataset.qa_pairs
        if qa.supporting_doc_ids
    ]

    print(f"  文章数：{dataset.num_docs}，QA 对：{dataset.num_qa}（有效：{len(eval_qa)}）")
    print(f"  问题类型：{type_counts}")

    # ── 2. 构建 TextUnit（只处理 QA 涉及的文章）──────────────────
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    relevant_doc_ids: set = set()
    for qa in dataset.qa_pairs:
        relevant_doc_ids.update(qa.supporting_doc_ids)

    print(f"\n[2/5] 构建 TextUnit（相关文章 {len(relevant_doc_ids)}/{dataset.num_docs} 篇）...")
    text_units = build_pipeline_text_units(
        dataset, relevant_doc_ids=relevant_doc_ids if relevant_doc_ids else None,
        verbose=verbose, use_regex=use_regex
    )
    retrieval_units = corpus_to_text_units(dataset.corpus)

    # ── 3. 实体抽取（共享，带缓存）──────────────────────────────
    cache_tag = f"n{n_qa}" if n_qa else "full"
    print("\n[3/5] 实体抽取（所有组别共享）...")
    entities_df, relationships_df = _load_or_extract(
        text_units, cache_dir, cache_tag, verbose=verbose
    )

    if len(entities_df) == 0:
        print("  [警告] 未抽取到任何实体，请检查 spaCy 安装")

    # ── 4. 六组消融实验 ──────────────────────────────────────────
    all_configs = [
        RunConfig(
            name="[0] Baseline (lambda=0, sent)",
            lambda_init=0.0,
            anchor_granularity="sent",
            use_edge_schedule=False,
            intra_doc_merging=False,
        ),
        RunConfig(
            name="[1] 仅改锚点 (lambda=1000, para)",
            lambda_init=1000.0,
            anchor_granularity="para",
            use_edge_schedule=False,
            intra_doc_merging=False,
        ),
        RunConfig(
            name="[2] 仅分层加边 (lambda=1000, sent+EdgeSchedule)",
            lambda_init=1000.0,
            anchor_granularity="sent",
            use_edge_schedule=True,
            edge_schedule_cross_doc=False,
            intra_doc_merging=False,
        ),
        RunConfig(
            name="[3] 锚点+分层加边 (lambda=1000, para+EdgeSchedule)",
            lambda_init=1000.0,
            anchor_granularity="para",
            use_edge_schedule=True,
            edge_schedule_cross_doc=False,
            intra_doc_merging=False,
        ),
        RunConfig(
            name="[4] 完整方案+PathA (lambda=1000, para+EdgeSchedule+PathA)",
            lambda_init=1000.0,
            anchor_granularity="para",
            use_edge_schedule=True,
            edge_schedule_cross_doc=False,
            intra_doc_merging=True,
            intra_doc_edge_weight=0.5,
        ),
        RunConfig(
            name="[5] 完整方案+跨文档 (lambda=1000, para+EdgeSchedule+CrossDoc+PathA)",
            lambda_init=1000.0,
            anchor_granularity="para",
            use_edge_schedule=True,
            edge_schedule_cross_doc=True,
            intra_doc_merging=True,
            intra_doc_edge_weight=0.5,
        ),
    ]

    # 过滤：只运行指定组别
    if groups is not None:
        selected_configs = [all_configs[i] for i in groups if 0 <= i < len(all_configs)]
    else:
        selected_configs = all_configs

    print(f"\n[4/5] 运行 {len(selected_configs)} 组消融实验...")

    results: List[RunResult] = []
    for cfg in selected_configs:
        r = run_one(cfg, text_units, retrieval_units, eval_qa,
                    entities_df, relationships_df, verbose=verbose)
        results.append(r)

    # ── 5. 保存 & 打印结果 ───────────────────────────────────────
    print("\n[5/5] 保存结果...")
    _save_and_print_multi(results, output_dir, n_qa, len(eval_qa))

    print(f"\n实验完成！总耗时 {time.time()-t_total:.1f}s")
    return results


# ---------------------------------------------------------------------------
# 结果保存与打印
# ---------------------------------------------------------------------------

def _result_to_dict(r: RunResult) -> dict:
    rm = r.retrieval_metrics
    cm = r.community_metrics
    return {
        "name": r.config.name,
        "lambda_init": r.config.lambda_init,
        "spacy_available": SPACY_AVAILABLE,
        "num_entities": r.num_entities,
        "num_relationships": r.num_relationships,
        "num_communities": r.num_communities,
        "num_levels": r.num_levels,
        "elapsed_seconds": round(r.elapsed_seconds, 2),
        # 社区质量
        "modularity": round(cm.modularity, 4),
        "avg_structural_entropy": round(cm.avg_structural_entropy, 4),
        "level0_purity_rate": round(cm.level0_purity_rate, 4),
        "avg_community_size": round(cm.avg_community_size, 2),
        "entropy_by_level": {str(k): round(v, 4) for k, v in cm.entropy_by_level.items()},
        # 检索质量
        "mrr": round(rm.mrr, 4),
        "precision_at_1": round(rm.precision_at_k.get(1, 0), 4),
        "precision_at_3": round(rm.precision_at_k.get(3, 0), 4),
        "precision_at_5": round(rm.precision_at_k.get(5, 0), 4),
        "precision_at_10": round(rm.precision_at_k.get(10, 0), 4),
        "recall_at_5": round(rm.recall_at_k.get(5, 0), 4),
        "recall_at_10": round(rm.recall_at_k.get(10, 0), 4),
        "f1_at_5": round(rm.f1_at_k.get(5, 0), 4),
        "ndcg_at_5": round(rm.ndcg_at_k.get(5, 0), 4),
        "ndcg_at_10": round(rm.ndcg_at_k.get(10, 0), 4),
        "num_queries": rm.num_queries,
    }


def _save_and_print_multi(
    results: List[RunResult],
    output_dir: str,
    n_qa: Optional[int],
    valid_qa: int,
) -> None:
    """保存并打印多组（≥2）对照实验结果。

    results[0] 视为 Baseline，其余各组均与 Baseline 计算相对提升。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dicts = [_result_to_dict(r) for r in results]
    b = dicts[0]  # Baseline

    def _delta(key: str, target: dict) -> str:
        bv = b.get(key, 0) or 0
        tv = target.get(key, 0) or 0
        if bv == 0:
            return "N/A"
        d = (tv - bv) / abs(bv) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    # 构造 JSON 输出
    output_data: dict = {
        "meta": {
            "n_qa_requested": n_qa,
            "valid_qa": valid_qa,
            "spacy_available": SPACY_AVAILABLE,
        },
    }
    for d in dicts:
        key = d["name"].replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "")
        output_data[key] = d

    # 各组 vs Baseline 的提升
    improvements = []
    for d in dicts[1:]:
        improvements.append({
            "vs": d["name"],
            "mrr": _delta("mrr", d),
            "precision_at_5": _delta("precision_at_5", d),
            "ndcg_at_5": _delta("ndcg_at_5", d),
            "recall_at_10": _delta("recall_at_10", d),
            "ndcg_at_10": _delta("ndcg_at_10", d),
            "level0_purity_rate": _delta("level0_purity_rate", d),
            "avg_structural_entropy": _delta("avg_structural_entropy", d),
        })
    output_data["improvements_vs_baseline"] = improvements

    # 保存 JSON
    suffix = f"_n{n_qa}" if n_qa else "_full"
    result_path = out / f"multihop_results{suffix}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # ── 打印对比表格 ──────────────────────────────────────────────
    n_groups = len(dicts)
    # 列宽：指标列 22，每组数据列 14，提升列 10
    col_metric = 22
    col_val    = 14
    col_delta  = 10

    # 表头
    total_width = col_metric + 2 + n_groups * (col_val + 2) + (n_groups - 1) * (col_delta + 2)
    sep = "=" * max(total_width, 80)

    print("\n" + sep)
    print("  MultiHop-RAG 消融实验结果 (v5)")
    print(f"  QA 对：{valid_qa} 条有效  |  切句：{'spaCy' if SPACY_AVAILABLE else '正则降级'}")
    print(sep)

    # 列标题行
    header = f"  {'指标':<{col_metric}}"
    for d in dicts:
        short = d["name"][:col_val-1]
        header += f"  {short:<{col_val}}"
        if d is not b:
            header += f"  {'vs[0]':<{col_delta}}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    def _fmt_row(label: str, key: str, fmt: str = ".4f") -> None:
        line = f"  {label:<{col_metric}}"
        for d in dicts:
            val = d.get(key, 0) or 0
            if fmt == ".2%":
                cell = f"{val:.2%}"
            elif fmt == "int":
                cell = str(int(val))
            else:
                cell = f"{val:{fmt}}"
            line += f"  {cell:<{col_val}}"
            if d is not b:
                line += f"  {_delta(key, d):<{col_delta}}"
        print(line)

    def _sep_row() -> None:
        print("  " + "─" * (len(header) - 2))

    # 检索质量
    _fmt_row("MRR",           "mrr")
    _fmt_row("Precision@1",   "precision_at_1")
    _fmt_row("Precision@3",   "precision_at_3")
    _fmt_row("Precision@5",   "precision_at_5")
    _fmt_row("Precision@10",  "precision_at_10")
    _fmt_row("Recall@5",      "recall_at_5")
    _fmt_row("Recall@10",     "recall_at_10")
    _fmt_row("NDCG@5",        "ndcg_at_5")
    _fmt_row("NDCG@10",       "ndcg_at_10")
    _sep_row()
    # 社区质量
    _fmt_row("模块度 Q",       "modularity")
    _fmt_row("平均结构熵",      "avg_structural_entropy")
    _fmt_row("Level0 纯净率",  "level0_purity_rate", fmt=".2%")
    _fmt_row("社区数量",        "num_communities", fmt="int")
    _fmt_row("层次数量",        "num_levels",       fmt="int")
    _fmt_row("实体数量",        "num_entities",     fmt="int")
    _fmt_row("关系数量",        "num_relationships", fmt="int")

    print(sep)
    print(f"\n  结果已保存：{result_path}")

    # 各层结构熵对比（以 Baseline 和最后一组为例）
    last = dicts[-1]
    if last.get("entropy_by_level"):
        print("\n  各层平均结构熵（Baseline vs 最优组）：")
        all_levels = sorted(
            set(list(b.get("entropy_by_level", {}).keys()) +
                list(last.get("entropy_by_level", {}).keys())),
            key=lambda x: int(x),
        )
        for lv in all_levels:
            bh = b.get("entropy_by_level", {}).get(lv, 0)
            lh = last.get("entropy_by_level", {}).get(lv, 0)
            print(f"    Level {lv}: baseline={bh:.4f}  best={lh:.4f}  "
                  f"delta={lh-bh:+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MultiHop-RAG 六组消融实验 (v5 渐进合并架构)"
    )
    parser.add_argument("--data-dir", default="./data/multihop_rag",
                        help="数据集目录")
    parser.add_argument("--n-qa", type=int, default=200,
                        help="使用的 QA 对数量（默认 200，None=全量）")
    parser.add_argument("--output-dir", default="./experiments/results",
                        help="结果保存目录")
    parser.add_argument("--question-type", default=None,
                        choices=["inference_query", "comparison_query",
                                 "temporal_query", "null_query"],
                        help="只评估特定类型的问题")
    parser.add_argument("--full", action="store_true",
                        help="使用全量 QA 对（覆盖 --n-qa）")
    parser.add_argument("--use-spacy", action="store_true",
                        help="使用 spaCy 切句（精确但慢，默认使用正则快速切句）")
    parser.add_argument("--groups", default=None,
                        help="只运行指定组别，逗号分隔（如 '0,1,3'），默认运行全部六组")
    args = parser.parse_args()

    n_qa = None if args.full else args.n_qa
    use_regex = not args.use_spacy

    groups = None
    if args.groups:
        groups = [int(g.strip()) for g in args.groups.split(",") if g.strip().isdigit()]

    run_experiment(
        data_dir=args.data_dir,
        n_qa=n_qa,
        output_dir=args.output_dir,
        question_type=args.question_type,
        verbose=True,
        use_regex=use_regex,
        groups=groups,
    )


if __name__ == "__main__":
    main()
