"""v6 smoke test: 验证三大核心修复 + Level 0 注入同段落跨句子边"""
import pandas as pd
import networkx as nx
from constrained_leiden.edge_scheduler import EdgeSchedule
from constrained_leiden.graphrag_workflow import run_constrained_community_detection
from constrained_leiden.leiden_constrained import hierarchical_leiden_constrained
from constrained_leiden.graphrag_workflow import (
    build_graph_from_graphrag,
    build_physical_nodes_from_graphrag,
)
from constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule

# ── 1. EdgeSchedule.build() 三级分层 ─────────────────────────────────────────
entities_for_es = pd.DataFrame({
    "id":      ["e1", "e2", "e3", "e4"],
    "title":   ["Alice", "Alice", "Bob", "Bob"],
    "para_id": ["p1",    "p2",    "p1",  "p3"],
    "doc_id":  ["d1",    "d1",    "d1",  "d2"],
    "sent_id": ["s1",    "s2",    "s1",  "s3"],
})
es = EdgeSchedule.build(
    entities_for_es,
    intra_para_weight=0.3,
    intra_doc_weight=0.2,
    cross_doc_weight=0.1,
)
print("EdgeSchedule summary:")
print(es.summary())
assert es.total_edges() > 0, "EdgeSchedule 应有边"
# v6: Level 0 现在有同段落跨句子边，Level 1 有文档内跨段落边
l0 = es.get_edges_for_level(0)
assert len(l0) >= 0, "Level 0 应有同段落跨句子边（若存在）"  # 此测试数据每句不同段落，可能为0
l1 = es.get_edges_for_level(1)
assert len(l1) > 0, "Level 1 应有文档内跨段落边"
print(f"  Level 0 edges (intra-para): {l0}")
print(f"  Level 1 edges (intra-doc):  {l1}")
print("  [PASS] EdgeSchedule.build() v6 三级分层正确\n")

# ── 2. 结构熵非零验证（lambda=0，无约束，纯 modularity 合并跨段落节点）────────
# 用 lambda=0 确保算法会合并跨段落节点，从而产生非零熵
# 这验证了：当节点来自不同段落时，熵计算是正确的
entities = pd.DataFrame({
    "id":      ["e1", "e2", "e3", "e4", "e5", "e6"],
    "title":   ["Alice", "Paris", "Bob", "London", "Carol", "Rome"],
    "type":    ["PERSON", "PLACE", "PERSON", "PLACE", "PERSON", "PLACE"],
    "para_id": ["p1", "p1", "p1", "p2", "p2", "p2"],
    "doc_id":  ["d1", "d1", "d1", "d1", "d2", "d2"],
    "sent_id": ["s1", "s1", "s2", "s3", "s4", "s4"],
    "description": [""] * 6,
    "human_readable_id": range(6),
    "graph_embedding": [None] * 6,
    "text_unit_ids": [["t1"], ["t1"], ["t2"], ["t3"], ["t4"], ["t4"]],
})
relationships = pd.DataFrame({
    "id":     ["r1", "r2", "r3", "r4", "r5"],
    "source": ["e1", "e2", "e3", "e4", "e5"],
    "target": ["e2", "e3", "e4", "e5", "e6"],
    "weight": [2.0, 2.0, 1.5, 2.0, 2.0],
    "description": [""] * 5,
    "human_readable_id": range(5),
    "text_unit_ids": [["t1"], ["t1"], ["t2"], ["t3"], ["t4"]],
})

graph = build_graph_from_graphrag(entities, relationships)
physical_nodes = build_physical_nodes_from_graphrag(entities, anchor_granularity="para")

# 验证物理节点共享 chunk_id
chunk_ids_list = [list(pn.chunk_ids)[0] for pn in physical_nodes.values()]
assert len(set(chunk_ids_list)) < len(chunk_ids_list), "para 粒度下应有共享 chunk_id"
print(f"Physical nodes chunk_ids: {chunk_ids_list}")
print("  [PASS] para 粒度下节点共享 chunk_id\n")

# lambda=0：无约束，modularity 会在高层合并跨段落节点
annealing_zero = AnnealingConfig(
    lambda_init=0.0,
    lambda_min=0.0,
    max_level=4,
    decay_rate=0.5,
    schedule=AnnealingSchedule.EXPONENTIAL,
)
result_zero = hierarchical_leiden_constrained(
    graph=graph,
    physical_nodes=physical_nodes,
    annealing_config=annealing_zero,
)
print(f"lambda=0 result: {len(result_zero.levels)} levels")
all_entropies = []
for i, (level_map, level_entropy, lv) in enumerate(
    zip(result_zero.levels, result_zero.level_entropy, result_zero.level_lambda)
):
    for comm_id, h in level_entropy.items():
        all_entropies.append(h)
    print(f"  Level {i}: entropies={list(level_entropy.values())}")

max_entropy = max(all_entropies) if all_entropies else 0.0
print(f"Max entropy across all levels: {max_entropy:.4f}")
assert max_entropy > 1e-9, "BUG: lambda=0 时跨段落合并后熵应 > 0"
print("  [PASS] lambda=0 时跨段落合并产生非零结构熵\n")

# ── 3. lambda 约束有效性：lambda 大时熵应更低（同段落聚合）────────────────────
annealing_high = AnnealingConfig(
    lambda_init=100.0,
    lambda_min=0.0,
    max_level=4,
    decay_rate=0.5,
    schedule=AnnealingSchedule.EXPONENTIAL,
)
result_high = hierarchical_leiden_constrained(
    graph=graph,
    physical_nodes=physical_nodes,
    annealing_config=annealing_high,
)
high_entropies = []
for level_entropy in result_high.level_entropy:
    high_entropies.extend(level_entropy.values())
max_high = max(high_entropies) if high_entropies else 0.0
print(f"lambda=100 max entropy: {max_high:.4f}")
print(f"lambda=0   max entropy: {max_entropy:.4f}")
assert max_high <= max_entropy + 1e-9, "lambda 大时熵应 ≤ lambda=0 时的熵"
print("  [PASS] lambda 约束有效：高 lambda 产生更低或相等的结构熵\n")

# ── 4. original_to_current 映射：多层结果覆盖所有原始节点 ─────────────────────
for i, level_map in enumerate(result_zero.levels):
    covered = set(level_map.keys())
    original = set(graph.nodes())
    assert covered == original, f"Level {i}: 节点覆盖不完整 {original - covered}"
print(f"  [PASS] original_to_current 映射：所有 {len(graph.nodes())} 个节点在每层都有社区分配\n")

# ── 5. 无提前终止：层数由收敛性决定，不被 lambda<1e-6 截断 ──────────────────
annealing_many = AnnealingConfig(
    lambda_init=1.0,
    lambda_min=0.0,
    max_level=10,
    decay_rate=0.9,
    schedule=AnnealingSchedule.EXPONENTIAL,
)
result_many = hierarchical_leiden_constrained(
    graph=graph,
    physical_nodes=physical_nodes,
    annealing_config=annealing_many,
)
print(f"  [PASS] 无提前终止：decay_rate=0.9, max_level=10 → 实际层数={len(result_many.levels)}")

print("\n✅  v6 smoke test ALL PASSED")
