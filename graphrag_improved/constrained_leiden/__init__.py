"""
constrained_leiden
------------------
带结构熵惩罚的 Leiden 社区发现算法包。

v5 新增：
    - edge_scheduler.EdgeSchedule：分层加边调度器
    - graphrag_workflow.AnchorGranularity：锚点粒度类型
"""

from .annealing import AnnealingConfig, AnnealingSchedule, get_lambda
from .edge_scheduler import EdgeSchedule
from .graphrag_workflow import (
    AnchorGranularity,
    build_cross_document_edges,
    build_graph_from_graphrag,
    build_intra_doc_entity_edges,
    build_intra_paragraph_edges,
    build_physical_nodes_from_graphrag,
    convert_result_to_communities_df,
    run_constrained_community_detection,
)
from .leiden_constrained import (
    HierarchicalCommunityResult,
    hierarchical_leiden_constrained,
)
from .physical_anchor import PhysicalNode, compute_structural_entropy

__all__ = [
    # annealing
    "AnnealingConfig",
    "AnnealingSchedule",
    "get_lambda",
    # edge_scheduler
    "EdgeSchedule",
    # graphrag_workflow
    "AnchorGranularity",
    "build_cross_document_edges",
    "build_graph_from_graphrag",
    "build_intra_doc_entity_edges",
    "build_intra_paragraph_edges",
    "build_physical_nodes_from_graphrag",
    "convert_result_to_communities_df",
    "run_constrained_community_detection",
    # leiden_constrained
    "HierarchicalCommunityResult",
    "hierarchical_leiden_constrained",
    # physical_anchor
    "PhysicalNode",
    "compute_structural_entropy",
]
