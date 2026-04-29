"""
edge_scheduler.py
-----------------
分层加边调度器（v5 新增）。

设计目标：
    实现渐进合并架构的核心组件之一。
    通过逐层注入更远距离的边，让 Leiden 算法能够发现
    跨句子、跨段落、跨文档的社区结构。

边注入策略（v6 修改：所有边在 Level 0 一次性注入）：
    Level 0: 同时注入三类同名实体边
             - 同段落跨句子边（weight=0.3）
             - 同文档跨段落边（weight=0.2）
             - 跨文档边（weight=0.1，可选）
             v6 根因：仅注入同段落边不够（para_id 相同 → 熵仍为 0），
             必须有跨段落的连通性才能产生跨段落社区 → 非零熵。
    Level 1+: 无额外边（所有边已在 Level 0 注入）

权重设计原则：
    距离越远的边权重越低，使 Leiden 的模块度增益 delta_Q 对远距离合并的驱动力更弱。
    配合 lambda 退火的 delta_H 惩罚递减，形成双重控制：
    - 低层（lambda 大）：delta_H 惩罚强 + 只有近距离边 -> 优先合并同段落节点
    - 高层（lambda 小）：delta_H 惩罚弱 + 远距离边注入 -> 允许跨文档合并

与 Path A 的关系：
    Path A（build_intra_doc_entity_edges）是一次性注入所有文档内边，
    EdgeSchedule 是按层次分批注入，两者可以独立使用或组合使用。
    推荐使用 EdgeSchedule 替代 Path A，以获得更精细的控制。

使用方式：
    from constrained_leiden.edge_scheduler import EdgeSchedule

    schedule = EdgeSchedule.build(entities_df)
    communities_df = run_constrained_community_detection(
        entities, relationships,
        edge_schedule=schedule,
        anchor_granularity="para",
    )
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 噪声实体名过滤（与 graphrag_workflow.py 保持一致）
# ---------------------------------------------------------------------------

_NOISE_TITLES = frozenset({
    "which", "there", "one", "the company", "available", "a lot",
    "people", "able", "good", "more", "something", "some",
    "get", "make", "use", "take", "have", "be", "do", "go",
    "it", "this", "that", "these", "those", "they", "we",
    "he", "she", "who", "what", "where", "when", "how",
    "the ball", "the game", "the team", "the player",
    "there", "There",
})


def _is_noise_title(title: str) -> bool:
    """判断实体名是否为噪声（过短或在停用词表中）。"""
    if len(title) < 3:
        return True
    if title in _NOISE_TITLES:
        return True
    if title.lower() in _NOISE_TITLES:
        return True
    return False


# ---------------------------------------------------------------------------
# 辅助：从 sent_id 解析 para_id / doc_id
# ---------------------------------------------------------------------------

def _parse_para_id(sent_id: str) -> str:
    """从 sent_id 中提取段落 ID。格式：{doc_id}-p{NNN}-s{NNN} -> {doc_id}-p{NNN}"""
    parts = sent_id.rsplit("-s", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return sent_id


def _parse_doc_id(sent_id: str) -> str:
    """从 sent_id 中提取文档 ID。格式：{doc_id}-p{NNN}-s{NNN} -> {doc_id}"""
    parts = sent_id.rsplit("-p", 1)
    if len(parts) == 2 and parts[1].split("-")[0].isdigit():
        return parts[0]
    return sent_id


# ---------------------------------------------------------------------------
# 边构建辅助函数
# ---------------------------------------------------------------------------

def _build_intra_paragraph_edges(
    entities: pd.DataFrame,
    weight: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """
    同段落内、不同句子的同名实体之间的边。

    匹配条件：
    - 同一段落（para_id 相同）
    - 不同句子（sent_id 不同）
    - 实体名相同（title 大小写不敏感）

    连接方式：链式连接（避免 O(n^2) 爆炸）
    """
    if "title" not in entities.columns:
        return []

    has_para_id = "para_id" in entities.columns
    has_sent_id = "sent_id" in entities.columns

    if not has_para_id and not has_sent_id:
        return []

    # 按 (para_id, title_lower) 分组，收集同段落同名实体
    # key -> [(sent_id, node_id)]
    para_title_to_nodes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for _, row in entities.iterrows():
        title = str(row.get("title", "")).strip()
        node_id = str(row["id"])

        if _is_noise_title(title):
            continue

        sent_id = ""
        if has_sent_id and row.get("sent_id") and str(row["sent_id"]).strip():
            sent_id = str(row["sent_id"]).strip()

        if has_para_id and row.get("para_id") and str(row["para_id"]).strip():
            para_id = str(row["para_id"]).strip()
        elif sent_id:
            para_id = _parse_para_id(sent_id)
        else:
            continue

        key = f"{para_id}|||{title.lower()}"
        para_title_to_nodes[key].append((sent_id, node_id))

    # 链式连接（只连接不同句子的节点）
    edges: List[Tuple[str, str, float]] = []
    for key, sent_node_pairs in para_title_to_nodes.items():
        if len(sent_node_pairs) < 2:
            continue
        # 按 sent_id 排序，保证连接顺序稳定
        sent_node_pairs.sort(key=lambda x: x[0])
        for i in range(len(sent_node_pairs) - 1):
            s1, n1 = sent_node_pairs[i]
            s2, n2 = sent_node_pairs[i + 1]
            if s1 != s2:  # 确保是不同句子
                edges.append((n1, n2, weight))

    return edges


def _build_intra_document_edges(
    entities: pd.DataFrame,
    weight: float = 0.2,
) -> List[Tuple[str, str, float]]:
    """
    同文档内、跨段落的同名实体之间的边。

    匹配条件：
    - 同一文档（doc_id 相同）
    - 不同段落（para_id 不同）
    - 实体名相同（title 大小写不敏感）

    连接方式：每个文档内，按段落顺序链式连接各段落的代表节点
    """
    if "title" not in entities.columns:
        return []

    has_doc_id = "doc_id" in entities.columns
    has_para_id = "para_id" in entities.columns
    has_sent_id = "sent_id" in entities.columns

    if not has_doc_id and not has_sent_id:
        return []

    # 按 (doc_id, title_lower) 分组，收集同文档同名实体
    # key -> [(para_id, node_id)]
    doc_title_to_nodes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for _, row in entities.iterrows():
        title = str(row.get("title", "")).strip()
        node_id = str(row["id"])

        if _is_noise_title(title):
            continue

        sent_id = ""
        if has_sent_id and row.get("sent_id") and str(row["sent_id"]).strip():
            sent_id = str(row["sent_id"]).strip()

        if has_doc_id and row.get("doc_id") and str(row["doc_id"]).strip():
            doc_id = str(row["doc_id"]).strip()
        elif sent_id:
            doc_id = _parse_doc_id(sent_id)
        else:
            continue

        if has_para_id and row.get("para_id") and str(row["para_id"]).strip():
            para_id = str(row["para_id"]).strip()
        elif sent_id:
            para_id = _parse_para_id(sent_id)
        else:
            para_id = doc_id  # 无法区分段落时退化为文档级

        key = f"{doc_id}|||{title.lower()}"
        doc_title_to_nodes[key].append((para_id, node_id))

    # 链式连接（只连接不同段落的代表节点）
    edges: List[Tuple[str, str, float]] = []
    for key, para_node_pairs in doc_title_to_nodes.items():
        if len(para_node_pairs) < 2:
            continue

        # 按 para_id 分组，每个段落取一个代表节点
        para_to_representative: Dict[str, str] = {}
        for para_id, node_id in para_node_pairs:
            if para_id not in para_to_representative:
                para_to_representative[para_id] = node_id

        representatives = sorted(para_to_representative.items())  # 按 para_id 排序
        if len(representatives) < 2:
            continue

        # 链式连接跨段落代表节点
        for i in range(len(representatives) - 1):
            _, n1 = representatives[i]
            _, n2 = representatives[i + 1]
            edges.append((n1, n2, weight))

    return edges


def _build_cross_document_edges(
    entities: pd.DataFrame,
    weight: float = 0.1,
    max_edges_per_entity: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    跨文档的同名实体之间的边。

    匹配条件：
    - 不同文档（doc_id 不同）
    - 实体名相同（title 大小写不敏感）

    连接方式：每个文档取一个代表节点，链式连接（限制最大边数）
    """
    if "title" not in entities.columns:
        return []

    has_doc_id = "doc_id" in entities.columns
    has_sent_id = "sent_id" in entities.columns

    if not has_doc_id and not has_sent_id:
        return []

    # 按 title_lower 分组，收集所有文档中的同名实体
    # title -> [(doc_id, node_id)]
    title_to_nodes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for _, row in entities.iterrows():
        title = str(row.get("title", "")).strip()
        node_id = str(row["id"])

        if _is_noise_title(title):
            continue

        sent_id = ""
        if has_sent_id and row.get("sent_id") and str(row["sent_id"]).strip():
            sent_id = str(row["sent_id"]).strip()

        if has_doc_id and row.get("doc_id") and str(row["doc_id"]).strip():
            doc_id = str(row["doc_id"]).strip()
        elif sent_id:
            doc_id = _parse_doc_id(sent_id)
        else:
            continue

        title_to_nodes[title.lower()].append((doc_id, node_id))

    # 只连接来自不同文档的同名实体（链式，限制最大边数）
    edges: List[Tuple[str, str, float]] = []
    for title_lower, doc_node_pairs in title_to_nodes.items():
        if len(doc_node_pairs) < 2:
            continue

        # 按 doc_id 分组，每个文档取一个代表节点
        doc_to_representative: Dict[str, str] = {}
        for doc_id, node_id in doc_node_pairs:
            if doc_id not in doc_to_representative:
                doc_to_representative[doc_id] = node_id

        representatives = list(doc_to_representative.values())
        if len(representatives) < 2:
            continue

        # 链式连接跨文档代表节点（限制最大边数）
        for i in range(min(len(representatives) - 1, max_edges_per_entity)):
            edges.append((representatives[i], representatives[i + 1], weight))

    return edges


# ---------------------------------------------------------------------------
# EdgeSchedule 主类
# ---------------------------------------------------------------------------

@dataclass
class EdgeSchedule:
    """
    分层边调度器：控制每一层引入什么距离的边。

    Attributes
    ----------
    level_edges : Dict[int, List[Tuple[str, str, float]]]
        {层次: [(source_node_id, target_node_id, weight), ...]}
        层次 0 通常为空（仅保留原始句内边）。

    Methods
    -------
    build(entities, ...)
        从 entities DataFrame 构建 EdgeSchedule。
    get_edges_for_level(level)
        获取指定层次应注入的边列表。
    total_edges()
        返回所有层次的边总数。
    """

    level_edges: Dict[int, List[Tuple[str, str, float]]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        entities: pd.DataFrame,
        intra_para_weight: float = 2.0,
        intra_doc_weight: float = 1.5,
        cross_doc_weight: float = 1.0,
        max_cross_doc_edges: int = 5,
        include_cross_doc: bool = True,
    ) -> "EdgeSchedule":
        """
        从 entities DataFrame 构建 EdgeSchedule。

        Parameters
        ----------
        entities : pd.DataFrame
            必须包含列：id（node_id）, title
            优先列：para_id, doc_id, sent_id
        intra_para_weight : float
            同段落跨句子边的权重，默认 2.0
            v6 调整：原 0.3，但 delta_Q ≈ w/m，m≈6000 时 delta_Q≈0.00005，
            远小于 lambda * delta_H，需提高权重使约束生效。
        intra_doc_weight : float
            同文档跨段落边的权重，默认 1.5
        cross_doc_weight : float
            跨文档边的权重，默认 1.0
        max_cross_doc_edges : int
            每个实体最多连接的跨文档同名实体数量，默认 5
        include_cross_doc : bool
            是否包含跨文档边，默认 True
            设为 False 可用于消融实验（仅段落内+文档内）

        Returns
        -------
        EdgeSchedule
            构建好的分层边调度器
        """
        schedule: Dict[int, List[Tuple[str, str, float]]] = {}

        # v6 修复：所有同名实体边在 Level 0 一次性注入
        # ---------------------------------------------------------------
        # 根因：原始图是句内断裂森林（node_id 含 sent_id，所有边限句内），
        # 导致每个连通分量内节点共享同一 para_id → 熵恒为 0 → lambda 失效。
        #
        # 关键发现：仅注入同段落跨句子边（133条）不够——因为同段落节点
        # 的 para_id 相同，即使合并也不产生熵。必须在 Level 0 就注入
        # 跨段落（甚至跨文档）的同名实体边，让连通分量跨越多个段落，
        # 这样 Leiden 合并时才会产生跨段落的社区 → 非零熵 → lambda 生效。
        #
        # 新策略：Level 0 注入所有三类边（同段落+跨段落+跨文档）；
        # 权重差异化保证近距离合并优先；lambda 约束控制合并边界。
        # ---------------------------------------------------------------

        # Level 0: 一次性注入所有同名实体边（打破断裂森林）
        all_l0_edges: List[Tuple[str, str, float]] = []
        all_l0_edges.extend(
            _build_intra_paragraph_edges(entities, weight=intra_para_weight)
        )
        all_l0_edges.extend(
            _build_intra_document_edges(entities, weight=intra_doc_weight)
        )
        if include_cross_doc:
            all_l0_edges.extend(
                _build_cross_document_edges(
                    entities,
                    weight=cross_doc_weight,
                    max_edges_per_entity=max_cross_doc_edges,
                )
            )
        schedule[0] = all_l0_edges

        # Level 1+: 无额外边（所有边已在 Level 0 注入）
        schedule[1] = []
        schedule[2] = []

        return cls(level_edges=schedule)

    def get_edges_for_level(self, level: int) -> List[Tuple[str, str, float]]:
        """
        获取指定层次应注入的边列表。

        注意：返回的是该层次"新增"的边，不是累积边。
        Leiden 主循环负责将边累积到图中（已有边则累加权重）。

        Parameters
        ----------
        level : int
            当前层次（从 0 开始）

        Returns
        -------
        List[Tuple[str, str, float]]
            [(source_node_id, target_node_id, weight), ...]
            若该层次无预设边，返回空列表。
        """
        return self.level_edges.get(level, [])

    def total_edges(self) -> int:
        """返回所有层次的边总数（用于调试和统计）。"""
        return sum(len(edges) for edges in self.level_edges.values())

    def summary(self) -> str:
        """返回各层次边数量的摘要字符串（用于日志输出）。"""
        lines = ["EdgeSchedule summary:"]
        for level in sorted(self.level_edges.keys()):
            n = len(self.level_edges[level])
            lines.append(f"  Level {level}: {n} edges")
        lines.append(f"  Total: {self.total_edges()} edges")
        return "\n".join(lines)
