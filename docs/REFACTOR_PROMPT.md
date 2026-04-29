# GraphRAG Improved v5 重构指引

> 本文档是一份面向大模型的结构化提示词，包含项目的完整问题诊断和改进方案。  
> 编写时间：2026-04-27（v5 代码已完成，实验待运行）  
> **当前状态**：文档中描述的所有代码修改（第四节-第六节）已全部完成。本文档现在的主要用途是帮助理解 v5 的设计思路和代码变更，以及在后续实验阶段指导参数调优。

---

## 一、项目背景与设计意图

### 1.1 项目是什么

GraphRAG Improved 是对微软 GraphRAG 的改进实现，核心创新是在 Leiden 社区发现算法中引入 **结构熵约束** 和 **λ 退火机制**，使社区合并过程受物理来源信息（文本在文档中的位置）的约束。

### 1.2 作者的原始设想（尚未实现）

社区合并应呈现 **渐进式的空间扩展**，由 λ 退火控制节奏：

```
层次 0（λ 最大）：同一句子内的实体优先合并为基础社区
层次 1（λ 较大）：同一段落内、不同句子的社区开始合并
层次 2（λ 中等）：同一文档内、不同段落的社区合并
层次 3+（λ 趋近 0）：跨文档的社区合并，结构熵约束逐步消失
最终：λ → 0 后，退化为标准 Leiden，完成高层社区的自由构建
```

核心思想：**物理距离近的实体应该先合并，物理距离远的实体应该后合并**，λ 是控制这一节奏的旋钮。

### 1.3 核心目标函数

```
J = Q_leiden − λ · H_structure
```

其中 Q 是模块度增益，H 是社区内节点物理来源分散程度的 Shannon 熵。λ 大时 H 的惩罚强，算法倾向于把物理来源相近的节点放在一起；λ 小时惩罚弱，允许跨来源合并。

---

## 二、代码架构（v5，已完成修改）

### 2.1 项目结构

```
graphrag_improved/
├── constrained_leiden/
│   ├── leiden_constrained.py    # 核心：结构熵约束 Leiden（v5: 移除终止条件，接入 EdgeSchedule）
│   ├── physical_anchor.py       # PhysicalNode 定义 + 结构熵计算
│   ├── graphrag_workflow.py     # 物理锚点构建（v5: anchor_granularity 参数）+ Path A
│   ├── annealing.py             # λ 退火调度器
│   └── edge_scheduler.py        # v5 新增：分层加边调度器（约 430 行）
├── extraction/
│   └── extractor.py             # spaCy 依存句法三元组抽取（v5: 新增 doc_id/para_id）
├── retrieval/
│   └── retriever.py             # BottomUp / TopDown 检索器
├── experiments/
│   └── run_multihop_eval.py     # 六组消融实验（v5 扩展，支持 --groups）
└── config.yaml                  # 默认配置（v5: 新增 anchor_granularity 等参数）
```

### 2.2 数据流概览

```
原始文档
  → extraction/extractor.py: 按句子切分，spaCy 抽取实体和关系
    → 实体 node_id = "{sent_id}-{entity_name_normalized}"
    → 物理结构边 co_occurs（weight=1.0）仅在同句内实体之间创建
  → constrained_leiden/graphrag_workflow.py: 构建 PhysicalNode
    → 每个节点的 chunk_ids = frozenset([sent_id])，即单一句子 ID
  → constrained_leiden/leiden_constrained.py: 层次化约束 Leiden
    → 每层获取 λ(level)，执行 local_moving + refinement + aggregation
    → local_moving 中：delta_j = delta_q - lambda_val * delta_h
  → retrieval/retriever.py: 基于社区层次的检索
```

---

## 三、问题诊断（三个根本缺陷）

### 问题 1：图是断裂的"句子森林"，Leiden 无法跨句合并

**现状**：`extraction/extractor.py` 中，物理结构边 `co_occurs` 严格限定在同句内：

```python
# extractor.py L301-L334
def _extract_physical_structure_edges(entities):
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if e1.sent_id != e2.sent_id:
                continue  # ← 不同句子的实体之间没有边
            edges.append(Relation(predicate="co_occurs", weight=1.0, ...))
```

**后果**：初始图是 N 个互不相连的句子级子图。Leiden 算法只能在连通分量内部做社区发现，永远无法把来自不同句子的节点合并到同一社区——因为它们之间没有路径。

**与设想的差距**：作者希望"同段落的句子逐渐合并"，但图中根本没有跨句子的边供 Leiden 发现社区结构。

### 问题 2：物理锚点粒度太细（sent_id 唯一），H_structure ≡ 0

**现状**：`graphrag_workflow.py` 中，每个节点的物理锚点是其唯一的 sent_id：

```python
# graphrag_workflow.py L182-L243
physical_nodes[node_id] = PhysicalNode(
    node_id=node_id,
    chunk_ids=frozenset([anchor_id]),  # anchor_id = sent_id，每个节点独一无二
    level=0,
)
```

**后果**：当考虑把节点 v 移入候选社区 C 时，计算 δH = H(C ∪ {v}) − H(C)。由于 v 的 chunk_ids 是一个全局唯一的 sent_id，无论 C 是同段落的社区还是跨文档的社区，v 带入的都是"一个全新的、从未出现过的 chunk_id"——δH 对所有候选社区相等。

这意味着在比较不同候选社区时，`lambda_val * delta_h` 项完全抵消，节点去向 **完全由 delta_q（模块度增益）单独决定**。λ 在节点分配决策中没有任何实际影响力。

**数学证明**：设社区 C 有 n 个节点，每个节点 chunk_ids 互不相同。v 的 chunk_id 也与所有已有 chunk_id 不同。则：
- H(C) = −∑(1/n · log(1/n)) = log(n)
- H(C ∪ {v}) = log(n+1)
- δH = log(n+1) − log(n) = log((n+1)/n)

这个 δH **只取决于社区的当前大小 n**，与社区内节点的物理来源无关。对于大小相同的任意两个候选社区，δH 完全相等。

### 问题 3：λ 唯一的实际效果是控制层级数量，而非控制合并节奏

**现状**：`leiden_constrained.py` 中的终止条件：

```python
# leiden_constrained.py L645-L646
if lambda_val < 1e-6:
    break
```

**后果**：
- λ=0（实验 [0]）：首层 `get_lambda(0, ...)` 返回 0.0 < 1e-6 → 只跑 1 层 → 产生约 29K 社区
- λ=1000（实验 [1]）：λ 指数衰减，需要多层才能降到 1e-6 以下 → 跑 3 层 → 产生约 79K 社区（含多层次）

v4 实验中 P@5 +21.4% 的提升，**实际原因是搜索空间不同**（3 层 vs 1 层），而非结构熵约束带来的社区质量提升。PROJECT_STATUS.md 中所有实验的 structural_entropy 均为 0.0000，印证了这一点。

---

## 四、改进方案

### 4.1 整体思路

要实现作者的渐进合并设想，需要同时满足两个条件：

**条件 A**：图中存在跨句/跨段/跨文档的边（否则 Leiden 没有路径可走）
**条件 B**：物理锚点的粒度足以区分"同段落"和"跨文档"（否则 H 无法提供差异化惩罚）

### 4.2 方案一：多粒度物理锚点（修改 chunk_ids 的语义）

**核心思想**：将 chunk_ids 从 sent_id 改为 **段落级 ID**（或文档级 ID），使同段落节点共享 chunk_id。

#### 4.2.1 修改 `extraction/extractor.py`

每个实体需要携带多粒度位置信息：

```python
# 当前：Entity 只有 sent_id
# 改为：Entity 携带 doc_id, para_id, sent_id 三级标识

@dataclass
class Entity:
    title: str
    entity_type: str
    description: str
    sent_id: str          # 已有，格式 "{doc_id}-p{para_idx:03d}-s{sent_idx:03d}"
    doc_id: str           # 新增：文档级标识（12 位 hex，md5(file_path)[:12]）
    para_id: str          # 新增：段落级标识，格式 "{doc_id}-p{para_idx:03d}"
```

sent_id 的格式已经是 `{doc_id}-p{para_idx:03d}-s{sent_idx:03d}`（其中 doc_id 是 12 位 hex 哈希），所以 doc_id 和 para_id 可以从 sent_id 中解析出来，不需要修改抽取逻辑，只需在 Entity 构造时补充解析。

#### 4.2.2 修改 `constrained_leiden/graphrag_workflow.py`

将 PhysicalNode 的 chunk_ids 从 sent_id 改为 para_id：

```python
# 当前：
physical_nodes[node_id] = PhysicalNode(
    node_id=node_id,
    chunk_ids=frozenset([sent_id]),      # 每个节点唯一
)

# 改为：
para_id = _extract_para_id(sent_id)      # 从 "a3f2b1c4d5e6-p002-s001" 提取 "a3f2b1c4d5e6-p002"
physical_nodes[node_id] = PhysicalNode(
    node_id=node_id,
    chunk_ids=frozenset([para_id]),       # 同段落的节点共享 chunk_id
)
```

**效果**：
- 同段落不同句子的节点共享同一个 para_id → 合并它们时 H 不增加
- 跨段落的节点有不同的 para_id → 合并它们时 H 增加 → λ·δH 产生惩罚
- 高 λ 时惩罚强 → 优先合并同段落节点；λ 衰减后 → 允许跨段落合并

如需更精细的控制，可以使用 **多级锚点**：每层的 chunk_ids 使用不同粒度：

```python
# level 0: chunk_ids = frozenset([para_id])    → 同段落优先
# level 1: chunk_ids = frozenset([doc_id])     → 同文档优先
# level 2+: chunk_ids 不再有区分度             → 自由合并
```

这需要在聚合阶段动态切换锚点粒度，详见 4.4 节。

### 4.3 方案二：分层加边（逐层扩展图的拓扑）

**核心思想**：不在初始图中一次性加入所有边，而是随层次递增逐步加入更远距离的边。

#### 4.3.1 修改 `constrained_leiden/leiden_constrained.py`

在 `hierarchical_leiden_constrained` 的主循环中，每层开始前根据当前层次向图中注入新边：

```python
# 伪代码
def hierarchical_leiden_constrained(..., edge_schedule=None):
    while True:
        lambda_val = get_lambda(level, annealing_config)

        # 新增：分层加边
        if edge_schedule is not None:
            new_edges = edge_schedule.get_edges_for_level(level)
            for u, v, w in new_edges:
                current_graph.add_edge(u, v, weight=w)

        state = _initialize_state(current_graph, current_physical)
        # ... 正常 Leiden 流程 ...
```

#### 4.3.2 新增 `constrained_leiden/edge_scheduler.py`

```python
@dataclass
class EdgeSchedule:
    """分层边调度器：控制每一层引入什么距离的边"""

    # level → edges 映射
    level_edges: Dict[int, List[Tuple[str, str, float]]]

    @classmethod
    def build(cls, entities: pd.DataFrame) -> "EdgeSchedule":
        schedule = {}

        # Level 0: 无额外边（仅保留原始 co_occurs 句内边）
        schedule[0] = []

        # Level 1: 同段落内、不同句子的同名实体边（weight=0.3）
        schedule[1] = _build_intra_paragraph_edges(entities, weight=0.3)

        # Level 2: 同文档内、跨段落的同名实体边（weight=0.2）
        schedule[2] = _build_intra_document_edges(entities, weight=0.2)

        # Level 3: 跨文档的同名实体边（weight=0.1）
        schedule[3] = _build_cross_document_edges(entities, weight=0.1)

        return cls(level_edges=schedule)
```

**权重设计原则**：距离越远的边权重越低，使 Leiden 的模块度增益 δQ 对远距离合并的驱动力更弱。配合 λ 退火的 δH 惩罚递减，形成双重控制。

### 4.4 方案三（推荐）：方案一 + 方案二的结合

单独使用方案一或方案二都有局限：

- 只用方案一（改锚点粒度）：如果图仍是断裂森林，δH 有区分度但节点无法跨句移动
- 只用方案二（分层加边）：如果锚点仍是 sent_id，δH 仍然对所有候选社区相等

**推荐方案：同时修改锚点粒度 + 分层加边**，形成完整的渐进合并机制。

具体实现分为五步：

#### 第一步：增强 Entity 的位置信息

文件：`extraction/extractor.py`

为 Entity 解析出 doc_id 和 para_id（从已有的 sent_id 中提取）。修改 Entity 的 `__post_init__` 或新增属性：

```python
@property
def doc_id(self) -> str:
    """从 sent_id 中提取文档 ID"""
    # sent_id 格式: "{doc_id}-p{NNN}-s{NNN}"（NNN 为零填充 3 位数字）
    parts = self.sent_id.rsplit("-p", 1)
    return parts[0] if len(parts) > 1 else self.sent_id

@property
def para_id(self) -> str:
    """从 sent_id 中提取段落 ID"""
    # sent_id 格式: "{doc_id}-p{NNN}-s{NNN}"（NNN 为零填充 3 位数字）
    parts = self.sent_id.rsplit("-s", 1)
    return parts[0] if len(parts) > 1 else self.sent_id
```

确保 entities DataFrame 在传递给 graphrag_workflow 时包含 doc_id 和 para_id 列。

#### 第二步：修改物理锚点粒度

文件：`constrained_leiden/graphrag_workflow.py`

将 `build_physical_nodes_from_graphrag` 中的 chunk_ids 从 sent_id 改为 para_id：

```python
def build_physical_nodes_from_graphrag(entities: pd.DataFrame) -> Dict[str, PhysicalNode]:
    physical_nodes = {}
    for _, row in entities.iterrows():
        node_id = str(row["id"])

        # 改为段落级锚点
        para_id = str(row.get("para_id", "")).strip()
        if not para_id:
            # 从 sent_id 解析
            sent_id = str(row.get("sent_id", ""))
            para_id = sent_id.rsplit("-s", 1)[0] if "-s" in sent_id else sent_id

        physical_nodes[node_id] = PhysicalNode(
            node_id=node_id,
            chunk_ids=frozenset([para_id]),  # ← 段落级
            level=0,
        )
    return physical_nodes
```

#### 第三步：实现分层加边

新建文件：`constrained_leiden/edge_scheduler.py`

实现 `EdgeSchedule` 类，根据层次返回不同距离的边。需要三个辅助函数：

- `_build_intra_paragraph_edges(entities, weight)`：同段落、不同句子、同名实体之间的边
- `_build_intra_document_edges(entities, weight)`：同文档、不同段落、同名实体之间的边
- `_build_cross_document_edges(entities, weight)`：不同文档、同名实体之间的边

实体匹配逻辑：复用 `graphrag_workflow.py` 中 `build_intra_doc_entity_edges` 的思路（按 title_lower 分组），但按距离层次过滤。

#### 第四步：修改层次化 Leiden 主循环

文件：`constrained_leiden/leiden_constrained.py`

在 `hierarchical_leiden_constrained` 中接入 EdgeSchedule：

```python
def hierarchical_leiden_constrained(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
    annealing_config: Optional[AnnealingConfig] = None,
    edge_schedule: Optional[EdgeSchedule] = None,   # ← 新增参数
    ...
) -> HierarchicalCommunityResult:

    while True:
        lambda_val = get_lambda(level, annealing_config)

        # 新增：按层次注入边
        if edge_schedule is not None:
            for u, v, w in edge_schedule.get_edges_for_level(level):
                if current_graph.has_node(u) and current_graph.has_node(v):
                    current_graph.add_edge(u, v, weight=w)

        state = _initialize_state(current_graph, current_physical)
        # ... 原有逻辑不变 ...
```

注意：聚合阶段产生超级节点后，后续层次的节点 ID 已变为 `super_xxx_lN` 格式。EdgeSchedule 中的边使用的是原始节点 ID，所以注入边时需要将原始 ID 映射为当前层的超级节点 ID。需要维护一个 `original_to_current` 映射表。

#### 第五步：修改终止条件语义

文件：`constrained_leiden/leiden_constrained.py`

当前终止条件 `if lambda_val < 1e-6: break` 的语义是"λ 太小就停止"。在新方案中，λ → 0 应该意味着"结构熵约束消失，进入自由 Leiden 模式"，而非"停止迭代"：

```python
# 当前（有问题）：
if lambda_val < 1e-6:
    break  # ← λ 小了就停止，导致层数受 λ 控制

# 改为：
# 不再以 λ 作为终止条件
# 只保留收敛性终止条件：
if num_communities <= 1 or num_communities == len(current_graph.nodes()):
    break
if level > max_level_cap:
    break
```

这样 λ 的衰减仅影响 δH 的惩罚力度，不影响层数。层数由图的收敛性自然决定。

### 4.5 高级选项：动态锚点粒度切换

如果需要更精细的控制，可以在聚合阶段动态切换物理锚点的粒度：

```python
def _aggregation_phase(graph, state, level):
    for comm_id, comm_nodes in state.community_to_nodes.items():
        super_node_id = f"super_{comm_id}_l{level}"

        child_physical_nodes = [state.physical_nodes[n] for n in comm_nodes if n in state.physical_nodes]

        # 动态粒度：低层用 para_id，高层用 doc_id
        if level < 2:
            # 保持 para_id 粒度
            super_pnode = PhysicalNode.merge(super_node_id, child_physical_nodes, level + 1)
        else:
            # 切换为 doc_id 粒度：将 chunk_ids 中的 para_id 映射为 doc_id
            doc_ids = set()
            for pn in child_physical_nodes:
                for chunk_id in pn.chunk_ids:
                    doc_id = chunk_id.rsplit("-p", 1)[0]  # "a3f2b1c4d5e6-p002" → "a3f2b1c4d5e6"
                    doc_ids.add(doc_id)
            super_pnode = PhysicalNode(
                node_id=super_node_id,
                chunk_ids=frozenset(doc_ids),
                level=level + 1,
            )
```

这实现了：
- Level 0-1：结构熵基于段落粒度 → 惩罚跨段落合并
- Level 2+：结构熵基于文档粒度 → 惩罚跨文档合并
- λ → 0 后：惩罚消失 → 自由合并

---

## 五、实验设计建议

### 5.1 更新消融实验

六组消融实验已在 `run_multihop_eval.py` 中实现为 RunConfig：

```
[0] Baseline          : λ=0, 无分层加边, sent_id 锚点（复现 v4 的 [0]）
[1] 仅改锚点          : λ=1000, 无分层加边, para_id 锚点
[2] 仅分层加边        : λ=1000, 分层加边, sent_id 锚点
[3] 完整方案           : λ=1000, 分层加边, para_id 锚点（v5 核心方案）
[4] +Path A           : λ=1000, 分层加边, para_id 锚点, 文档内消解边
[5] +跨文档边          : [4] + EdgeSchedule 包含跨文档同名实体边
```

### 5.2 验证指标

除 P@5 外，增加以下验证项来确认渐进合并是否生效：

- **结构熵非零验证**：使用 para_id 锚点后，每层的平均社区 H_structure 应 > 0（当前全为 0.0000）
- **层间社区组成分析**：统计每层社区中 "同段落节点占比" 和 "跨文档节点占比"，验证低层同段落占比高、高层跨文档占比高
- **合并顺序追踪**：记录每次 `_move_node` 时节点的物理距离（same_para / same_doc / cross_doc），绘制随层次变化的曲线

### 5.3 公平性修复

当前 [0]（1 层）和 [1]（3 层）的对比不公平（搜索空间不同）。修复终止条件后，所有组别的层数应由图收敛性自然决定，不再被 λ 的阈值截断。这样不同 λ 值的实验在相同层数下对比，才能反映结构熵约束的真实效果。

---

## 六、需要修改的文件清单

| 优先级 | 文件 | 修改内容 |
|--------|------|----------|
| P0 | `extraction/extractor.py` | Entity 增加 doc_id / para_id 属性（从 sent_id 解析） |
| P0 | `constrained_leiden/graphrag_workflow.py` | chunk_ids 从 sent_id 改为 para_id |
| P0 | `constrained_leiden/leiden_constrained.py` | 移除 `lambda_val < 1e-6` 终止条件；接入 EdgeSchedule |
| P1 | `constrained_leiden/edge_scheduler.py`（新建） | 实现分层加边逻辑 |
| P1 | `constrained_leiden/graphrag_workflow.py` | 新增 `build_intra_paragraph_edges`、`build_cross_document_edges` |
| P2 | `constrained_leiden/leiden_constrained.py` | 聚合阶段可选动态锚点粒度切换 |
| P2 | `experiments/run_multihop_eval.py` | 更新消融实验分组 |
| P3 | `README.md` | 更新实验结果（当前是过时的 v1 数据） |
| P3 | `PROJECT_STATUS.md` | 记录 v5 改进 |

---

## 七、关键约束与注意事项

1. **不要破坏现有可运行的代码**：所有修改应向后兼容，可通过配置开关（如 `anchor_granularity: "para" | "sent" | "doc"`）控制新旧行为。

2. **sent_id 格式假设**：当前 sent_id 格式为 `{doc_id}-p{NNN}-s{NNN}`（NNN 为零填充 3 位数字，doc_id 为 12 位 hex 哈希 `md5(file_path)[:12]`），解析 para_id 和 doc_id 时依赖这一格式。需在代码中做格式校验和回退处理。

3. **边的节点 ID 映射**：分层加边在 level > 0 时，图中的节点已经是超级节点 `super_xxx_lN`。注入边时需要将原始节点 ID 映射到当前层的超级节点 ID，否则边会指向不存在的节点。

4. **性能考量**：跨文档同名实体边的数量可能很大（O(n²) 同名实体对）。应采用链式连接（当前 Path A 已使用）或 KNN 限制最大边数。

5. **现有 Path A 的关系**：当前 `build_intra_doc_entity_edges` 实现的 Path A（同文档同名实体 soft edge）与新方案的"分层加边"有重叠。建议将 Path A 重构为 EdgeSchedule 的一部分，统一管理所有非句内边。

6. **config.yaml 中 `use_lcc: true`**：当前实验使用 `use_lcc=False`。修改后的实验应明确记录 use_lcc 设置。

7. **结构熵计算的正确性**：修改锚点粒度后，需验证 `physical_anchor.py` 中 `compute_structural_entropy` 和 `CommunityEntropyState` 的增量计算逻辑在新粒度下仍然正确。特别是当多个节点共享同一个 chunk_id 时，增量更新的数学推导需要复查。

---

## 八、预期效果

修改完成后，运行层次化 Leiden 应该能观察到：

1. **H_structure > 0**：社区内有不同 para_id 的节点时，熵非零
2. **低层社区局部性强**：level 0 的社区主要由同段落节点组成
3. **高层社区逐渐跨越文档**：level 2+ 的社区开始包含不同文档的节点
4. **λ 对节点分配有真实影响**：相同图结构下，λ=1000 和 λ=0 应产生不同的社区组成（而非仅层数不同）
5. **评估公平**：所有实验组的层数由收敛性决定，不再被 λ 阈值截断

## 九、验证清单

完成所有修改后，按以下清单逐项验证：

代码修改已完成（✅），实验验证待运行（⏳）：

- [x] `PhysicalNode.chunk_ids` 中存储的是 `para_id`（格式 `{doc_id}-p{NNN}`），而非 `sent_id` — ✅ 已修改
- [x] 同段落内不同句子的实体共享相同的 `chunk_ids` 值 — ✅ 已修改
- [x] 终止条件不再依赖 `lambda_val < 1e-6`，而是依赖收敛性判断 — ✅ 已修改
- [x] EdgeSchedule 实现分层加边（Level 1/2/3） — ✅ 已实现（edge_scheduler.py 约 430 行）
- [ ] 仅有句内 co_occurs 边的图上，level 0 的 Leiden 只在句内合并 — ⏳ 待实验验证
- [ ] level 1 加入同段落跨句边后，社区开始跨句合并 — ⏳ 待实验验证
- [ ] level 2 加入同文档跨段落边后，社区开始跨段落合并 — ⏳ 待实验验证
- [ ] `compute_structural_entropy` 对含有不同 `para_id` 的社区返回 > 0 — ⏳ 待实验验证
- [ ] `lambda_val > 0` 时，δH 对不同候选社区产生不同的值 — ⏳ 待实验验证
- [ ] 六组实验在相同终止条件下产出相同层数，差异体现在社区组成而非层数 — ⏳ 待实验验证
- [ ] P@5 等指标的提升来自社区质量差异，而非搜索空间大小差异 — ⏳ 待实验验证
