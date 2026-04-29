# GraphRAG-Improved：LLM 参考文档

> **用途**：这是一份专为 LLM（大语言模型）上下文窗口优化的单一权威参考文档。所有数字、API 签名、数据结构均直接从代码提取，不与其他文档重复。当其他文档与本文档冲突时，以本文档为准。
>
> **最后同步代码**：2026-04-27
>
> **项目一句话描述**：在 GraphRAG 的社区检测阶段注入结构熵惩罚（J = Q_leiden − λ·H_structure），使同一来源文档的实体倾向于聚入同一社区，从而提升多跳问答的检索召回率。

---

## §1 核心公式与概念（7 个，无歧义定义）

**1.1 目标函数**

```
J = Q_leiden − λ · H_structure
```

- `Q_leiden`：标准 Leiden 算法的模块度增量（ΔQ）
- `H_structure`：社区内的来源分布熵，公式为 `H = −Σ p_i · log(p_i)`，其中 `p_i = weight_of_chunk_i / total_weight_in_community`
- `λ`：惩罚强度，由退火策略控制

**1.2 物理优先架构（v3 起）**

每个节点是一个 **实例级实体**，不做跨句合并。

- 节点 ID 格式：`{sent_id}-{entity_title_normalized}`
- sent_id 格式：`{doc_id}-p{para_idx:03d}-s{sent_idx:03d}`
- para_id 格式：`{doc_id}-p{para_idx:03d}`
- doc_id 格式：12 位 hex（源文件路径的 MD5）

**1.3 结构熵的"锚点粒度"**

`anchor_granularity` 参数决定 `PhysicalNode.chunk_ids` 的粒度：

| 值 | chunk_id 取值 | 语义 | 版本 |
|---|---|---|---|
| `"sent"` | sent_id | 每个节点唯一一个 chunk → 社区内 H ≡ 0（已验证失效） | v3/v4 |
| `"para"` | para_id | 同段落不同句子共享 chunk → H > 0 | v5 |
| `"doc"` | doc_id | 同文档所有节点共享 chunk（未测试） | 预留 |

**1.4 λ 退火**

```python
# 指数衰减（默认）
λ(level) = max(λ_init · exp(−decay_rate · level), λ_min)
```

四种曲线可选：`exponential`（默认）、`linear`、`cosine`、`step`。

**1.5 EdgeSchedule 分层加边（v5 新增）**

层次化社区检测过程中，每经过一层聚合，注入更远距离的同名实体边：

| 注入层级 | 边类型 | 默认权重 | 含义 |
|---|---|---|---|
| Level 0 | 无新增边 | — | 仅保留原始句内共现边 |
| Level 1 | intra_paragraph | 0.3 | 同段落跨句子的同名实体 |
| Level 2 | intra_document | 0.2 | 同文档跨段落的同名实体 |
| Level 3 | cross_document | 0.1 | 跨文档的同名实体（可关闭） |

**1.6 Path A（文档内链式合并边）**

在图构建阶段，为同文档内的同名实体按 sent_id 排序后添加链式软连接边（A→B→C），权重默认 0.5。对应参数 `intra_doc_merging=True`。

**1.7 Hierarchical Leiden 终止条件（v5）**

以下任一条件满足即停止：

1. 社区数 ≤ 1
2. 社区数 = 节点数（无合并发生，收敛）
3. level > max_level × 2（安全上限）

v5 已移除 v4 的 `lambda_val < 1e-6` 终止条件（该条件曾导致 λ 衰减后提前终止，变相让 λ 控制层数而非合并节奏）。

---

## §2 数据流水线（从原文到评估的完整路径）

```
raw_texts/
  ├── *.txt (609 篇新闻文章)
  │
  ▼ [ingestion.py] load_documents → documents_to_text_units
  │   - 按段落切分 → TextUnit (chunk_id = para_id)
  │   - 每段用 spaCy 切句 → SentenceUnit (sent_id)
  │
  ▼ [extractor.py] extract(text_units) → (entities_df, relationships_df)
  │   - 基于 spaCy NER + 句内共现窗口
  │   - Entity 含 sent_id, para_id, doc_id
  │   - Relation 含 source_node_id, target_node_id, weight
  │
  ▼ [graphrag_workflow.py] run_constrained_community_detection(entities_df, relationships_df, ...)
  │   ├── build_graph_from_graphrag → nx.Graph (无向, 带 weight)
  │   ├── build_physical_nodes_from_graphrag(anchor_granularity="para") → Dict[node_id, PhysicalNode]
  │   ├── EdgeSchedule.build(entities_df, ...) → EdgeSchedule (如启用)
  │   └── hierarchical_leiden_constrained(graph, physical_nodes, annealing_config, edge_schedule, ...)
  │       ├── 每层: _local_moving_phase(ΔJ = ΔQ − λ·ΔH) → _refinement_phase → _aggregation_phase
  │       ├── 每层结束后: edge_schedule.get_edges_for_level(level) → 注入新边
  │       └── → HierarchicalCommunityResult
  │
  ▼ [graphrag_workflow.py] convert_result_to_communities_df → communities_df (pd.DataFrame)
  │
  ▼ [retriever.py] URetriever(communities_df, text_units, entities_df)
  │   ├── TopDownRetriever: 按社区层级 TF-IDF 匹配 query
  │   └── BottomUpRetriever: 按文本块 TF-IDF 匹配 query（⚠️ 存在锚点 bug，见 §5）
  │
  ▼ [evaluator.py] Evaluator.evaluate_retrieval(qa_pairs, retriever) → RetrievalMetrics
      - 指标: MRR, P@K, Recall@K, F1@K, NDCG@K
```

---

## §3 权威实验数据（唯一数字来源）

### 3.1 数据集说明

- **来源**：MultiHop-RAG (COLM 2024)，609 篇新闻文章，2556 条 QA
- **评估子集**：随机采样 200 条（`--n-qa 200`），剔除无法索引的查询后有效 169 条
- **实际使用数据**：v4 消融实验四组全部使用同一份抽取结果——47,142 实体 / 61,812 关系（来源文件 `multihop_results_n200.json` 中四组 `num_entities` / `num_relationships` 一致）
- **历史说明**：项目早期存在两套抽取缓存（`*_full.parquet` 约 60,439 实体和 `*_full_b.parquet` 约 47,142 实体，分别对应原始抽取和噪声过滤后版本），但实际跑通的实验结果均基于后者

### 3.2 v4 消融实验（唯一已跑完的实验，来源文件 `experiments/results/multihop_results_n200.json`）

| 指标 | [0] Baseline λ=0 | [1] Ours λ=1000 | [2] +Path A | [3] +Path A+B |
|---|---|---|---|---|
| MRR | 0.3492 | 0.3552 | 0.3552 | 0.3552 |
| P@1 | 0.2663 | 0.2663 | 0.2663 | 0.2663 |
| P@5 | 0.1325 | 0.1633 | 0.1633 | 0.1633 |
| Recall@5 | 0.2648 | 0.3230 | 0.3230 | 0.3230 |
| NDCG@5 | 0.2347 | 0.2680 | 0.2680 | 0.2680 |
| NDCG@10 | 0.2737 | 0.2792 | 0.2792 | 0.2792 |
| 结构熵 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 社区数 | 22,973 | 63,497 | 63,497 | 63,497 |
| 层数 | 1 | 3 | 3 | 3 |
| 实体数 | 47,142 | 47,142 | 47,142 | 47,142 |
| 关系数 | 61,812 | 61,812 | 61,812 | 61,812 |
| 模块度 Q | 0.8025 | 0.8025 | 0.8025 | 0.8025 |
| 运行时间 | 322.8s | 285.6s | — | — |

### 3.3 Naive RAG Baseline（来源文件 `baselines/eval_results/n200/naive_rag_eval_results.json`）

| MRR | P@5 | Recall@5 | NDCG@5 | NDCG@10 | 有效查询数 |
|---|---|---|---|---|---|
| 0.6389 | 0.2568 | 0.5059 | 0.4792 | 0.5216 | 169 |

### 3.4 GraphRAG Official Baseline（来源文件 `baselines/eval_results/graphrag_full/graphrag_local_eval_results.json`）

| MRR | P@5 | Recall@5 | NDCG@5 | 有效查询数 |
|---|---|---|---|---|
| 0.8784 | 0.3243 | 0.7252 | 0.7486 | 37 |

⚠️ 仅 37 条有效查询（总共 338 条），与其他方法的 169 条有效查询不可比。

### 3.5 差距总结

与 Naive RAG 在相同 169 条查询上对比，Our Method 最佳组 [1]：MRR 落后 44%，P@5 落后 36%，Recall@5 落后 36%。**结构熵惩罚项在 v4 中完全失效（H ≡ 0），改善仅来自 λ > 0 时的层次结构增加。**

---

## §4 API 签名速查（精确到类型注解和默认值）

### 4.1 数据结构

```python
# physical_anchor.py
@dataclass
class PhysicalNode:
    node_id: str                       # "{sent_id}-{entity_title}"
    chunk_ids: FrozenSet[str]          # 锚点集合，粒度由 anchor_granularity 决定
    level: int = 0

# annealing.py
@dataclass
class AnnealingConfig:
    lambda_init: float = 1000.0
    lambda_min: float = 0.0
    max_level: int = 10
    decay_rate: float = 0.5
    schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL  # "exponential"|"linear"|"cosine"|"step"

# edge_scheduler.py
@dataclass
class EdgeSchedule:
    level_edges: Dict[int, List[Tuple[str, str, float]]]  # level → [(src, dst, weight), ...]

# extraction/extractor.py
@dataclass
class Entity:
    title: str
    entity_type: str
    sent_id: str            # "{doc_id}-p{NNN}-s{NNN}"
    para_id: str            # "{doc_id}-p{NNN}"
    doc_id: str             # 12-hex
    description: str = ""
    # property node_id → "{sent_id}-{title_normalized}"

@dataclass
class Relation:
    source_node_id: str
    target_node_id: str
    predicate: str
    weight: float = 1.0
    description: str = ""
    sent_id: str = ""

# data/ingestion.py
@dataclass
class SentenceUnit:
    sent_id: str            # "{doc_id}-p{para_idx:03d}-s{sent_idx:03d}"
    text: str
    doc_id: str
    para_id: str            # "{doc_id}-p{para_idx:03d}"
    sent_index: int
    para_index: int

@dataclass
class TextUnit:
    chunk_id: str           # = para_id（段落级）
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    sentences: List[SentenceUnit] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

# leiden_constrained.py
@dataclass
class HierarchicalCommunityResult:
    levels: List[Dict[str, int]]               # 每层 node → community_id 映射
    level_entropy: List[Dict[int, float]]       # 每层 community_id → H
    level_lambda: List[float]                   # 每层使用的 λ 值
    node_physical_map: Dict[str, PhysicalNode]

# retriever.py
@dataclass
class RetrievalResult:
    query: str
    top_down_hits: List[CommunityHit]
    bottom_up_hits: List[TextUnitHit]
    merged_context: str
    metadata: Dict

# evaluator.py
@dataclass
class RetrievalMetrics:
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    num_queries: int
```

### 4.2 核心函数

```python
# === 社区检测入口 ===
# graphrag_workflow.py L617
def run_constrained_community_detection(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    annealing_config: Optional[AnnealingConfig] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
    use_lcc: bool = True,
    min_edge_weight: float = 1.0,
    intra_doc_merging: bool = False,          # Path A 开关
    intra_doc_edge_weight: float = 0.5,       # Path A 边权
    anchor_granularity: AnchorGranularity = "para",  # Literal["sent","para","doc"]
    edge_schedule: Optional[EdgeSchedule] = None,     # ← 注意：对象，不是 bool
) -> pd.DataFrame

# === 层次化 Leiden ===
# leiden_constrained.py L580
def hierarchical_leiden_constrained(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
    annealing_config: Optional[AnnealingConfig] = None,
    edge_schedule: Optional[EdgeSchedule] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
) -> HierarchicalCommunityResult

# === EdgeSchedule 构建 ===
# edge_scheduler.py L340
@classmethod
EdgeSchedule.build(
    cls,
    entities: pd.DataFrame,
    intra_para_weight: float = 0.3,
    intra_doc_weight: float = 0.2,
    cross_doc_weight: float = 0.1,
    max_cross_doc_edges: int = 5,
    include_cross_doc: bool = True,
) -> EdgeSchedule

# === 检索器 ===
# retriever.py L497
class URetriever:
    def __init__(
        self,
        communities_df: pd.DataFrame,
        text_units: List[dict],
        entities_df: pd.DataFrame,
        top_k_communities: int = 5,
        top_k_chunks: int = 5,
        max_context_chars: int = 4000,
    ): ...

    def retrieve(
        self,
        query: str,
        entity_mentions: Optional[List[str]] = None,
        alpha: float = 0.5,       # top_down 与 bottom_up 的混合权重
    ) -> RetrievalResult

# === 实验运行 ===
# experiments/run_multihop_eval.py L298
# 注意：RunConfig.use_edge_schedule 是 bool，在 run_one() 中转换：
#   if cfg.use_edge_schedule:
#       edge_schedule = EdgeSchedule.build(entities_df, include_cross_doc=cfg.edge_schedule_cross_doc)
#   传给 run_constrained_community_detection(edge_schedule=edge_schedule)
```

---

## §5 已知缺陷与 v5 修复方案

### 5.1 v4 三大缺陷（已诊断，v5 代码已修复，实验未跑）

| # | 缺陷 | 根因 | 影响 | v5 修复 |
|---|---|---|---|---|
| D1 | 结构熵 H ≡ 0 | `anchor_granularity="sent"` → 每节点一个唯一 chunk_id → 社区内无分布可言 | λ·H 恒为 0，结构熵惩罚完全不工作 | 改为 `"para"` → 同段落多节点共享 para_id → H > 0 |
| D2 | 底层图为断裂的句子森林 | 节点是实例级（sentence-scoped），仅有句内共现边 → ~24,858 个连通分量 | Leiden 无法跨句合并，层次结构意义有限 | EdgeSchedule 分 3 级注入跨句/跨段/跨文档边 |
| D3 | λ 控制层数而非合并节奏 | `lambda_val < 1e-6: break` 终止条件 → λ 衰减到阈值以下就停止 | λ 值直接决定层次深度，而非调节合并倾向 | 移除该终止条件，改用收敛判定 |

### 5.2 检索模块潜在 Bug（未修复）

**BottomUpRetriever 锚点加权不生效**：

- `_entity_chunks` 字典存储 `entity_title → Set[sent_id]`（retriever.py L370-L379）
- 但 `text_units` 中每个 unit 的 `chunk_id` = `para_id`（ingestion.py TextUnit.chunk_id）
- 锚点匹配时 `chunk_id in anchor_chunk_ids`（L438），sent_id 格式 `xxx-p001-s002` 永远不等于 para_id 格式 `xxx-p001`
- **结果**：锚点命中的 ×1.5 加权从不触发，BottomUp 退化为纯 TF-IDF

**影响评估**：此 bug 在 v4 实验中已存在，修复后可能改善 bottom-up 检索质量。建议在 v5 实验前修复。

---

## §6 v5 六组消融实验设计（代码已就绪，尚未运行）

来源：`experiments/run_multihop_eval.py` L493-L542

| 组号 | 名称 | λ_init | anchor | edge_schedule | cross_doc | intra_doc_merge |
|---|---|---|---|---|---|---|
| [0] | Baseline | 0 | sent | ✗ | — | ✗ |
| [1] | 仅改锚点 | 1000 | **para** | ✗ | — | ✗ |
| [2] | 仅分层加边 | 1000 | sent | **✓** | ✗ | ✗ |
| [3] | 锚点+分层加边 | 1000 | **para** | **✓** | ✗ | ✗ |
| [4] | 完整方案+Path A | 1000 | **para** | **✓** | ✗ | **✓** (w=0.5) |
| [5] | 完整方案+跨文档 | 1000 | **para** | **✓** | **✓** | **✓** (w=0.5) |

**消融逻辑**：[0]→[1] 验证锚点粒度影响；[0]→[2] 验证边注入影响；[3] 验证两者叠加；[4] 增加文档内 Path A；[5] 增加跨文档边。

**运行命令**：

```bash
cd /Users/ttung/Desktop/个人学习/graphrag_improved
python -m experiments.run_multihop_eval --n-qa 200 --use-spacy --groups 0,1,2,3,4,5
```

---

## §7 目录结构速查

```
graphrag_improved/
├── constrained_leiden/
│   ├── graphrag_workflow.py      # 主入口 run_constrained_community_detection
│   ├── leiden_constrained.py     # hierarchical_leiden_constrained + 移动/聚合逻辑
│   ├── edge_scheduler.py         # EdgeSchedule 三级边注入
│   ├── physical_anchor.py        # PhysicalNode + 结构熵计算
│   └── annealing.py              # AnnealingConfig + 4 种退火曲线
├── extraction/
│   └── extractor.py              # Entity/Relation 抽取（spaCy NER + 共现）
├── data/
│   └── ingestion.py              # Document → TextUnit → SentenceUnit
├── retrieval/
│   └── retriever.py              # URetriever = TopDown + BottomUp
├── evaluation/
│   └── evaluator.py              # Evaluator + RetrievalMetrics + CommunityMetrics
├── experiments/
│   ├── run_multihop_eval.py      # 消融实验主脚本（RunConfig + 6 组配置）
│   └── results/                  # 实验结果 JSON
├── baselines/
│   └── eval_results/             # Naive RAG / GraphRAG Official 结果
├── config.yaml                   # 默认配置（v5 参数已就绪）
├── README.md                     # 项目概述（面向人类阅读）
├── PROJECT_STATUS.md             # 详细实验数据与版本历史
├── PROJECT_PLAN.md               # 项目计划与路线图
├── CHANGELOG.md                  # 版本变更日志
└── REFACTOR_PROMPT.md            # 重构提示词（面向 LLM 辅助开发）
```

---

## §8 关键约定（消除歧义）

**8.1 抽取缓存与实验数据**

- 项目中存在两套抽取缓存：`*_full.parquet`（原始抽取，约 60,439 实体）和 `*_full_b.parquet`（噪声过滤后，47,142 实体 / 61,812 关系）
- **v4 消融实验四组全部使用 47,142 实体 / 61,812 关系**（来源 `multihop_results_n200.json` 中四组 `num_entities` 和 `num_relationships` 完全一致）
- 早期文档中声称组 [0][1][2] 使用 60,439 实体版本为历史错误，已在本文档中修正

**8.2 `use_edge_schedule` 的两种含义**

- 在 `RunConfig`（run_multihop_eval.py）中：`use_edge_schedule: bool`，是一个开关
- 在 `run_constrained_community_detection` 中：`edge_schedule: Optional[EdgeSchedule]`，是对象或 None
- 转换发生在 `run_one()` 函数中：`if cfg.use_edge_schedule: edge_schedule = EdgeSchedule.build(...)`

**8.3 P@5 改善百分比**

- **+23.2%**：(0.1633 − 0.1325) / 0.1325 = 23.2%，来源 `multihop_results_n200.json` 组 [0] vs [1]
- **+21.4%**：出现在 PROJECT_STATUS.md 的早期版本中，对应 P@5 = 0.1609，可能来自 60,439 实体版本的未保存实验
- **本文档统一使用 JSON 源文件数据**，即 P@5 改善 = +23.2%

**8.4 结构熵全零不是 bug（在 v4 中）**

v4 使用 `anchor_granularity="sent"`，每个 PhysicalNode 的 `chunk_ids` 只含一个唯一 sent_id。当社区内所有节点的 chunk_id 各不相同时，`H = −Σ (1/n)·log(1/n) = log(n)`…但实际上因为每个节点只贡献一个 chunk 且 weight=1，分布是均匀的，理论上 H = log(n) > 0。然而代码中 `CommunityEntropyState` 的实际表现为 `H ≈ 0`，这是因为 Level 0 社区极小（平均 2.06 个节点）且每个节点的 chunk_id 唯一——当社区只有 1 个节点时 H = 0，2 个节点时 `H = log(2) ≈ 0.693` 但被平均后数值很小，加上 Leiden 初始化每节点独立成社区后大多数社区为单节点，导致 `avg_structural_entropy ≈ 0.0000`（四位小数显示为零）。

**修正**：更精确地说，v4 的结构熵不是数学上恒为 0，而是因为物理架构使得绝大多数社区为单节点（H=0），少数多节点社区的 H 被均值稀释后在四位小数精度下显示为 0.0000。v5 的 `anchor_granularity="para"` 通过共享 para_id 使得即使单节点社区也可能有非零 H（当同段落有多个实体时），从根本上解决此问题。
