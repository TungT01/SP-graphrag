# GraphRAG-Improved：LLM 参考文档

> **用途**：这是一份专为 LLM（大语言模型）上下文窗口优化的单一权威参考文档。所有数字、API 签名、数据结构均直接从代码提取，不与其他文档重复。当其他文档与本文档冲突时，以本文档为准。
>
> **最后同步代码**：2026-05-11
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

**1.5 EdgeSchedule 同名实体边注入（v5 引入，v6 重新校准）**

在 Level 0 一次性注入三类同名实体边，打破句内孤岛森林：

| 边类型 | 匹配条件 | 权重(v6+) | 连接方式 |
|---|---|---|---|
| 段落内跨句子 | 同段落 + 不同句子 + 同名实体 | 2.0 | 链式 |
| 文档内跨段落 | 同文档 + 不同段落 + 同名实体 | 1.5 | 代表节点链式 |
| 跨文档 | 不同文档 + 同名实体 | 1.0 | 代表节点链式(上限5) |

> **v6 校准说明**：v5 原设计为分层注入（Level 1/2/3），权重 0.3/0.2/0.1。v6 发现必须在 Level 0 同时注入跨段落边才能产生非零熵，且原权重过小（δQ ≈ w/(2m) 被 λ·δH 压制），故提升至 2.0/1.5/1.0 并改为 Level 0 一次性注入。

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
  ▼ [summarizer.py] generate_community_summaries(communities_df, llm_config) → communities_df（+summary 列）
  │   - 只为 level >= min_level 的社区调用 LLM
  │   - 结果缓存到 summary_cache/*.json（key = MD5(实体列表+level)）
  │
  ▼ [retriever.py] URetriever(communities_df, text_units, entities_df, retrieval_mode=...)
  │   ├── TF-IDF 模式: TopDownRetriever + BottomUpRetriever（✅ anchor bug v8 修复）
  │   └── 向量模式: VectorTopDownRetriever + VectorBottomUpRetriever（all-MiniLM-L6-v2）
  │       retrieval_mode 支持: uretrieval / topdown_only / bottomup_only（及 vector_* 前缀版本）
  │
  ▼ [evaluator.py] Evaluator.evaluate_retrieval(qa_pairs, retriever) → RetrievalMetrics
  │   - 指标: MRR, P@K, Recall@K, F1@K, NDCG@K + Bootstrap 95%CI
  │
  ▼ [qa_evaluator.py] evaluate_qa_end_to_end(qa_pairs, retriever, llm_config) → QAMetrics
      - 检索 → LLM 生成答案 → EM / Token F1 / ROUGE-L
```

---

## §3 权威实验数据（索引）

> 完整实验数据见 `docs/EXPERIMENT_RESULTS.md`（最后更新 2026-05-11）。本节只列关键数字，避免重复。

### 3.1 数据集

- **MultiHop-RAG** (COLM 2024)，609 篇新闻文章，2556 条 QA
- **当前主要评估子集**：n=500 采样，429 条有效 QA（n=1000 采样时 881 条有效）
- 实体规模：n=500 → 26,289 实体 / 34,532 关系

### 3.2 核心结论数字（v11b，n=881，向量检索，有摘要）

| 组 | MRR | P@5 | EM | MRR 95%CI |
|---|---|---|---|---|
| A+VS 标准 Leiden（GraphRAG 复现） | 0.440 | 0.116 | 0.089 | [0.410, 0.471] |
| **B3+VS 约束 Leiden λ=0.003（本方法）** | **0.489** | **0.159** | **0.124** | **[0.458, 0.516]** |
| 提升 | +10.9% | +37.4% | +39.3% | CI 完全不重叠 |

来源文件：`experiments/results_v11b/multihop_results_n1000.json`

### 3.3 传导机制关键数字（E3b，向量检索框架）

| 系统 | 无摘要 MRR | 有摘要 MRR | 摘要收益 |
|---|---|---|---|
| 标准 Leiden | 0.423 | 0.435 | +2.8% |
| 约束 Leiden λ=0.003 | 0.446 | **0.495** | **+10.9%（是标准的 3.9×）** |

来源文件：`experiments/results_v11_lambda/e3_supplementary.json`

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
    intra_para_weight: float = 2.0,     # v6+ 校准值（v5 为 0.3）
    intra_doc_weight: float = 1.5,      # v6+ 校准值（v5 为 0.2）
    cross_doc_weight: float = 1.0,      # v6+ 校准值（v5 为 0.1）
    max_cross_doc_edges: int = 5,
    include_cross_doc: bool = True,
) -> EdgeSchedule

# === 检索器（支持 TF-IDF 和向量两种后端）===
# retriever.py
class URetriever:
    MODES = {"uretrieval", "topdown_only", "bottomup_only",
             "vector_uretrieval", "vector_topdown_only", "vector_bottomup_only"}

    def __init__(
        self,
        communities_df: pd.DataFrame,
        text_units: List[dict],
        entities_df: pd.DataFrame,
        top_k_communities: int = 5,
        top_k_chunks: int = 5,
        max_context_chars: int = 4000,
        retrieval_mode: str = "uretrieval",       # 见 MODES
        embedding_model: str = "all-MiniLM-L6-v2",  # vector_* 模式使用
        vector_min_level: int = 2,                # 向量 TopDown 最低层
    ): ...

    def retrieve(self, query: str, entity_mentions=None, alpha=0.5) -> RetrievalResult

# 向量检索后端（当 retrieval_mode 以 "vector_" 开头时自动使用）
class VectorTopDownRetriever:   # 社区摘要向量检索，无摘要时退化为实体列表向量
class VectorBottomUpRetriever:  # 段落文本向量检索（替换 TF-IDF BottomUp）

# === QA 端到端评估 ===
# evaluation/qa_evaluator.py
def evaluate_qa_end_to_end(
    qa_pairs: List[QAPair],
    retriever: URetriever,
    llm_config: LlmConfig,
    max_context_chars: int = 3000,
    concurrency: int = 5,
) -> QAMetrics
# QAMetrics 包含：exact_match, token_f1, rouge_l, avg_context_chars, avg_latency_ms

# === 摘要质量评估 ===
# evaluation/summary_quality_evaluator.py
def evaluate_summary_quality(
    samples_a: List[CommunitySample],   # 标准 Leiden 样本
    samples_b: List[CommunitySample],   # 约束 Leiden 样本
    llm_config: LlmConfig,
    concurrency: int = 10,
) -> Tuple[SummaryQualityMetrics, SummaryQualityMetrics]
# SummaryQualityMetrics 包含：avg_focus_score(1-5), avg_entity_coverage, avg_num_docs, pct_single_doc

# === 实验运行 ===
# experiments/run_multihop_eval.py L298
# 注意：RunConfig.use_edge_schedule 是 bool，在 run_one() 中转换：
#   if cfg.use_edge_schedule:
#       edge_schedule = EdgeSchedule.build(entities_df, include_cross_doc=cfg.edge_schedule_cross_doc)
#   传给 run_constrained_community_detection(edge_schedule=edge_schedule)
```

---

## §5 已知缺陷与修复历史

### 5.1 v4 三大缺陷（已在 v5 修复，v7/v8 实验验证生效）

| # | 缺陷 | 根因 | 影响 | 修复版本 |
|---|---|---|---|---|
| D1 | 结构熵 H ≡ 0 | `anchor_granularity="sent"` → 每节点一个唯一 chunk_id → 社区内无分布可言 | λ·H 恒为 0，结构熵惩罚完全不工作 | v5：改为 `"para"` → H > 0 |
| D2 | 底层图为断裂的句子森林 | 节点是实例级（sentence-scoped），仅有句内共现边 → ~24,858 个连通分量 | Leiden 无法跨句合并，层次结构意义有限 | v5：EdgeSchedule 分 3 级注入跨句/跨段/跨文档边 |
| D3 | λ 控制层数而非合并节奏 | `lambda_val < 1e-6: break` 终止条件 → λ 衰减到阈值以下就停止 | λ 值直接决定层次深度，而非调节合并倾向 | v5：移除该终止条件，改用收敛判定 |

### 5.2 BottomUp 锚点 Bug（✅ 已在 v8 修复）

**问题**：BottomUpRetriever 的 `_entity_chunks` 存储 `entity_title → Set[sent_id]`，但 `text_units` 的 `chunk_id` = `para_id`，格式不匹配导致 ×1.5 锚点加权从未触发。

**v8 修复**：`retriever.py` 中 `_entity_chunks` 现在存储 `para_id`（通过 `re.sub(r"-s\d+$", "", sent_id)` 将 sent_id 截断为 para_id），与 text_units 的 chunk_id 格式一致。

**验证结果**：v8 实验中 Para-MRR 全 6 组恒定为 0.4275，确认段落级匹配路径已统一生效。

---

## §6 当前实验脚本（v11 向量检索，4 组）

来源：`graphrag_improved/experiments/run_multihop_eval.py`

当前配置（n=500，429 有效 QA，向量检索模式）：

| 组号 | 名称 | λ | 检索模式 | 摘要 |
|---|---|---|---|---|
| [0] | A+VS GraphRAG-replica | 0 | vector_topdown_only | ✓ |
| [1] | B3+VS SP-GraphRAG（核心） | 0.003 | vector_topdown_only | ✓ |
| [2] | C3+VS 完整系统 | 0.003 | vector_uretrieval | ✓ |
| [3] | D+V 纯向量段落 | — | vector_bottomup_only | ✗ |

**运行命令**（需 Kimi API key）：
```bash
cd /Users/ttung/Desktop/个人学习/SP-GraphRAG/graphrag_improved
python -m experiments.run_multihop_eval \
    --n-qa 500 --with-summary --provider kimi --api-key KEY \
    --data-dir ../data/multihop_rag --output-dir experiments/results_v11

# λ 消融（摘要质量实验，零成本）：
python -m experiments.run_summary_quality_eval \
    --provider kimi --api-key KEY --n-samples 200 \
    --output-dir experiments/results_summary_quality_full
```

---

## §7 目录结构速查

```
graphrag_improved/
├── constrained_leiden/
│   ├── graphrag_workflow.py      # 主入口 run_constrained_community_detection
│   ├── leiden_constrained.py     # hierarchical_leiden_constrained
│   ├── edge_scheduler.py         # EdgeSchedule 三级边注入
│   ├── physical_anchor.py        # PhysicalNode + 结构熵计算
│   └── annealing.py              # AnnealingConfig + 4 种退火曲线
├── extraction/
│   └── extractor.py              # Entity/Relation 抽取（spaCy NER）
├── data/
│   └── ingestion.py              # Document → TextUnit → SentenceUnit
├── retrieval/
│   └── retriever.py              # URetriever（6 种检索模式，TF-IDF/向量）
├── summarization/
│   └── summarizer.py             # generate_community_summaries（Kimi/OpenAI）
├── evaluation/
│   ├── evaluator.py              # 检索评估（MRR/P@K/R@K/NDCG + Bootstrap CI）
│   ├── qa_evaluator.py           # 端到端 QA 评估（EM/F1，需 LLM）
│   └── summary_quality_evaluator.py  # 摘要质量评估（主题聚焦度打分）
├── experiments/
│   ├── run_multihop_eval.py      # 主实验脚本（4 组向量检索，含 QA 评估）
│   ├── run_summary_quality_eval.py   # 摘要质量对比实验
│   ├── results_v7/               # v7 消融结果（n=200，TF-IDF）
│   ├── results_v8/               # v8 结果（n=500，TF-IDF，Bootstrap CI）
│   ├── results_v8b/              # v8b 变量隔离基准
│   ├── results_v9/               # v9 LLM 摘要实验（TF-IDF）
│   ├── results_v11/              # v11 向量检索（n=500）⭐ 核心
│   ├── results_v11b/             # v11b 向量检索大样本（n=1000）⭐ 核心
│   └── results_v11_lambda/       # λ 消融 + E3a/E3b 补充实验
├── summary_cache/
│   ├── summaries_leiden_standard.json      # λ=0 摘要缓存（15,177 条）
│   └── summaries_leiden_constrained_003.json  # λ=0.003 摘要缓存（109,122 条）
├── config.yaml
├── README.md
└── docs/archive/                 # 历史文档归档
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
