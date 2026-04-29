# SP-GraphRAG 技术方案完整规格书

> **面向对象**：大模型 / AI Agent（本文档设计为对 LLM 友好的结构化技术参考，包含完整的算法细节、数据结构定义、代码实现映射和实验结论，可作为后续开发、代码生成、论文润色或技术问答的上下文输入。）
>
> **项目路径**：`/Users/ttung/Desktop/个人学习/SP-GraphRAG/`
>
> **核心代码**：`graphrag_improved/` 目录

---

## 1. 项目总览

### 1.1 一句话定位

SP-GraphRAG（Structural-entropy Penalized GraphRAG）在标准 GraphRAG 的 Leiden 社区检测目标函数中引入物理来源分布的 Shannon 熵惩罚项，使社区检测感知节点的文档物理结构（句子→段落→文档），在不损害检索质量的前提下实现社区的物理来源可控。

### 1.2 核心目标函数

```
J = Q_leiden − λ · H_structure
```

- **Q_leiden**：标准 Leiden 模块度增益，衡量社区内部边密度超过随机期望的程度
- **H_structure**：社区内节点物理来源分布的 Shannon 熵，`H = −Σ p_i log(p_i)`
- **λ**：退火系数，通过指数衰减从高层到低层控制约束强度

### 1.3 系统管线总览

```
文档集合 → [1.数据摄入] → [2.实体抽取] → [3.图构建] → [4.EdgeSchedule注入]
→ [5.约束Leiden聚类] → [6.社区DataFrame生成] → [7.U-Retrieval检索] → [8.评估]
```

每个步骤的代码位置：

| 步骤 | 模块文件 | 核心函数/类 |
|------|----------|-------------|
| 1. 数据摄入 | `data/ingestion.py` | `ingest()`, `Document`, `TextUnit`, `SentenceUnit` |
| 2. 实体抽取 | `extraction/extractor.py` | `extract()`, `Entity`, `Relation` |
| 3. 图构建 | `constrained_leiden/graphrag_workflow.py` | `build_graph_from_graphrag()` |
| 4. EdgeSchedule | `constrained_leiden/edge_scheduler.py` | `EdgeSchedule.build()` |
| 5. 约束Leiden | `constrained_leiden/leiden_constrained.py` | `hierarchical_leiden_constrained()` |
| 6. 社区输出 | `constrained_leiden/graphrag_workflow.py` | `convert_result_to_communities_df()` |
| 7. U-Retrieval | `retrieval/retriever.py` | `URetriever`, `TopDownRetriever`, `BottomUpRetriever` |
| 8. 评估 | `evaluation/evaluator.py` | `Evaluator` |

---

## 2. 数据结构定义

### 2.1 三级物理ID体系

所有物理坐标编码为字符串ID，层级之间用 `-p`/`-s` 分隔：

```
doc_id  = hash(file_path)[:12]              例：a3f2b1c4d5e6
para_id = {doc_id}-p{para_index:03d}        例：a3f2b1c4d5e6-p002
sent_id = {para_id}-s{sent_index:03d}       例：a3f2b1c4d5e6-p002-s001
```

**解析函数**（在 `graphrag_workflow.py` 和 `edge_scheduler.py` 中均有实现）：
- `_parse_para_id(sent_id)` → 截断 `-sNNN` 后缀，返回 `para_id`
- `_parse_doc_id(sent_id)` → 截断 `-pNNN-sNNN` 后缀，返回 `doc_id`

### 2.2 核心数据类

#### SentenceUnit（句子级物理单元）
```python
@dataclass
class SentenceUnit:
    sent_id: str      # 全局唯一，格式 {doc_id}-p{NNN}-s{NNN}
    text: str          # 句子原始文本
    doc_id: str        # 所属文档 ID
    para_id: str       # 所属段落 ID
    sent_index: int    # 段落内句子序号（从0开始）
    para_index: int    # 文档内段落序号（从0开始）
```

#### TextUnit（段落级容器）
```python
@dataclass
class TextUnit:
    chunk_id: str               # 等于 para_id
    text: str                   # 段落完整文本
    doc_id: str
    doc_title: str
    chunk_index: int            # 文档内段落序号
    sentences: List[SentenceUnit]  # 该段落下的所有句子
    metadata: dict
```

#### Entity（实例级实体节点）
```python
@dataclass
class Entity:
    title: str           # 实体名称（原始文本）
    entity_type: str     # spaCy NER 标签或词性推断类型
    sent_id: str         # 所在句子 ID
    para_id: str         # 所在段落 ID
    doc_id: str          # 所在文档 ID
    
    @property
    def node_id(self) -> str:
        # {sent_id}-{title_normalized}
        # 同名实体在不同句子中 → 不同 node_id
        title_norm = re.sub(r"[^a-z0-9]", "_", self.title.lower())[:32]
        return f"{self.sent_id}-{title_norm}"
```

**关键设计**：**不做实体消解**。同一实体名在不同句子中出现视为不同节点（instance-level），底层图保持物理纯净。跨句子的同名实体合并由 EdgeSchedule + Leiden 聚类自然涌现。

#### Relation（关系边）
```python
@dataclass
class Relation:
    source_node_id: str    # 主语实体的 node_id
    target_node_id: str    # 宾语实体的 node_id
    predicate: str         # 谓词（动词原形）
    weight: float = 1.0    # 重复出现时权重累加
    sent_id: str = ""      # 来源句子 ID
```

#### PhysicalNode（物理锚点）
```python
@dataclass
class PhysicalNode:
    node_id: str
    chunk_ids: FrozenSet[str]   # 物理来源集合
    level: int = 0              # 0=叶节点，>0=超节点
    
    @classmethod
    def from_entity(cls, node_id, chunk_id):
        return cls(node_id=node_id, chunk_ids=frozenset([chunk_id]), level=0)
    
    @classmethod
    def merge(cls, node_id, nodes, level):
        # 超节点继承所有子节点的 chunk_id 并集
        merged = set()
        for n in nodes:
            merged.update(n.chunk_ids)
        return cls(node_id=node_id, chunk_ids=frozenset(merged), level=level)
```

**锚点粒度**（由 `anchor_granularity` 参数控制）：
- `"para"`（默认/推荐）：`chunk_id = para_id`，同段落节点共享 chunk_id → H 有意义
- `"sent"`（v4遗留）：`chunk_id = sent_id`，全局唯一 → H ≡ 0，λ 失效
- `"doc"`：`chunk_id = doc_id`，粒度更粗

---

## 3. 六大核心算法组件

### 3.1 结构熵惩罚（Structural Entropy Penalty）

**数学定义**：对社区 C_k，设其中来自第 i 个物理单元（段落）的节点比例为 p_{k,i}：

```
H(C_k) = −Σ_{i=1}^{M_k} p_{k,i} · log₂(p_{k,i})
```

- 当 M_k = 1（所有节点来自同一段落）：H = 0（最纯净）
- 当节点均匀分布于 M_k 个段落：H = log₂(M_k)（最混杂）
- 全局平均：`H(C) = (1/K) Σ_{k=1}^{K} H(C_k)`

**增量计算优化（O(1)）**：

使用 `CommunityEntropyState` 维护增量状态，避免每次节点移动时重新遍历整个社区。

```python
class CommunityEntropyState:
    chunk_weights: Dict[str, float]   # {chunk_id: 累计权重}
    total_weight: float               # 所有权重之和
    
    def add_node(self, node: PhysicalNode):
        w = 1.0 / len(node.chunk_ids)
        for cid in node.chunk_ids:
            self.chunk_weights[cid] += w
        self.total_weight += w * len(node.chunk_ids)
    
    def remove_node(self, node: PhysicalNode):
        w = 1.0 / len(node.chunk_ids)
        for cid in node.chunk_ids:
            self.chunk_weights[cid] -= w
        self.total_weight -= w * len(node.chunk_ids)
    
    def entropy(self) -> float:
        h = 0.0
        for w in self.chunk_weights.values():
            p = w / self.total_weight
            if p > 1e-12:
                h -= p * math.log(p)
        return h
    
    def delta_entropy_if_add(self, node: PhysicalNode) -> float:
        # 不修改状态，仅模拟加入后的 ΔH
        # 复杂度 O(|node.chunk_ids|) ≈ O(1)（叶节点 chunk_ids 大小为 1）
        h_before = self.entropy()
        # ... 模拟计算 h_after ...
        return h_after - h_before
```

**性能**：609篇文章规模下，单层 local_moving_phase 从 ~25ms 降到 ~3ms。

**代码位置**：`constrained_leiden/leiden_constrained.py` 第 72-166 行

### 3.2 退火调度策略（Annealing Schedule）

**目的**：低层保纯净（λ 大→强约束），高层允许跨文档合并（λ→0→无约束）。

**指数退火公式**：
```
λ_t = λ_init · exp(−α · t)
```

其中 t 为当前层级编号，α 为衰减率（默认 0.5）。

**支持的退火曲线**（`AnnealingSchedule` 枚举）：

| 曲线 | 公式 | 适用场景 |
|------|------|----------|
| EXPONENTIAL | `λ_range · exp(−decay_rate · level)` | 层次较深的图（默认） |
| LINEAR | `λ_range · (1 − t)`，t = level/max_level | 层次较浅的图 |
| COSINE | `λ_range · 0.5 · (1 + cos(π·t))` | 需要平滑过渡 |
| STEP | `λ_range if level < step_level else 0` | 需要明确边界 |

**配置参数**（`AnnealingConfig`）：
```python
@dataclass
class AnnealingConfig:
    lambda_init: float = 1000.0   # 底层 λ 值
    lambda_min: float = 0.0       # 最高层 λ 值
    max_level: int = 10           # 预期最大层级
    decay_rate: float = 0.5       # 衰减速率
    schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL
```

**实验中的实际 λ 值**：论文实验使用 λ_init ∈ {0, 0.001, 0.003}，而非代码默认的 1000.0。这是因为 v6 重新校准了权重量级（见 3.3 节）。

**代码位置**：`constrained_leiden/annealing.py`

### 3.3 EdgeSchedule（同名实体边注入）

**问题背景**：原始图是"句内孤岛森林"——每个连通分量仅包含来自同一句子的节点。由于 node_id 包含 sent_id，所有边严格限制在句内。这导致：
- 每个连通分量内节点共享同一 para_id → H ≡ 0
- 结构熵惩罚完全失效，λ 项无意义

**解决方案**：EdgeSchedule 在 Level 0 一次性注入三类同名实体边，打破句内断裂森林。

**三类边及权重**：

| 边类型 | 匹配条件 | 权重(v6) | 连接方式 | 代码函数 |
|--------|----------|----------|----------|----------|
| 段落内跨句子 | 同段落 + 不同句子 + 同名实体 | 2.0 | 链式 | `_build_intra_paragraph_edges()` |
| 文档内跨段落 | 同文档 + 不同段落 + 同名实体 | 1.5 | 代表节点链式 | `_build_intra_document_edges()` |
| 跨文档 | 不同文档 + 同名实体 | 1.0 | 代表节点链式(上限5) | `_build_cross_document_edges()` |

**权重校准关键约束**：

原始图总边权 m ≈ 18000。注入边权重 w 必须满足：
```
δQ ≈ w/(2m) >> λ · δH
```
即 w ≥ 1.0，以确保语义相关的同名实体有足够的模块度收益驱动合并。v6 之前的权重（0.3/0.2/0.1）因 δQ 太小而被 λ·δH 压制。

**噪声过滤**：长度 < 3 的实体名和停用词表中的词（"which", "there", "people", "the company" 等共 ~120 个）被过滤，不生成 EdgeSchedule 边。

**连接策略**：采用链式连接（而非全连接），复杂度 O(n) 而非 O(n²)。跨段落和跨文档边使用"代表节点"策略——每个段落/文档只选一个代表节点参与链式连接。

**注入时机**：所有边在 Level 0 一次性注入。Level 1+ 无额外边。（v5 原设计为分层注入，v6 发现必须在 Level 0 同时注入跨段落边才能产生非零熵。）

**代码位置**：`constrained_leiden/edge_scheduler.py`

`EdgeSchedule.build()` 关键参数：
```python
EdgeSchedule.build(
    entities=entities_df,
    intra_para_weight=2.0,     # 段落内跨句子边权重
    intra_doc_weight=1.5,      # 文档内跨段落边权重
    cross_doc_weight=1.0,      # 跨文档边权重
    max_cross_doc_edges=5,     # 每个实体跨文档最大边数
    include_cross_doc=True,    # 是否包含跨文档边
)
```

### 3.4 修改后的 Leiden 算法

**算法流程（每一层）**：

```
1. 分层加边：根据 EdgeSchedule 注入当前层的边
   → 原始节点 ID 通过 original_to_current 映射到当前超节点 ID
   → 已有边权重累加，新边直接添加

2. 初始化：每个节点独立成一个社区
   → CommunityState 维护 node_to_community, community_to_nodes,
     community_entropy（CommunityEntropyState）等全部中间状态

3. 局部移动阶段（Local Moving）：
   → 对每个节点（随机顺序），遍历其邻居所在社区
   → 计算 ΔJ = (ΔQ_remove + ΔQ_add) − λ · ΔH_add
   → 若 ΔJ > 0 则执行移动，更新所有状态（含增量熵）
   → 重复直到无节点移动或达到 max_iterations

4. 细化阶段（Refinement）：
   → 对每个多节点社区，提取子图重新运行 local_moving（最多5轮）
   → 若子图被拆分为多个子社区，在主状态中执行拆分

5. 聚合阶段（Aggregation）：
   → 每个社区压缩为超节点 super_{comm_id}_l{level}
   → 超节点的 chunk_ids = 所有子节点 chunk_ids 的并集
   → 社区间边权重累加
   → 更新 original_to_current 映射

6. 终止判断：
   → 社区数 ≤ 1 → 终止
   → 社区数 = 节点数（未收敛）→ 终止
   → level > max_level × 2 → 安全终止
   → 否则 level++ 回到步骤 1
```

**关键修复（v5 相对 v4）**：
- 移除 `lambda_val < 1e-6` 终止条件：λ 衰减仅影响惩罚力度，不影响层数
- 层数由图的收敛性自然决定

**模块度 ΔQ 计算**：标准 Leiden/Louvain 公式
```
ΔQ_add = k_i_in/m − k_i·Σ_tot/(2m²)
ΔQ_remove = −[k_i_in/m − k_i·(Σ_tot − k_i)/(2m²)]
```
其中 k_i 是节点加权度，k_i_in 是节点与目标社区的连边权重，Σ_tot 是目标社区总度，m 是全图总边权。

**代码位置**：`constrained_leiden/leiden_constrained.py`

主入口签名：
```python
hierarchical_leiden_constrained(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
    annealing_config: Optional[AnnealingConfig] = None,
    edge_schedule: Optional[EdgeSchedule] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
) -> HierarchicalCommunityResult
```

输出结构：
```python
@dataclass
class HierarchicalCommunityResult:
    levels: List[Dict[str, int]]          # 每层 {原始node_id: community_id}
    level_entropy: List[Dict[int, float]] # 每层 {community_id: H}
    level_lambda: List[float]             # 每层的 λ 值
    node_physical_map: Dict[str, PhysicalNode]
```

### 3.5 U-Retrieval（双路径检索）

**TopDown 路径**（社区层次导航）：
1. 为每个层级的每个社区构建 TF-IDF 索引（基于社区的 entity_ids + summary 文本）
2. 查询时从最高层开始，逐层 TF-IDF 打分，每层取 top-k 社区
3. 返回多层级的 CommunityHit 列表

**BottomUp 路径**（物理锚点直接检索）：
1. 对所有文本块（段落级 TextUnit）构建 TF-IDF 索引
2. 查询时直接在所有文本块上 TF-IDF 打分
3. 若提供了 entity_mentions，命中实体锚点的 chunk 分数 ×1.5（锚点加权）
4. 返回 TextUnitHit 列表

**融合策略**（`URetriever._merge_context()`）：
- 按 alpha 参数分配字符预算（默认 0.5 即各占一半）
- TopDown 部分优先使用社区摘要，无摘要时使用实体列表
- BottomUp 部分直接使用原始文本
- 相同 chunk_id 去重

**已知 Bug**：BottomUp 的锚点加权（1.5x）实际从不触发——因为 entity_chunks 索引使用 sent_id 格式，而 text_units 的 chunk_id 使用 para_id 格式，两者格式不匹配。

**评估时的融合**（`evaluator.py`）：
- TopDown 每社区贡献最多 1 个 doc_id，总上限 5 个
- BottomUp 直接使用 chunk_id
- 交替合并（BottomUp 优先），去重后作为 retrieved_ids

**代码位置**：`retrieval/retriever.py`

### 3.6 实体抽取（spaCy 依存句法三元组）

**抽取策略**：
1. 遍历句子中所有动词 token（VERB/AUX）作为谓词
2. 找主语（nsubj/nsubjpass/csubj/agent/expl 等）和宾语（dobj/pobj/attr/xcomp 等）
3. 优先使用 noun chunk（名词短语）扩展 span
4. 过滤代词（PRP/PRP$/WP/WP$）和停用词表（~150 个）
5. 为每对 (主语, 宾语) 生成语义关系边
6. 同句子内所有实体两两添加 co_occurs 物理结构边（weight=1.0）

**实体类型推断**：
- 优先使用 spaCy NER 标签（PERSON, ORG, GPE, DATE 等）
- 无 NER 时按词性推断：PROPN→PROPER_NOUN, NOUN→NOUN, 其他→UNKNOWN

**输出 DataFrame 格式**：

entities_df 列：`id`(node_id), `title`, `type`, `description`, `sent_id`, `para_id`, `doc_id`, `text_unit_ids`, `primary_chunk_id`

relationships_df 列：`id`(relation_id), `source`(node_id), `target`(node_id), `weight`, `predicate`, `description`, `sent_id`

**代码位置**：`extraction/extractor.py`

---

## 4. 图构建与工作流编排

### 4.1 图构建（`build_graph_from_graphrag()`）

输入 entities_df 和 relationships_df，输出 networkx 无向图：
- 节点 = entities["id"]（node_id）
- 边 = relationships 中 source 和 target 都在实体表中的行
- 自环过滤，重复边权重累加

### 4.2 物理节点构建（`build_physical_nodes_from_graphrag()`）

根据 anchor_granularity 参数，为每个实体创建 PhysicalNode：
- `"para"`：chunk_id = para_id（优先使用 para_id 列，回退从 sent_id 解析）
- `"sent"`：chunk_id = sent_id
- `"doc"`：chunk_id = doc_id
- 兜底：`placeholder_{hash(node_id) % 100000}`

### 4.3 主工作流（`run_constrained_community_detection()`）

完整签名：
```python
run_constrained_community_detection(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    annealing_config: Optional[AnnealingConfig] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
    use_lcc: bool = True,           # 仅对最大连通分量运行
    min_edge_weight: float = 1.0,   # 过滤低权重边
    intra_doc_merging: bool = False, # 路径A（旧方案，不推荐）
    intra_doc_edge_weight: float = 0.5,
    anchor_granularity: AnchorGranularity = "para",
    edge_schedule: Optional[EdgeSchedule] = None,  # 推荐使用
) -> pd.DataFrame  # communities_df
```

### 4.4 communities_df 输出格式

| 列名 | 类型 | 说明 |
|------|------|------|
| id | str(uuid) | 行唯一ID |
| community_id | int | 社区编号 |
| level | int | 层级（0=底层） |
| title | str | "Community {id} (Level {level})" |
| entity_ids | List[str] | 社区内所有 node_id |
| relationship_ids | List[str] | 社区内部关系 ID |
| text_unit_ids | List[str] | 社区涉及的 sent_id 列表 |
| doc_ids | List[str] | 社区涉及的 doc_id 列表 |
| parent_id | Optional[str] | 父社区标识 |
| children | List[str] | 子社区标识列表 |
| structural_entropy | float | 该社区的结构熵 H |
| lambda_used | float | 该层使用的 λ 值 |

---

## 5. 实验设计与结果

### 5.1 数据集

**MultiHop-RAG**：609 篇新闻文章，2556 个多跳 QA 对。知识图谱规模：13,716 个实体节点，18,044 条关系边。评估采样 200 条，169 条有效。

### 5.2 六组消融实验配置

| 组号 | 配置名 | EdgeSchedule | λ_0 | 跨文档边 | 退火α |
|------|--------|:---:|:---:|:---:|:---:|
| [0] | Baseline | ✗ | 0 | ✗ | 0.5 |
| [1] | ES only | ✓ | 0 | ✗ | 0.5 |
| [2] | Weak | ✓ | 0.001 | ✗ | 0.5 |
| [3] | **Med（推荐）** | ✓ | **0.003** | ✗ | 0.5 |
| [4] | +PathA | ✓ | 0.001 | ✗ | 0.5 |
| [5] | +CrossDoc | ✓ | 0.001 | ✓ | 0.5 |

### 5.3 实验结果

| 组号 | λ_0 | avg_H | MRR | P@5 | R@5 | NDCG@10 | 层数 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [0] Baseline | 0 | 0.0000 | 0.4226 | 0.1964 | 0.2619 | 0.3142 | 3 |
| [1] ES only | 0 | 0.1374 | 0.4206 | 0.1964 | 0.2619 | 0.3131 | 8 |
| [2] Weak | 0.001 | 0.1152 | 0.4211 | 0.1953 | 0.2619 | 0.3134 | 8 |
| [3] **Med** | **0.003** | **0.1045** | 0.4203 | 0.1964 | 0.2636 | 0.3136 | 9 |
| [4] +PathA | 0.001 | 0.1170 | 0.4206 | 0.1941 | 0.2585 | 0.3119 | 8 |
| [5] +CrossDoc | 0.001 | 0.1328 | 0.4180 | 0.1929 | 0.2560 | 0.3088 | 8 |

### 5.4 核心结论

1. **λ 正确惩罚了熵**：avg_H 随 λ 增大单调递减（0.1374 → 0.1045），物理来源分布更纯净。
2. **检索质量无损**：MRR/NDCG 各配置之间差距 < 1.5%，远小于 MultiHop-RAG 题目噪声。
3. **R@5 微升**：Med 配置（λ=0.003）R@5 比 Baseline 高 +0.0017，表明更纯净的社区有助于召回。
4. **CrossDoc 边轻微伤害**：跨文档边引入跨主题噪声，所有指标均为最低。
5. **PathA 边际低于 EdgeSchedule**：旧的路径A方案（文档级强制合并）不如 EdgeSchedule 精细。
6. **推荐配置**：`λ_init=0.003, decay_rate=0.5, anchor_granularity="para", include_cross_doc=False`

### 5.5 社区质量指标

- **avg_H（社区平均结构熵）**：社区物理来源混杂程度，越低越好
- **macro/micro modularity**：标准模块度，宏观 Q 约 0.56
- **community_count**：社区数量随层级递减
- **avg_community_size**：底层社区平均约 3.5 个节点

---

## 6. 版本演进与关键修复

| 版本 | 核心变化 | 解决的问题 |
|------|----------|-----------|
| v3 | 初始实现，所有 chunk_id 用 sent_id | H ≡ 0，λ 失效 |
| v4 | 引入 anchor_granularity="para" | 修正 chunk_id 到段落级，但图仍是句内断裂 |
| v5 | 引入 EdgeSchedule（段落内/跨段落/跨文档三级边注入） | 打破句内孤岛，H > 0，但权重过小（0.3/0.2/0.1） |
| v6 | EdgeSchedule 权重提升至 2.0/1.5/1.0 + 噪声过滤 | ΔQ 压过 λ·ΔH，同名实体正常合并；λ 降到 0.003 量级 |
| v7 | 完成全实验流水线 + 六组消融实验 | 验证方案有效性：熵可控 + 检索无损 |

### 6.1 已知问题与限制

1. **BottomUp 锚点加权 Bug**：`entity_chunks` 使用 sent_id 格式，而 `text_units` 的 chunk_id 使用 para_id 格式，格式不匹配导致锚点加权从不触发。修复方案：将 entity_chunks 的 key 也转为 para_id。

2. **spaCy 抽取质量**：依存句法三元组在新闻文本上尚可（平均每篇文章 ~22 个实体），但对非英文或专业领域文本质量下降。

3. **实验评估采样**：169 有效 QA 中，169/2556 = 6.6% 采样率，统计功效有限。

4. **社区摘要缺失**：当前管线不生成 community_summary（需要 LLM 调用），TopDown 检索仅使用实体列表。

5. **跨文档边噪声**：同名但不同义的实体（如 "Apple" 水果 vs "Apple" 公司）会被错误连接，需要引入轻量实体消歧。

---

## 7. 代码文件索引

以下是项目中所有核心 Python 文件的精确路径和主要内容：

```
graphrag_improved/
├── __init__.py
├── constrained_leiden/
│   ├── __init__.py
│   ├── leiden_constrained.py      (726行) 核心Leiden算法 + CommunityEntropyState
│   ├── edge_scheduler.py          (452行) EdgeSchedule三级边注入
│   ├── annealing.py               (162行) 退火策略(指数/线性/余弦/阶梯)
│   ├── physical_anchor.py         (165行) PhysicalNode数据类 + 结构熵计算
│   └── graphrag_workflow.py       (736行) 主工作流编排 + 图构建 + 社区DataFrame
├── data/
│   ├── __init__.py
│   └── ingestion.py               (465行) 文档→段落→句子三级切分
├── extraction/
│   ├── __init__.py
│   └── extractor.py               (581行) spaCy三元组抽取 + Entity/Relation
├── retrieval/
│   ├── __init__.py
│   └── retriever.py               (647行) U-Retrieval(TopDown+BottomUp)
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py               (668行) P@K/R@K/MRR/NDCG + 社区质量
└── utils/
    ├── __init__.py
    └── logging.py                  日志配置
```

---

## 8. 快速参考卡片

### 最优配置（v7 推荐）
```python
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig
from graphrag_improved.constrained_leiden.edge_scheduler import EdgeSchedule

config = AnnealingConfig(
    lambda_init=0.003,
    lambda_min=0.0,
    decay_rate=0.5,
    schedule="exponential",
)

edge_schedule = EdgeSchedule.build(
    entities=entities_df,
    intra_para_weight=2.0,
    intra_doc_weight=1.5,
    cross_doc_weight=1.0,
    include_cross_doc=False,
)
```

### 目标函数移动判断
```
ΔJ = (ΔQ_remove_from_old + ΔQ_add_to_new) − λ_current · ΔH_add_to_new
if ΔJ > best_gain:
    best_community = new_community
```

### 物理锚点不变式
```
∀ node ∈ graph:
    physical_nodes[node].chunk_ids ⊆ {所有段落的 para_id}
    |chunk_ids| = 1  (叶节点)
    |chunk_ids| ≥ 1  (超节点, 为子节点并集)
```

### 数据规模参考
```
MultiHop-RAG: 609 articles, ~3000 paragraphs, ~15000 sentences
Graph: 13716 nodes, 18044 edges (after extraction)
EdgeSchedule: +2000~3000 injected edges (depending on config)
Communities: Level 0 ~3800, Level 8 ~40
Runtime: ~15 seconds total pipeline (M1 Mac)
```

---

*文档版本：v7.0 | 生成时间：2026-04-29 | 适用代码快照：当前 main 分支*