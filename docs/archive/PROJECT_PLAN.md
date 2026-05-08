# GraphRAG Improved 项目计划书

> **文档状态**：本计划书于 2026-04-27 编写，反映 v5 的设计意图和实施计划。  
> **当前进度**：阶段 A（核心算法修复）和阶段 B（分层加边实现）的代码已完成，阶段 C（实验与评估）和阶段 D（文档与收尾）待执行。

## 一、项目概述

### 1.1 项目名称

GraphRAG Improved：基于结构熵约束的层次化社区发现增强图检索增强生成

### 1.2 研究动机

微软 GraphRAG 通过将文档构建为知识图谱并做层次化社区发现，为大语言模型提供结构化的检索上下文。但其标准 Leiden 算法在社区划分时完全依赖拓扑模块度，忽略了一个关键信息：**文本实体在原始文档中的物理位置**。物理位置近的实体往往语义关联更紧密——同一句话中的实体几乎必然相关，同一段落的实体通常讨论同一话题，而跨文档的同名实体可能指代完全不同的概念。

本项目的核心假设是：**如果在社区发现过程中引入物理位置约束，让物理距离近的实体优先合并为社区，就能构建出语义一致性更高的社区层次结构，从而提升检索质量。**

### 1.3 核心创新点

本项目提出在 Leiden 算法的目标函数中引入结构熵惩罚项，形成约束优化目标：

```
J = Q_leiden − λ · H_structure
```

其中 Q 是标准模块度增益，H 是社区内节点物理来源分散程度的 Shannon 熵，λ 是约束强度系数。通过 λ 退火（annealing）机制，实现从局部到全局的渐进式社区合并：高层次的 λ 值大，惩罚物理来源分散的合并，优先形成物理位置紧凑的底层社区；随层次升高 λ 衰减，惩罚减弱，允许跨越更远物理距离的合并，最终 λ→0 时退化为标准 Leiden，完成高层社区的自由构建。

### 1.4 设想的合并节奏

```
层次 0（λ 最大）→ 同一句子内的实体优先合并为基础社区
层次 1（λ 较大）→ 同一段落内不同句子的社区开始合并
层次 2（λ 中等）→ 同一文档内不同段落的社区合并
层次 3+（λ 趋近 0）→ 跨文档的社区合并，结构熵约束逐步消失
最终状态　　　　→ λ = 0，退化为标准 Leiden，完成高层社区自由构建
```

---

## 二、技术方案

### 2.1 整体架构

系统采用四阶段流水线架构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GraphRAG Improved Pipeline                   │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────┐ │
│  │  Ingest   │──▶│   Extract    │──▶│  Community   │──▶│  Save   │ │
│  │  三级切分  │   │  三元组抽取   │   │  约束Leiden   │   │  输出   │ │
│  └──────────┘   └──────────────┘   └──────────────┘   └─────────┘ │
│       │                │                   │                │      │
│   doc→para→sent   spaCy依存句法      λ退火+结构熵        CSV/HTML  │
│   三级ID体系      实体+关系+物理边   层次化社区发现       Parquet   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
               ┌──────────────────────┐
               │    U-Retrieval 检索   │
               │  TopDown + BottomUp  │
               │    双轨融合检索       │
               └──────────────────────┘
```

### 2.2 阶段一：文档摄入与三级切分（data/ingestion.py）

原始文档经过三级切分，形成层次化的文本单元：

第一级 **Document**：每篇文档分配唯一 doc_id（`md5(file_path)[:12]`，12 位 hex 哈希）。第二级 **Paragraph (TextUnit)**：按双换行符切分段落，分配 para_id，格式为 `{doc_id}-p{NNN}`（NNN 为零填充 3 位序号）。第三级 **Sentence (SentenceUnit)**：使用 spaCy 句子分割，分配 sent_id，格式为 `{para_id}-s{NNN}`。

最终每个句子携带完整的物理路径信息 `{doc_id}-p{para_idx}-s{sent_idx}`，这为后续的物理位置约束提供了坐标基础。

### 2.3 阶段二：实体与关系抽取（extraction/extractor.py）

采用 spaCy 依存句法分析（`en_core_web_sm` 模型），逐句提取实体和关系。

**实体抽取**：每个实体是"实例节点"而非"概念节点"，节点 ID 格式为 `{sent_id}-{entity_name_normalized}`（小写化 + 非字母数字替换为下划线 + 截断 32 字符）。同一个现实实体在不同句子中会产生多个独立节点——这是 v3 "物理优先架构"的核心设计，旨在保留物理位置信息，由社区发现阶段自然涌现实体合并。

**关系抽取**：使用 spaCy 依存句法树提取主谓宾三元组，包括主动态（nsubj→ROOT→dobj）、被动态（nsubjpass→ROOT→agent）和介词短语（ROOT→prep→pobj）三种模式。

**物理结构边**：同一句子内的所有实体两两创建 `co_occurs` 边（weight=1.0）。这些边是图中唯一的连接，严格限定在句子边界内，不同句子的实体之间没有任何边。

**噪声过滤**：维护超过 150 个停用词的过滤列表，包括常见代词、冠词、连词和高频误识别词，在抽取阶段直接过滤。

### 2.4 阶段三：结构熵约束的层次化 Leiden（constrained_leiden/）

这是本项目的核心算法模块，包含四个子组件。

**2.4.1 物理锚点（physical_anchor.py）**

每个节点被封装为 `PhysicalNode`，携带 `chunk_ids`（物理来源标识集合）和 `level`（层次级别）。初始叶节点的 chunk_ids 为单一标识符，超级节点通过 `merge` 方法继承所有子节点 chunk_ids 的并集。

结构熵采用 Shannon 熵公式：对社区内所有节点的 chunk_ids 按频率统计概率分布，计算 H = −∑ p_i · log(p_i)。每个超节点的每个 chunk_id 贡献 1/|chunk_ids| 的权重，使得物理来源越分散的社区熵越高。

**2.4.2 退火调度（annealing.py）**

λ 退火支持四种曲线——指数衰减（exponential，默认）、线性衰减（linear）、余弦退火（cosine）和阶梯函数（step）。默认配置：λ_init=1000.0，λ_min=0.0，decay_rate=0.5，max_level=10。

`get_lambda(level, config)` 函数在 level=0 时返回 λ_init，后续层次按指数衰减 `λ_init · exp(-decay_rate · level)`。

**2.4.3 约束 Leiden 算法（leiden_constrained.py）**

核心算法为 `hierarchical_leiden_constrained`，每层执行三个阶段。

局部移动阶段（local moving）：遍历所有节点，对每个节点计算移入各候选社区的综合增益 `ΔJ = ΔQ − λ · ΔH`。其中 ΔQ 为模块度增益，ΔH 为结构熵增量（通过 `CommunityEntropyState` 的增量更新实现 O(1) 复杂度，相比朴素实现的 O(|community|) 是重要优化）。选择 ΔJ 最大的候选社区进行移动。

细化阶段（refinement）：在大社区（超过 max_cluster_size）内部做二次划分，确保社区粒度适中。

聚合阶段（aggregation）：将同社区节点合并为超级节点，继承 chunk_ids 并集，社区间边做权重聚合，形成下一层的压缩图。

主循环重复上述三阶段直至收敛（社区数不再变化或 ≤1），逐层产出社区划分结果，最终形成层次化社区树。

**2.4.4 GraphRAG 工作流（graphrag_workflow.py）**

封装了从 DataFrame 格式的实体/关系数据到约束 Leiden 输入的完整转换，包括图构建、物理锚点构建、可选的文档内同名实体消解边（Path A，weight=0.5 的 soft edge）注入等。

### 2.5 检索阶段：U-Retrieval 双轨检索（retrieval/retriever.py）

检索采用 Top-Down 和 Bottom-Up 双轨策略，融合社区层次导航与物理锚点定位的优势。

**Top-Down 检索**：从最高层社区开始，基于 TF-IDF 余弦相似度逐层向下导航，在每层选出最相关的 top_k 个社区，收集其关联的 text_unit（文本片段）作为候选上下文。

**Bottom-Up 检索**：对查询做 TF-IDF 匹配，直接定位最相关的 text_unit，同时利用实体的物理锚点（sent_id）给命中实体所在句子加权 1.5 倍（anchor bonus），提升物理位置匹配的优先级。

**融合策略**：两路结果交替合并去重，Top-Down 结果优先（提供全局语境），Bottom-Up 结果补充（提供精确定位），最终裁剪到指定 token 数量。

### 2.6 评估体系（evaluation/evaluator.py）

评估覆盖三个维度。

**检索质量**：Precision@K（K=1,3,5,10）、Recall@K（K=5,10）、F1@K、MRR（平均倒数排名）、NDCG@K（K=5,10）。

**社区质量**：模块度 Q、平均结构熵、Level 0 物理纯净率（同社区节点是否来自同一物理来源）、各层平均社区大小及标准差。

**文本匹配**：Exact Match (EM)、Token-level F1、ROUGE-L。

### 2.7 评估数据集

使用 **MultiHop-RAG** 数据集（Tang & Yang, 2024, COLM 2024），包含 609 篇新闻文章和 2556 个多跳推理 QA 对，分为 inference_query、comparison_query、temporal_query、null_query 四种类型。当前实验采样 200 条 QA，其中 169 条有效（排除 supporting_doc_ids 为空的样本）。

### 2.8 对照基线

**Naive RAG**（baselines/naive_rag/）：使用 sentence-transformers/all-MiniLM-L6-v2 对句子做向量化，余弦相似度 Top-K 检索。

**标准 Leiden（λ=0）**：与本项目共享同一管线，但关闭结构熵约束（λ=0），作为消融基线。

**Microsoft GraphRAG 官方**（baselines/graphrag_official/）：已搭建索引框架，对 20 篇文档完成了索引构建，但因需要 OpenAI API 的成本问题，尚未完成全量索引和对照评估。

---

## 三、当前进展与问题诊断

### 3.1 版本演进

项目经历了五个主要版本：

**v1.0**（初始版本）建立了完整的基础框架，包括带结构熵惩罚的 Leiden 变体（J = Q − λ·H）、λ 退火机制（支持指数/线性/余弦/阶梯四种曲线）、U-Retrieval 双轨检索（Top-Down 社区导航 + Bottom-Up 物理锚点）以及 MultiHop-RAG 对照实验框架。但此时存在物理锚点语义错误——使用 `text_unit_ids`（实体出现过的所有文档）作为锚点，高频实体天然跨多个 chunk，物理约束形同虚设（Level-0 纯净率仅 24.73%）。

**v1.1**（primary_chunk_id 修复 + 增量熵优化）做了两个关键修复。一是修正物理锚点语义错误：新增 `primary_chunk_id` 字段（用 Counter.most_common 确定主锚点）替代原来的 `text_unit_ids`，将 Level-0 纯净率从 24.73% 修复至 100%，平均结构熵从 0.8827 降至 0.0000。二是引入增量熵状态 `CommunityEntropyState`，将 ΔH 计算复杂度从 O(|community|) 降至 O(1)，实测 6.8x 加速。检索指标小幅提升：MRR +1.3%，P@5 +3.5%，NDCG@10 +1.9%。

**v3**（物理优先架构重设计）做了重大架构变革——将节点从“概念级”改为“实例级”（每个句子中的实体是独立节点，ID 含完整物理路径），移除实体消解（由 Leiden 社区发现自然涌现），引入 spaCy 依存句法三元组提取替代旧的共现窗口抽取，新增三级 ID 切分体系（Document→Paragraph→Sentence），并将 U-Retrieval 的 Bottom-Up 检索从 chunk_id 级别精化到 sent_id 级别。这一版本确立了当前的整体架构。

**v4**（消融验证版）在 MultiHop-RAG 的 200 条采样上运行四组消融实验，发现 P@5 +23.2% 的提升。但深入分析后发现这一提升的根本原因是层数差异（λ=0 只跑 1 层产生 22,973 个社区，λ=1000 跑 3 层产生 63,497 个社区），而非结构熵约束的效果——所有实验的结构熵恒为 0.0000。

### 3.2 v4 实验数据

> **数据来源说明**：v4 存在两组实验数据。「数据集 A」中 [0]-[2] 使用未过滤实体集（60,439），[3] 使用过滤后实体集（47,142），数据仅存于 PROJECT_STATUS.md 的历史记录中。「数据集 B」来自 `experiments/results/multihop_results_n200.json`，四组均使用过滤后实体集（47,142）。本文统一引用数据集 B 作为权威来源，详细对比见 PROJECT_STATUS.md 第四节。

四组消融实验在 169 条有效 QA 上的核心指标（数据集 B）：

实验 [0]（λ=0 基线）的 MRR 为 0.3492，P@5 为 0.1325，NDCG@10 为 0.2737，产出 1 层 22,973 个社区。实验 [1]（λ=1000）的 MRR 为 0.3552，P@5 为 0.1633（+23.2%），NDCG@10 为 0.2792，产出 3 层 63,497 个社区。实验 [2]（λ=1000 + Path A）和 [3]（λ=1000 + Path A + Path B）的所有指标与 [1] 完全相同，说明文档内消解边和噪声过滤在该实体集下均无额外效果。

四组实验的结构熵均为 0.0000，模块度均为 0.8025，物理纯净率均为 100%。

对照基线 Naive RAG 的 MRR 为 0.6389，P@5 为 0.2568，NDCG@10 为 0.5216，全面大幅超过本项目的最优结果。

### 3.3 三个根本缺陷

通过对 v4 实验结果和源代码的深入分析，诊断出三个根本缺陷。

**缺陷一：图是断裂的"句子森林"，Leiden 无法跨句合并。** `extractor.py` 中的物理结构边 `co_occurs` 严格限定在同句内（不同句子的实体之间 `continue` 跳过），导致初始图是 N 个互不相连的句子级子图。Leiden 只能在连通分量内部做社区发现，永远无法把来自不同句子的节点合并到同一社区，因为它们之间根本没有路径。

**缺陷二：物理锚点粒度太细（sent_id 全局唯一），H_structure ≡ 0。** `graphrag_workflow.py` 中每个节点的 `chunk_ids = frozenset([sent_id])`，而 sent_id 全局唯一。当考虑把节点 v 移入候选社区 C 时，δH = log((n+1)/n)（n 为社区大小），这个值只取决于社区的当前大小，与社区内节点的物理来源无关。对于大小相同的任意两个候选社区，δH 完全相等，λ·δH 项在比较不同候选社区时完全抵消，节点去向完全由 δQ 单独决定。

**缺陷三：λ 的唯一实际效果是控制层级数量，而非控制合并节奏。** `leiden_constrained.py` 中的终止条件 `if lambda_val < 1e-6: break` 使得 λ=0 时首层即终止（只跑 1 层），而 λ=1000 时需要多层衰减才能触发终止（跑 3 层）。v4 中 P@5 的提升实际来自 3 层 vs 1 层的搜索空间差异，不是结构熵约束带来的社区质量提升。

### 3.4 与设计意图的差距对照

| 设计意图 | 当前实现 | 差距分析 |
|---------|---------|---------|
| 同句内实体优先合并 | 图只有句内边，Leiden 只能句内合并 | 形式上满足，但原因是"别无选择"而非 λ 的主动控制 |
| 同段落→同文档→跨文档渐进扩展 | 图无跨句边，无论 λ 多少都无法跨句合并 | 完全不满足，需要引入跨句边 |
| λ 退火控制合并节奏 | H ≡ 0，λ 不影响节点分配决策 | 完全不满足，需要修改锚点粒度 |
| λ→0 后进入无约束高层构建 | λ 衰减到阈值后直接终止层次化迭代 | 语义错误：当前是"停止"而非"放松约束" |

---

## 四、改进方案（v5）

### 4.1 改进目标

让结构熵约束真正生效，实现从同句→同段→同文档→跨文档的渐进式社区合并，使 P@5 等检索指标的提升确实来自社区质量的改善而非搜索空间的差异。

### 4.2 改进策略：双管齐下

要实现渐进合并，需要同时满足两个条件。条件 A 是图中存在跨句/跨段/跨文档的边（否则 Leiden 没有路径可走）。条件 B 是物理锚点的粒度足以区分"同段落"和"跨文档"（否则 H 无法提供差异化惩罚）。单独满足任一条件都不够——只改锚点粒度但图仍断裂，节点无法跨句移动；只加跨句边但锚点仍是 sent_id，δH 对所有候选社区仍然相等。

### 4.3 修改一：物理锚点粒度从 sent_id 提升为 para_id

**涉及文件**：`extraction/extractor.py`、`constrained_leiden/graphrag_workflow.py`

**核心变更**：将 PhysicalNode 的 `chunk_ids` 从 sent_id 改为 para_id（格式 `{doc_id}-p{NNN}`）。由于 sent_id 格式已包含 para_id 信息（`{doc_id}-p{NNN}-s{NNN}`），通过字符串 `rsplit("-s", 1)[0]` 即可提取 para_id，无需修改上游抽取逻辑。

**预期效果**：同段落不同句子的节点共享同一个 para_id，合并它们时 H 不增加；跨段落的节点有不同的 para_id，合并它们时 H 增加，λ·δH 产生惩罚。高 λ 时优先合并同段落节点，λ 衰减后允许跨段落合并。

**高级选项**：在聚合阶段实现动态锚点粒度切换——Level 0-1 使用 para_id 粒度（惩罚跨段落合并），Level 2+ 将 chunk_ids 中的 para_id 映射为 doc_id 粒度（惩罚跨文档合并），配合 λ 衰减形成双层约束递减。

**向后兼容**：通过配置项 `anchor_granularity: "para" | "sent" | "doc"` 控制锚点粒度，默认 `"para"`，设为 `"sent"` 时行为与 v4 完全一致。

### 4.4 修改二：分层加边（EdgeSchedule）

**涉及文件**：新建 `constrained_leiden/edge_scheduler.py`，修改 `leiden_constrained.py`

**核心变更**：不在初始图中一次性加入所有边，而是随层次递增逐步引入更远距离的边。实现 `EdgeSchedule` 类，维护 level → edges 的映射。在 `hierarchical_leiden_constrained` 主循环中，每层开始前调用 `edge_schedule.get_edges_for_level(level)` 注入新边。

**分层策略**：Level 0 不加额外边，仅保留原始句内 co_occurs 边（weight=1.0），Leiden 在句子内部完成基础社区构建。Level 1 加入同段落内不同句子的同名实体边（weight=0.3），允许跨句合并。Level 2 加入同文档内跨段落的同名实体边（weight=0.2），允许跨段落合并。Level 3 加入跨文档的同名实体边（weight=0.1），允许跨文档合并。

**权重设计原则**：距离越远的边权重越低，使模块度增益 δQ 对远距离合并的驱动力更弱，配合 λ 退火的 δH 惩罚递减形成双重控制。

**实体匹配逻辑**：复用 `graphrag_workflow.py` 中 `build_intra_doc_entity_edges` 的思路（按 `title.lower()` 分组），但按物理距离层次过滤。同名实体间采用链式连接（避免 O(n²) 全连接），跨文档边可设置最大数量上限（如 `max_cross_doc_edges=5`）控制图的稠密度。

**节点 ID 映射**：Level > 0 时图中的节点已是超级节点（`super_xxx_lN`），注入边需将原始节点 ID 映射为当前层的超级节点 ID。需在主循环中维护 `original_to_current: Dict[str, str]` 映射表，在每次聚合后更新。

### 4.5 修改三：修正终止条件语义

**涉及文件**：`constrained_leiden/leiden_constrained.py`

**核心变更**：移除 `if lambda_val < 1e-6: break` 终止条件。这一行的存在使得 λ 的衰减直接决定了层数——λ=0 只跑 1 层，λ=1000 跑 3 层——导致不同实验组的搜索空间不同，评估不公平。

在新方案中，λ→0 应意味着"结构熵约束消失，进入自由 Leiden 模式"，而非"停止迭代"。层数应由图的收敛性自然决定：当社区数不再变化（`num_communities <= 1` 或 `num_communities == len(nodes)`）或达到安全上限时终止。

**预期效果**：所有实验组在相同终止条件下产出相同或相近的层数，差异仅体现在社区组成而非层数，确保评估公平。

### 4.6 可选改进：Path A/B 重构

当前 Path A（文档内同名实体消解边）在 v4 实验中完全无效（[1] 和 [2] 指标完全相同），原因是这些边连接的节点本身就在断裂的子图中，加边后仍无法被 Leiden 的 local moving 有效利用。在 v5 中，Path A 的功能被 EdgeSchedule 的 Level 2（同文档跨段落边）吸收，建议将 Path A 重构为 EdgeSchedule 的一个 level，统一管理所有非句内边。

Path B（噪声过滤）是独立的预处理优化，与社区发现算法正交，可保留但不纳入 EdgeSchedule。

---

## 五、实验设计

### 5.1 消融实验分组

v5 设计六组消融实验，逐步验证每个改进的贡献：

**组 [0] 旧基线**：λ=0，无分层加边，sent_id 锚点。复现 v4 的 [0] 作为起始对照。

**组 [1] 仅改锚点**：λ=1000，无分层加边，para_id 锚点。验证修改锚点粒度后结构熵是否非零，但由于图仍是断裂森林，预期 Leiden 仍只能在句内合并，改善有限。

**组 [2] 仅分层加边**：λ=1000，分层加边，sent_id 锚点。验证分层加边是否允许跨句合并，但由于锚点仍是 sent_id，δH 对所有候选社区仍相等，预期 λ 不影响节点分配。

**组 [3] 完整方案**：λ=1000，分层加边，para_id 锚点。这是 v5 的核心方案，预期结构熵非零且 λ 真正影响合并节奏。

**组 [4] 完整方案 + Path A**：λ=1000，分层加边，para_id 锚点，加入文档内消解边。验证 Path A 在新架构下的效果。

**组 [5] 完整方案 + 跨文档边**：[4] + EdgeSchedule 包含跨文档同名实体边。验证跨文档边的影响。

> 注：此六组设计与 `run_multihop_eval.py` 中的 RunConfig 一一对应，详见代码。

### 5.2 评估指标

**检索质量指标（主要）**：MRR、P@K（K=1,5,10）、R@K（K=5,10）、F1@5、NDCG@K（K=5,10）。

**社区质量指标（诊断）**：平均结构熵（验证是否非零）、模块度 Q、各层平均社区大小、Level 0 物理纯净率。

**渐进合并验证指标（新增）**：层间社区物理组成分析——统计每层社区中"同段落节点占比"和"跨文档节点占比"，验证低层同段落占比高、高层跨文档占比增加。合并顺序追踪——记录每次 `_move_node` 时节点的物理距离类型（same_para / same_doc / cross_doc），绘制随层次变化的频率曲线。

**效率指标**：社区发现耗时、检索平均延迟（ms）、平均 context token 数。

### 5.3 评估数据

继续使用 MultiHop-RAG 数据集。为提升结果可靠性，考虑以下扩展：将采样量从 200 条扩大到全量 2556 条（或至少 500 条），按问题类型分层采样确保各类型覆盖均衡，重复实验 3 次取平均值以降低随机性影响。

### 5.4 基线对比

**必须完成**的对比：v5 各组之间的消融对比，以及与 Naive RAG 基线的对比（当前 Naive RAG 大幅领先，v5 的首要目标是缩小差距）。

**争取完成**的对比：与 Microsoft GraphRAG 官方版的头对头对比。这需要完成 609 篇文章的全量索引（需要 OpenAI API 费用），但对论文的完整性至关重要。若 API 成本过高，可考虑在 20 篇子集上做初步对比。

---

## 六、实施计划

### 6.1 阶段划分

整个 v5 开发分为四个阶段：

**阶段 A：核心算法修复（预计 3-5 天）** ✅ 已完成

任务 A1 是修改物理锚点粒度。在 `extractor.py` 中为 Entity 增加 `doc_id` 和 `para_id` 属性（从 sent_id 解析）。在 `graphrag_workflow.py` 的 `build_physical_nodes_from_graphrag` 中将 chunk_ids 从 sent_id 改为 para_id，并增加 `anchor_granularity` 配置开关。单元测试验证同段落节点共享 chunk_ids。

任务 A2 是修正终止条件。在 `leiden_constrained.py` 中移除 `if lambda_val < 1e-6: break`，保留收敛性终止条件（社区数不变或 ≤1）和安全上限。验证 λ=0 和 λ=1000 在新终止条件下产出相同或相近层数。

任务 A3 是验证结构熵非零。在修改后的系统上运行小规模测试，确认 `compute_structural_entropy` 返回 > 0 的值，确认 δH 对不同候选社区产生不同的值。

**阶段 B：分层加边实现（预计 3-5 天）** ✅ 已完成

任务 B1 是实现 `EdgeSchedule` 类。新建 `constrained_leiden/edge_scheduler.py`，实现三个辅助函数：`_build_intra_paragraph_edges`（同段落跨句边）、`_build_intra_document_edges`（同文档跨段落边）、`_build_cross_document_edges`（跨文档边），以及 `EdgeSchedule` 类的 `build` 和 `get_edges_for_level` 方法。

任务 B2 是接入主循环。修改 `hierarchical_leiden_constrained` 接受 `edge_schedule` 参数，在每层开始前注入新边。实现 `original_to_current` 映射表的维护和更新逻辑。

任务 B3 是重构 Path A。将现有的 `build_intra_doc_entity_edges` 整合进 EdgeSchedule，统一所有非句内边的管理。

**阶段 C：实验与评估（预计 5-7 天）** ⏳ 待执行

任务 C1 是更新实验脚本。修改 `run_multihop_eval.py`，实现六组 RunConfig，增加渐进合并验证指标的收集逻辑（层间物理组成分析、合并顺序追踪）。

任务 C2 是运行六组消融实验。在 200 条采样上运行，记录完整指标，特别关注结构熵是否非零、渐进合并是否按预期发生。

任务 C3 是分析与调参。根据实验结果调整分层加边的权重（0.3/0.2/0.1）、退火参数（decay_rate、λ_init）和锚点粒度策略。

任务 C4 是扩大评估规模。将采样量扩大到 500 条或全量 2556 条，多次运行取平均，确认结论的稳健性。

**阶段 D：文档与收尾（预计 2-3 天）** ⏳ 待执行

任务 D1 是更新 README.md，替换过时的 v1 实验数据为 v5 结果。任务 D2 是更新 PROJECT_STATUS.md，记录 v5 的完整实验结论。任务 D3 是更新 CHANGELOG.md，添加 v5 版本条目。任务 D4 是代码清理——移除调试用的 probe 脚本、统一配置管理、补充类型注释和文档字符串。

### 6.2 关键里程碑

| 里程碑 | 验收标准 | 预计时间点 |
|--------|---------|-----------|
| M1：结构熵非零 | `compute_structural_entropy` 对 para_id 锚点的社区返回 > 0 | 阶段 A 完成 |
| M2：λ 真实影响节点分配 | 相同图结构下，λ=1000 和 λ=0 产生不同社区组成（而非仅层数不同） | 阶段 A 完成 |
| M3：渐进合并可观测 | Level 0 社区主要由同段落节点组成，Level 2+ 开始包含跨文档节点 | 阶段 B 完成 |
| M4：评估公平 | 所有实验组层数由收敛性决定，不被 λ 阈值截断 | 阶段 B 完成 |
| M5：指标提升归因正确 | P@5 等提升来自社区质量差异，可通过消融实验明确归因 | 阶段 C 完成 |
| M6：文档完整 | README、PROJECT_STATUS、CHANGELOG 全部更新至 v5 | 阶段 D 完成 |

### 6.3 风险与应对

**风险 1：渐进合并生效但检索指标无提升或下降。** 这说明更好的社区结构不一定转化为更好的检索效果，问题可能出在检索阶段的 U-Retrieval 策略上。应对：单独评估社区质量指标，分析社区结构改善是否被检索策略的不足所掩盖；考虑优化 Top-Down 检索的层次导航逻辑。

**风险 2：跨文档边的数量爆炸导致图过于稠密。** 同名实体在大规模语料中可能有大量跨文档实例。应对：限制每个实体名的最大跨文档边数（`max_cross_doc_edges`），使用链式连接替代全连接，或引入基于 TF-IDF 相似度的边过滤。

**风险 3：sent_id 格式不一致导致 para_id 解析失败。** 如果上游数据源修改了 sent_id 格式，`rsplit("-s", 1)[0]` 的解析逻辑可能失效。应对：在解析函数中增加格式校验（正则 `r"^.+-p\d{3}-s\d{3}$"`），解析失败时回退到 sent_id 作为 chunk_id 并输出警告日志。

**风险 4：Naive RAG 持续大幅领先。** 当前 Naive RAG 的 MRR 0.6389 远超本项目的 0.3552，差距巨大。如果 v5 改进后仍无法显著缩小差距，需要反思 GraphRAG 的社区层次结构是否适合 MultiHop-RAG 这类数据集，或者问题是否出在检索策略和实体抽取质量上。应对：分析 Naive RAG 在哪些问题类型上领先（当前 comparison_query 的 MRR=0.7143 最高），针对性改进。

**风险 5：Microsoft GraphRAG 官方对比的 API 成本。** 609 篇文章的全量索引需要大量 OpenAI API 调用（实体抽取 + 社区摘要）。应对：先在 50-100 篇子集上做初步对比，评估成本后决定是否全量运行；或使用开源 LLM（如 Llama）替代 OpenAI API 降低成本。

---

## 七、需修改文件清单

| 优先级 | 文件路径 | 修改内容 | 所属阶段 |
|--------|---------|---------|---------|
| P0 | `extraction/extractor.py` | Entity 增加 `doc_id`/`para_id` 属性（从 sent_id 解析） | A1 |
| P0 | `constrained_leiden/graphrag_workflow.py` | `build_physical_nodes_from_graphrag` 中 chunk_ids 改为 para_id，增加 `anchor_granularity` 配置 | A1 |
| P0 | `constrained_leiden/leiden_constrained.py` | 移除 `lambda_val < 1e-6` 终止条件 | A2 |
| P1 | `constrained_leiden/edge_scheduler.py`（新建） | 实现 `EdgeSchedule` 类和三级分层加边逻辑 | B1 |
| P1 | `constrained_leiden/leiden_constrained.py` | `hierarchical_leiden_constrained` 接入 `edge_schedule` 参数，维护节点映射表 | B2 |
| P1 | `constrained_leiden/graphrag_workflow.py` | 重构 Path A 进 EdgeSchedule，新增 `build_intra_paragraph_edges` 等函数 | B3 |
| P2 | `constrained_leiden/leiden_constrained.py` | 聚合阶段可选动态锚点粒度切换（para→doc） | A/B 高级 |
| P2 | `experiments/run_multihop_eval.py` | 更新为六组消融实验，增加渐进合并验证指标 | C1 |
| P2 | `evaluation/evaluator.py` | 增加层间物理组成分析和合并顺序追踪指标 | C1 |
| P2 | `config.yaml` | 增加 `anchor_granularity`、`use_edge_schedule`、各级边权重等配置项 | A/B |
| P3 | `README.md` | 替换过时的 v1 实验数据为 v5 结果 | D1 |
| P3 | `PROJECT_STATUS.md` | 记录 v5 改进方案和实验结论 | D2 |
| P3 | `CHANGELOG.md` | 添加 v5 版本条目 | D3 |

---

## 八、成功标准

v5 的成功需要同时满足以下标准：

**底线目标（必须达成）**：结构熵在 para_id 锚点下非零（所有实验组的平均 H_structure > 0）；λ 在节点分配决策中有真实影响力（相同图结构下 λ=1000 和 λ=0 产生不同社区组成）；渐进合并可观测（低层社区以同段落节点为主，高层社区逐步跨越文档边界）；评估公平（所有实验组层数由收敛性决定，不被 λ 截断）。

**期望目标（努力达成）**：v5 最优组在 P@5、MRR、NDCG@10 上均优于 v4 的 [1]（λ=1000），且提升可通过消融实验明确归因于结构熵约束而非搜索空间差异；与 Naive RAG 的差距显著缩小（P@5 差距从当前的 0.0935 缩小到 0.05 以内）。

**理想目标（争取达成）**：v5 在部分指标上接近或超过 Naive RAG；完成与 Microsoft GraphRAG 官方版的头对头对比，在多跳推理类问题上展现 GraphRAG 的结构化优势。

---

## 九、长期展望

### 9.1 LLM 集成

当前系统完全依赖 spaCy 规则抽取，实体和关系的质量受限。未来可在以下环节引入 LLM：实体抽取阶段使用 LLM 替代 spaCy 依存句法，提升三元组的准确性和覆盖度；社区摘要阶段让 LLM 为每个社区生成自然语言摘要，供检索阶段使用（这也是 Microsoft GraphRAG 的核心设计之一）；查询理解阶段使用 LLM 对用户查询做改写或分解，提升多跳推理的检索命中率。

### 9.2 检索策略优化

当前 U-Retrieval 的 Top-Down 和 Bottom-Up 两路完全独立，融合策略是简单的交替合并。未来可探索：基于查询类型动态调整两路权重（如 comparison_query 更依赖 Top-Down 的全局视野，temporal_query 更依赖 Bottom-Up 的精确定位）；引入重排序模型对候选上下文做二次排序；利用社区层次结构做动态深度搜索而非固定层数遍历。

### 9.3 可扩展性

当前实验在 200 条 QA / 609 篇文章的小规模数据上进行。扩展到更大规模时需要关注：分层加边在大规模语料下的边数控制（可能需要引入 LSH 或近似最近邻加速同名实体匹配）；层次化 Leiden 的计算效率（当前已有增量熵状态优化，但大规模图上可能需要并行化）；索引和检索的实时性（当前是离线构建 + 在线检索的模式，可考虑增量更新）。
