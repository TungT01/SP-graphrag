# Changelog

本文件记录 GraphRAG Improved 项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v5] — 2026-04-27 — 渐进合并修复：三个根本缺陷的修正

> **状态：代码已完成，六组消融实验待运行。**

### 背景与动机

v4 消融实验揭示了三个根本缺陷：(1) 图是断裂的"句子森林"，Leiden 无法跨句合并；(2) 结构熵恒为 0（sent_id 唯一，δH 对所有候选社区相等）；(3) λ 通过终止条件控制层数而非合并节奏。P@5 +23.2% 的提升实际来自层数差异（1 层 vs 3 层），而非结构熵约束。v5 逐一修复这三个缺陷。

### Added

#### `constrained_leiden/edge_scheduler.py`（新建，约 430 行）
- 实现 `EdgeSchedule` 类：分层边注入调度器
- Level 0：无额外边，仅保留原始句内 co_occurs 边
- Level 1：注入同段落跨句边（`_build_intra_paragraph_edges`，weight=0.3）
- Level 2：注入同文档跨段落边（`_build_intra_document_edges`，weight=0.2）
- Level 3：注入跨文档同名实体边（`_build_cross_document_edges`，weight=0.1）
- 实体匹配按 `title.lower()` 分组，同层内采用链式连接避免 O(n²)
- 维护 `original_to_current` 映射表，支持超级节点 ID 映射

### Changed

#### `extraction/extractor.py`
- Entity 新增 `doc_id` 和 `para_id` 属性（从 sent_id 解析）
- `doc_id`：`sent_id.rsplit("-p", 1)[0]`（12 位 hex 哈希）
- `para_id`：`sent_id.rsplit("-s", 1)[0]`（格式 `{doc_id}-p{NNN}`）

#### `constrained_leiden/graphrag_workflow.py`
- `build_physical_nodes_from_graphrag` 新增 `anchor_granularity` 参数（`"para"` | `"sent"` | `"doc"`）
- 默认 `"para"`：`chunk_ids = frozenset([para_id])`，同段落节点共享锚点
- `"sent"` 时行为与 v4 完全一致（向后兼容）
- `run_constrained_community_detection` 新增 `anchor_granularity` 和 `use_edge_schedule` 参数

#### `constrained_leiden/leiden_constrained.py`
- `hierarchical_leiden_constrained` 新增 `edge_schedule` 可选参数
- 每层开始前调用 `edge_schedule.get_edges_for_level(level)` 注入新边
- 移除 `if lambda_val < 1e-6: break` 终止条件，层数由收敛性自然决定

#### `experiments/run_multihop_eval.py`
- 实验从四组扩展为六组（RunConfig）
- 新增 `--groups` CLI 参数，支持选择性运行指定组别
- 六组消融设计：[0] Baseline → [1] 仅改锚点 → [2] 仅分层加边 → [3] 完整方案 → [4] +Path A → [5] +跨文档边

### Fixed
- **结构熵恒为 0**：锚点从 sent_id 改为 para_id，同段落合并时 H 不增，跨段落合并时 H 增加
- **图碎片化**：EdgeSchedule 逐层注入跨句/跨段/跨文档边，解决 24,858 个断裂连通分量问题
- **终止条件语义错误**：λ→0 现在意味着"约束消失"而非"停止迭代"，确保评估公平

---

## [v4] — 2026-04-15 — 消融验证：发现三个根本缺陷

### 背景与动机

在 v3 物理优先架构上运行四组消融实验，验证结构熵约束的实际效果。实验发现 P@5 +23.2%，但深入分析后确认这是层数差异所致，而非结构熵约束。

### Added
- `experiments/run_multihop_eval.py`：四组消融实验框架
- `baselines/naive_rag/`：Naive RAG 基线（sentence-transformers + TF-IDF 余弦相似度）
- Path A（文档内同名实体消解边，weight=0.5）
- Path B（噪声过滤，150+ 停用词）

### Results（数据来源：`experiments/results/multihop_results_n200.json`，169 条有效 QA）

四组消融：

| 指标 | [0] λ=0 基线 | [1] λ=1000 | [2] +Path A | [3] +Path A+B |
|------|:-----------:|:----------:|:-----------:|:-------------:|
| MRR | 0.3492 | 0.3552 | 0.3552 | 0.3552 |
| P@5 | 0.1325 | 0.1633 (+23.2%) | 0.1633 | 0.1633 |
| 层数 | 1 | 3 | 3 | 3 |
| 社区数 | 22,973 | 63,497 | 63,497 | 63,497 |
| 结构熵 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

与 Naive RAG 对比：MRR 0.6389 vs 0.3552，P@5 0.2568 vs 0.1633，全面落后约 40%。

### Key Findings
- **结构熵全为 0**：sent_id 唯一导致 δH 对所有候选社区相等，λ 无实际影响力
- **P@5 提升来自层数差异**：λ=0 只跑 1 层（终止条件 `lambda_val < 1e-6: break`），λ=1000 跑 3 层
- **Path A 无效**：[1] 和 [2] 指标完全相同
- **Path B 无额外效果**：在此实验配置下 [2] 和 [3] 指标完全相同

---

## [v3] — 2026-04-14 — 架构重设计：物理优先 + 实例级节点

### 背景与动机

原版实现存在一个根本性的架构问题：实体在抽取阶段就被合并（实体消解），
所有文档里的"阿司匹林"变成同一个节点，结构熵约束只能在聚类阶段"软性"
阻止跨文档合并，物理信息（chunk_id）仅作为节点属性存在，不是图结构本身。

新架构的核心思想：**先物理、后语义。物理结构是一等公民，语义从物理结构中涌现。**

### 架构变更概览

#### 节点定义变更
- **旧**：实体节点，ID = `md5(title)`，携带 `primary_chunk_id` 属性
- **新**：实体实例节点，ID = `{sent_id}-{entity_name_normalized}`，携带完整物理路径

同一实体在不同句子里出现 = 不同节点，底层图保持物理纯净。

#### 边的定义变更
- **旧**：跨 chunk 的共现边，所有文档里共现过的实体之间都有边
- **新**：
  - 语义边：spaCy 依存句法提取的三元组 `(主语, 谓词, 宾语)`，主宾必须在同一句子内
  - 物理结构边：同句内的两个实体节点之间，权重 1.0
  - 跨句/跨段/跨文档：**无任何预设边**，连通性完全由聚类过程产生

底层图是由若干孤立的句子级子图组成的森林。

#### 实体消解变更
- **旧**：图构建阶段预处理，字符串匹配合并同名实体
- **新**：不在图构建阶段做，由 λ 退火驱动的 Leiden 聚类在社区层面涌现

#### 三元组提取变更
- **旧**：规则（正则匹配大写词）+ 共现窗口
- **新**：spaCy 依存句法分析，提取真正的主谓宾三元组
- 代词处理：主语或宾语为代词（PRP/PRP$）时直接跳过（方案一）
- 备选升级：若召回率不理想，切换至共指消解方案（coreferee/neuralcoref）

#### 句子切分变更
- **旧**：正则按标点切分
- **新**：spaCy `doc.sents`（依存句法树判断边界，处理缩写/引号/数字等边界歧义）

### Changed

#### `data/ingestion.py`
- 新增三级物理结构：`Document → Paragraph → Sentence`
- 新增 `SentenceUnit` 数据类，携带 `sent_id`（格式：`{doc_id}-p{NNN}-s{NNN}`）
- `TextUnit` 升级为段落级容器，包含其下所有 `SentenceUnit`
- ID 生成规则：文档 ID 基于路径 hash，段落/句子 ID 基于序号

#### `extraction/extractor.py`
- 移除规则后端的共现窗口关系抽取
- 移除实体消解（`_disambiguate_entities`）
- 新增 spaCy 依存句法三元组提取后端
- 实体节点 ID 改为 `{sent_id}-{entity_name_normalized}`，包含完整物理路径
- `Entity.primary_chunk_id` 改为 `Entity.sent_id`（精确到句子级）

#### `constrained_leiden/graphrag_workflow.py`
- `build_graph_from_graphrag`：边构建逻辑重写，只在同句内建边
- `build_physical_nodes_from_graphrag`：物理锚点改为 `sent_id`（句子级）
- 移除 `min_edge_weight` 过滤（原用于过滤低频共现，新方案不适用）

#### `constrained_leiden/physical_anchor.py`
- `PhysicalNode.chunk_ids` 语义变更：由 chunk_id 集合改为 sent_id 集合
- 结构熵计算对象：社区内节点的句子 ID 分布熵（更细粒度的物理纯净度）

#### `retrieval/retriever.py`
- `BottomUpRetriever`：物理锚点检索改为 sent_id 级别定位
- 检索结果携带完整物理路径，支持精确溯源到原始句子

#### `experiments/run_experiment.py`
- `corpus_to_pipeline_text_units`：适配新的三级 ID 结构
- 评估时 ground-truth 对齐改为 doc_id 级别（MultiHop-RAG 的 evidence 是文章级）

---

## [v1.1.0] — 2026-04-14 — primary_chunk_id 修复 + 增量熵优化

### 背景
原版物理锚点使用 `text_unit_ids`（实体出现过的所有文档），导致高频实体
天然跨多个 chunk，结构熵无法降低，物理约束形同虚设。

### Changed
- `extraction/extractor.py`：新增 `primary_chunk_id` 字段，用 `Counter.most_common`
  确定主锚点（出现频次最高的 chunk）
- `constrained_leiden/graphrag_workflow.py`：`build_physical_nodes_from_graphrag`
  改为优先使用 `primary_chunk_id` 单一锚点
- `constrained_leiden/leiden_constrained.py`：新增 `CommunityEntropyState`，
  将 ΔH 计算从 O(|community|) 降到 O(1)，实测 6.8x 加速
- 移除 leidenalg 快速路径（事后拆分），统一使用纯 Python 过程约束实现

### Fixed
- 物理约束语义错误：100% 实体跨多 chunk → 修复后 Level-0 纯净率 100%

### Results
- 平均结构熵：0.8827 → 0.0000（-100%）
- Level-0 纯净率：24.73% → 100%
- MRR：+1.3%，P@5：+3.5%，NDCG@10：+1.9%

---

## [v1.0.0] — 2026-04-10 — 初始版本

### Added
- 带结构熵惩罚的 Leiden 变体：J = Q_leiden - λ·H_structure
- λ 退火机制（指数/线性/余弦/阶梯）
- U-Retrieval 双轨检索（自顶向下社区导航 + 自底向上物理锚点）
- MultiHop-RAG 对照实验框架
- 评估指标：MRR / Precision@K / Recall@K / NDCG@K / 结构熵 / 纯净率
