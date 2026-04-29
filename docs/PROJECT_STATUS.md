# GraphRAG Improved — 项目现状全面盘点

> 更新日期：2026-04-27  
> 当前版本：v5（代码已完成，实验待运行）  
> 仓库地址：https://github.com/TungT01/graphrag-improved

---

## 一、项目目标

### 研究问题

微软 GraphRAG 在构建知识图谱时，使用 Leiden 算法对实体进行社区检测，再以社区为单位做 RAG 检索。原版 Leiden 只优化模块度 Q，不关心社区内实体的物理来源（即实体来自哪些文档/段落/句子）。这导致跨文档的同名实体被错误合并进同一社区，检索时产生大量噪声。

### 核心假设

在 Leiden 目标函数中加入**结构熵惩罚项**，驱动社区向物理纯净方向演化，可以提升基于社区的 RAG 检索质量：

```
J = Q_leiden - λ · H_structure

H_structure = Σ_c  |c|/|V| · H(anchor_id 分布 in c)
```

其中 λ 采用指数退火：`λ(t) = λ_init · exp(-decay · t)`，初始值 λ_init=1000。v5 中 anchor_id 为 para_id（段落级），替代 v4 中无效的 sent_id（句子级）。

### 对比基线

| 系统 | 描述 |
|------|------|
| **Naive RAG** | TF-IDF 向量检索，sentence-transformers 编码，无图结构 |
| **GraphRAG Baseline (λ=0)** | 原版 Leiden，无结构熵约束，v3 物理优先架构 |
| **GraphRAG Improved (λ=1000)** | 结构熵约束 Leiden，λ 指数退火 |

---

## 二、系统架构

### 2.1 整体模块

```
graphrag_improved/
├── constrained_leiden/                 # 核心算法
│   ├── leiden_constrained.py               # 结构熵约束 Leiden 主实现
│   ├── graphrag_workflow.py                # 图构建工作流（v5: anchor_granularity 参数）
│   ├── physical_anchor.py                  # 物理锚点（v5: para_id 级别）
│   ├── annealing.py                        # λ 退火调度（指数/线性/余弦/阶梯）
│   └── edge_scheduler.py                  # 分层加边调度器（v5 新增，约 430 行）
├── extraction/
│   └── extractor.py                        # spaCy 依存句法三元组抽取（v5: 新增 doc_id/para_id）
├── retrieval/
│   └── retriever.py                        # U-Retrieval 双轨检索
├── baselines/
│   ├── naive_rag/                          # TF-IDF 基线
│   └── eval_results/                       # 基线评估结果
├── experiments/
│   ├── run_multihop_eval.py                # 六组消融实验脚本（v5 扩展）
│   ├── ablation_study_notes.md             # 消融实验详细记录
│   └── results/                            # 实验结果 JSON
├── evaluation/                             # 评估指标计算
├── data/                                   # 数据集接口
├── README.md
├── CHANGELOG.md
├── PROJECT_PLAN.md
├── REFACTOR_PROMPT.md
└── config.yaml
```

### 2.2 v5 渐进合并架构（当前版本）

**核心设计原则：先物理、后语义。物理结构是一等公民，语义从物理结构中涌现。v5 在 v3 基础上修复了三处根本性缺陷。**

**节点设计**：每个节点是带物理坐标的实体实例，而非概念节点：

```
节点 ID = {doc_id}-p{NNN}-s{NNN}-{entity_name_normalized}
示例：a3f2b1c4d5e6-p003-s001-aspirin
```

同一实体名在不同句子里出现 = 不同节点，底层图保持物理纯净。

**v5 三处修复**：

1. **锚点粒度（`anchor_granularity="para"`）**：结构熵计算的 chunk_ids 从 `sent_id`（全局唯一）改为 `para_id`（同段落共享），使 δH 真正有区分度，结构熵不再恒为 0。

2. **分层加边（`EdgeSchedule`）**：新增 `edge_scheduler.py`，实现三级边注入策略——Level 1 注入段落内跨句边（weight=0.3），Level 2 注入文档内跨段落边（weight=0.2），Level 3 注入跨文档语义边（weight=0.1）。每层迭代开始前按调度注入对应边，解决图碎片化问题。

3. **终止条件修复**：移除 `if lambda_val < 1e-6: break` 的提前终止，层数由收敛性决定，确保高层语义融合能够充分展开。

**边的设计**：语义边来自 spaCy 依存句法提取的主谓宾三元组（主宾必须在同一句子内）；物理结构边为同句内实体共现（predicate = "co_occurs"，weight=1.0）；EdgeSchedule 边在运行时按层注入，不在底层图中预设。

**底层图特征**：由若干孤立的句子级子图组成的森林，总连通分量数约 24,858，最大连通分量仅 9 个节点。EdgeSchedule 在运行时注入边以逐层提升连通性。

### 2.3 U-Retrieval 双轨检索

**自顶向下**：从高层社区摘要出发，逐层导航到相关子社区。**自底向上**：通过物理锚点（sent_id）直接定位原始句子，再向上聚合。

---

## 三、评估数据集

**MultiHop-RAG**（COLM 2024）

| 属性 | 数值 |
|------|------|
| 文章数量 | 609 篇新闻文章 |
| QA 对总数 | 2,556 条 |
| 问题类型 | inference_query / comparison_query / temporal_query / null |
| 证据分布 | 每条 QA 的证据分布在 2-4 篇文档中 |
| v4 实验规模 | 200 QA 采样（有效 169 条） |

---

## 四、v4 实验结果（已完成）

> **重要说明**：v4 实验存在两组不同数据，来自不同实体集。下文明确标注每组数据的来源。v5 实验结果待运行后更新。

### 4.1 v4 消融实验设计（四组）

| 组号 | 名称 | λ_init | Path A | Path B | 实体集 |
|------|------|--------|--------|--------|--------|
| [0] | Baseline (λ=0) | 0.0 | ✗ | ✗ | 见下文 |
| [1] | Ours (λ=1000) | 1000.0 | ✗ | ✗ | 见下文 |
| [2] | Ours+A | 1000.0 | ✓ | ✗ | 见下文 |
| [3] | Ours+A+B | 1000.0 | ✓ | ✓ | 见下文 |

### 4.2 检索质量指标

**数据集 A**（来源：项目开发过程中的一次实验运行，[0]-[2] 使用未过滤的 60,439 个实体 / 84,704 条关系，[3] 使用噪声过滤后的 47,142 个实体 / 61,812 条关系）：

| 指标 | [0] Baseline | [1] Ours | [2] Ours+A | [3] Ours+A+B |
|------|:---:|:---:|:---:|:---:|
| MRR | 0.3492 | 0.3558 (+1.9%) | 0.3558 (+1.9%) | 0.3552 (+1.7%) |
| Precision@1 | 0.2663 | 0.2663 (±0%) | 0.2663 (±0%) | 0.2663 (±0%) |
| Precision@3 | 0.1617 | 0.1617 (±0%) | 0.1617 (±0%) | 0.1617 (±0%) |
| **Precision@5** | **0.1325** | **0.1609 (+21.4%)** | **0.1609 (+21.4%)** | **0.1633 (+23.2%)** |
| Precision@10 | 0.0882 | 0.0882 (±0%) | 0.0882 (±0%) | 0.0882 (±0%) |
| **Recall@5** | **0.2648** | **0.3180 (+20.1%)** | **0.3180 (+20.1%)** | **0.3230 (+22.0%)** |
| Recall@10 | 0.3481 | 0.3481 (±0%) | 0.3481 (±0%) | 0.3481 (±0%) |
| F1@5 | 0.1744 | 0.2110 (+21.0%) | 0.2110 (+21.0%) | 0.2141 (+22.8%) |
| **NDCG@5** | **0.2347** | **0.2654 (+13.1%)** | **0.2654 (+13.1%)** | **0.2680 (+14.2%)** |
| NDCG@10 | 0.2737 | 0.2789 (+1.9%) | 0.2789 (+1.9%) | 0.2792 (+2.0%) |

**数据集 B**（来源：`experiments/results/multihop_results_n200.json`，四组均使用噪声过滤后的 47,142 个实体 / 61,812 条关系）：

| 指标 | [0] Baseline | [1] Ours | [2] Ours+A | [3] Ours+A+B |
|------|:---:|:---:|:---:|:---:|
| MRR | 0.3492 | 0.3552 | 0.3552 | 0.3552 |
| P@5 | 0.1325 | 0.1633 (+23.2%) | 0.1633 | 0.1633 |
| NDCG@10 | 0.2737 | 0.2792 | 0.2792 | 0.2792 |
| 层数 | 1 | 3 | 3 | 3 |
| 社区数 | 22,973 | 63,497 | 63,497 | 63,497 |
| 结构熵 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

> **两组数据的差异说明**：数据集 A 中 [0]-[2] 使用未过滤实体集（60,439），[3] 使用过滤后实体集（47,142），因此 [1] 的社区数为 79,114、P@5 为 0.1609（+21.4%）。数据集 B 中四组统一使用过滤后实体集，因此 [1] 的社区数为 63,497、P@5 为 0.1633（+23.2%）。后续文档中引用 v4 数据时，统一以数据集 B（JSON 文件）为权威来源。

### 4.3 图结构与社区质量指标（数据集 A）

| 指标 | [0] Baseline | [1] Ours | [2] Ours+A | [3] Ours+A+B |
|------|:---:|:---:|:---:|:---:|
| 实体数量 | 60,439 | 60,439 | 60,439 | 47,142 |
| 关系数量 | 84,704 | 84,704 | 84,704 | 61,812 |
| 社区数量 | 29,398 | 79,114 | 79,113 | 63,497 |
| 层次数量 | 1 | 3 | 3 | 3 |
| 模块度 Q | 0.7533 | 0.7533 | 0.7535 | **0.8025 (+6.5%)** |
| 平均结构熵 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Level-0 纯净率 | 100% | 100% | 100% | 100% |
| 平均社区大小 | 2.06 | 2.29 | 2.29 | 2.23 |

### 4.4 运行时间

| 组号 | 社区检测耗时 | 检索评估耗时 | 总耗时 |
|------|:---:|:---:|:---:|
| [0] Baseline | 97.6s | 23.9s | 121.5s |
| [1] Ours | 222.0s | 50.6s | 272.6s |
| [2] Ours+A | 205.9s | 50.0s | 255.9s |
| [3] Ours+A+B | 191.5s | 38.0s | 229.5s |

路径B 额外抽取耗时：348.7s（仅首次，后续命中缓存）。

### 4.5 与 Naive RAG 基线对比（n=200，valid=169）

| 指标 | Naive RAG | [3] Ours+A+B (数据集 B) | 差距 |
|------|:---:|:---:|:---:|
| MRR | **0.6389** | 0.3552 | -44.4% |
| Precision@5 | **0.2568** | 0.1633 | -36.4% |
| Recall@5 | **0.5059** | 0.3230 | -36.2% |
| NDCG@5 | **0.4792** | 0.2680 | -44.1% |
| avg_latency_ms | 70.6ms | ~1600ms | ~23x 慢 |

Naive RAG 按问题类型细分：

| 问题类型 | n | MRR | Recall@5 |
|---------|---|-----|---------|
| comparison_query | 57 | 0.714 | 0.652 |
| inference_query | 59 | 0.619 | 0.347 |
| temporal_query | 53 | 0.580 | 0.525 |

---

## 五、关键发现与结论（v4）

### 5.1 主线结论：P@5 +23.2% 的真实原因

[0] vs [1] 的对比显示 P@5 +23.2%（数据集 B），但这主要来自层次数不同（1 层 vs 3 层），而非结构熵约束本身。证据：所有组平均结构熵均为 0.0000，说明 sent_id 粒度下结构熵完全无效。

### 5.2 v5 修复的必要性

v4 实验揭示了三个根本性缺陷：

**缺陷一：结构熵恒为 0**。sent_id 粒度下每节点 sent_id 唯一，社区内熵恒为 0，结构熵惩罚项完全失效。v5 改用 para_id 粒度修复。

**缺陷二：图碎片化**。v3 底层图约 24,858 个连通分量，跨句子/跨段落无任何边，导致高层语义融合无法展开。v5 通过 EdgeSchedule 分层加边修复。

**缺陷三：提前终止**。`if lambda_val < 1e-6: break` 导致高层迭代被跳过，层次数不由收敛性决定。v5 移除该条件修复。

### 5.3 Path A 和 Path B 的效果（v4 结论）

**Path A（文档内消解边）**：在数据集 A 中 [1] 和 [2] 指标差异极小（P@5 均为 0.1609），在数据集 B 中完全相同。图碎片化使得加边无法被 Leiden 有效利用。

**Path B（噪声过滤）**：减少了 22% 实体和 27% 关系，模块度 Q 提升 6.5%（数据集 A 中 [1] vs [3]）。但在数据集 B（统一过滤后实体集）中 [1]-[3] 指标相同。噪声过滤本身有益，但对检索指标的独立贡献难以隔离。

### 5.4 与 Naive RAG 的差距

当前 GraphRAG Improved 在所有指标上均落后于 Naive RAG（TF-IDF），差距约 36-44%。这是 GraphRAG 类方法的已知问题：社区级检索的粒度较粗，在精确文档检索任务上不如直接向量检索。

---

## 六、v5 消融实验设计（待运行）

### 6.1 六组消融

| 组号 | 名称 | λ_init | anchor | EdgeSchedule | 验证目标 |
|:----:|------|:------:|:------:|:------------:|---------|
| [0] | Baseline | 0.0 | sent | ✗ | 起始对照（复现 v4 的 [0]） |
| [1] | 仅改锚点 | 1000.0 | **para** | ✗ | 验证锚点修复后结构熵是否非零 |
| [2] | 仅分层加边 | 1000.0 | sent | **✓** | 验证 EdgeSchedule 是否解决图碎片化 |
| [3] | 完整方案 | 1000.0 | **para** | **✓** | v5 核心方案，预期结构熵非零且 λ 真正影响合并 |
| [4] | +Path A | 1000.0 | **para** | **✓** | 加入文档内消解边 |
| [5] | +跨文档边 | 1000.0 | **para** | **✓**(含跨文档) | 加入跨文档同名实体边 |

消融逻辑：[0]→[1] 验证锚点粒度修复；[0]→[2] 验证分层加边；[1]+[2]→[3] 验证两者结合；[3]→[4] 验证 Path A；[4]→[5] 验证跨文档边。

### 6.2 验证指标（v5 新增）

除标准检索指标外，新增以下诊断指标：结构熵非零验证（para_id 锚点的 H_structure 应 > 0）；层间社区物理组成分析（低层同段落占比高 → 高层跨文档占比增加）；合并顺序追踪（`_move_node` 时记录节点物理距离类型分布随层次的变化）。

---

## 七、已知问题与待解决事项

### 7.1 检索质量问题

**问题 1：与 Naive RAG 差距显著** — 当前 MRR、P@5、NDCG@5 均落后约 40%。根本原因：社区级检索粒度过粗。待解决：改进 U-Retrieval 的文档级聚合逻辑，或引入混合检索。

**问题 2：@10 窗口提升有限** — 结构熵约束主要改善 @5 窗口，@10 窗口提升不足 2%。待解决：改进社区排序算法，引入重排序。

**问题 3：inference_query 类型表现最差** — Naive RAG 在 inference_query 上 Recall@5 仅 0.347。待解决：针对多跳推理设计检索策略。

### 7.2 图结构问题（v5 已修复代码，待实验验证）

**问题 4：图碎片化** — v5 通过 EdgeSchedule 分层加边修复，待运行实验验证效果。

**问题 5：结构熵恒为 0** — v5 通过 para_id 锚点修复，待运行实验验证 H > 0。

### 7.3 评估问题

**问题 6：评估规模不足** — 200 QA 采样（169 有效）统计功效有限。待解决：在全量 2556 QA 上复现。

**问题 7：缺少 GraphRAG 官方基线** — 待完成 GraphRAG 3.0.9 CLI 的全量评估。

### 7.4 工程问题

**问题 8：运行时间较长** — Ours 总耗时 272.6s（vs Baseline 121.5s），慢 2.2x。

---

## 八、版本历史

| 版本 | 日期 | 主要变更 | 状态 |
|------|------|---------|------|
| v5 | 2026-04-27 | 三处根本修复：锚点粒度(sent→para) + EdgeSchedule 分层加边 + 终止条件修复；六组消融设计 | **代码完成，实验待运行** |
| v4 | 2026-04-15 | 四组消融实验；发现三个根本缺陷（结构熵≡0、图碎片化、终止条件语义错误） | 已完成 |
| v3 | 2026-04-14 | 物理优先架构重设计：实例级节点、移除实体消解、spaCy 三元组抽取 | 已完成 |
| v1.1 | 2026-04-14 | primary_chunk_id 修复 + 增量熵 O(1) 优化（6.8x 加速） | 已完成 |
| v1.0 | 2026-04-10 | 初始版本：结构熵约束 Leiden + λ 退火 + U-Retrieval + MultiHop-RAG 评估 | 已完成 |

---

## 九、复现指南

### 环境准备

```bash
cd /Users/ttung/Desktop/个人学习/graphrag_improved
python3 -m venv .venv-graphrag
source .venv-graphrag/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 运行实验

```bash
# 快速验证（200 QA，六组消融，约 30-60 分钟）
python3 -u -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --n-qa 200 \
    --output-dir ./experiments/results

# 只运行指定组别（如只跑 Baseline 和 v5 核心组）
python3 -u -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --n-qa 200 \
    --groups 0,3

# 全量运行（2556 QA，约 4-6 小时）
python3 -u -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --full \
    --output-dir ./experiments/results
```

### 缓存管理

```
experiments/results/cache/
├── entities_full.parquet        # 原始抽取（共用）
├── relationships_full.parquet
├── entities_full_b.parquet      # 噪声过滤后
└── relationships_full_b.parquet
```

### 关键代码位置

| 功能 | 文件 | 关键符号 |
|------|------|---------|
| 结构熵约束 Leiden | `constrained_leiden/leiden_constrained.py` | `hierarchical_leiden_constrained()` |
| λ 退火调度 | `constrained_leiden/annealing.py` | `AnnealingSchedule` |
| 图构建工作流 | `constrained_leiden/graphrag_workflow.py` | `build_graph_from_graphrag()` |
| **分层加边调度（v5）** | `constrained_leiden/edge_scheduler.py` | `EdgeSchedule` |
| **锚点粒度控制（v5）** | `constrained_leiden/graphrag_workflow.py` | `anchor_granularity` 参数 |
| spaCy 三元组抽取 | `extraction/extractor.py` | `_STOPWORDS`, `extract_entities()` |
| 文档内消解边（Path A） | `constrained_leiden/graphrag_workflow.py` | `build_intra_doc_entity_edges()` |
| U-Retrieval 检索 | `retrieval/retriever.py` | `BottomUpRetriever` |
| **六组消融实验入口（v5）** | `experiments/run_multihop_eval.py` | `run_experiment()`, `--groups` |
| Naive RAG 基线 | `baselines/naive_rag/` | — |
