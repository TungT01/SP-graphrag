# GraphRAG Improved

基于**结构熵约束**的层次化 Leiden 社区检测，改进微软 [GraphRAG](https://github.com/microsoft/graphrag) 的社区发现与检索质量。

---

## 项目状态速览

| 项目 | 当前状态 |
|------|---------|
| 当前版本 | **v5**（代码已完成，实验待运行） |
| 核心算法 | 结构熵约束 Leiden，v5 修复了 v4 中发现的三个根本缺陷 |
| 已验证实验 | v4 四组消融（200 QA 采样），结论：P@5 +23.2% 但来自层数差异而非结构熵 |
| 待运行实验 | v5 六组消融，验证结构熵约束是否真正生效 |
| 与 Naive RAG 差距 | MRR 0.35 vs 0.64，P@5 0.16 vs 0.26 — **当前落后约 40%** |
| 评估数据集 | MultiHop-RAG (COLM 2024)，609 篇文章，2556 条 QA |

**一句话概括**：项目已完成 v5 的代码重构（修复结构熵失效、图碎片化、终止条件三个根本缺陷），但六组新消融实验尚未运行，v5 的实际效果有待验证。

---

## 核心思想

### 研究动机

微软 GraphRAG 使用标准 Leiden 算法做社区检测，纯粹依赖拓扑模块度，忽略实体在原始文档中的物理位置。本项目的假设是：**物理位置近的实体语义关联更紧密，社区发现应优先合并物理位置相近的实体。**

### 目标函数

在 Leiden 的模块度优化中引入结构熵惩罚项：

```
J = Q_leiden − λ · H_structure
```

Q 是标准模块度增益（衡量社区内连接的紧密程度），H 是社区内节点物理来源的 Shannon 熵（衡量节点来自多少不同的物理位置——段落或文档），λ 是约束强度系数。

### λ 退火机制

λ 随层级升高而衰减（支持指数/线性/余弦/阶梯四种曲线），实现渐进式合并：

```
层次 0（λ 最大）→ 优先合并物理位置相近的实体（同段落）
层次 1-2（λ 递减）→ 逐步允许跨段落、跨文档合并
层次 3+（λ → 0）→ 退化为标准 Leiden，自由合并
```

---

## 版本演进

项目经历了五个版本，从"概念验证"演进到"发现核心缺陷并修复"：

**v1.0**（初始版本）：建立完整框架——带结构熵惩罚的 Leiden、λ 退火、U-Retrieval 双轨检索、MultiHop-RAG 评估。但物理锚点使用 `text_unit_ids`（实体出现过的所有文档），高频实体天然跨多 chunk，物理约束形同虚设（Level-0 纯净率仅 24.73%）。

**v1.1**（锚点修复 + 性能优化）：修正锚点为 `primary_chunk_id`（单一主锚点），纯净率从 24.73% 修至 100%。引入增量熵状态 `CommunityEntropyState`，ΔH 计算从 O(|community|) 降至 O(1)，6.8x 加速。

**v3**（物理优先架构重设计）：重大架构变革——节点从"概念级"改为"实例级"（每句话中的实体是独立节点，ID 含完整物理路径），移除实体消解（由 Leiden 自然涌现），引入 spaCy 依存句法三元组提取替代共现窗口，新增三级 ID 切分（Document → Paragraph → Sentence）。

**v4**（消融验证）：运行四组消融实验，发现 P@5 +23.2%。但深入分析发现三个根本缺陷（详见下文"问题诊断"），提升实为层数差异所致。

**v5**（渐进合并修复，当前版本）：针对 v4 发现的三个缺陷完成代码修复，六组新消融实验待运行。

---

## 问题诊断（v4 发现的三个根本缺陷）

这是理解本项目的关键——v4 实验揭示了核心算法存在三个根本性问题，v5 逐一修复了它们。

### 缺陷一：图是断裂的"句子森林"

v3 架构中，物理结构边（`co_occurs`）严格限定在同句内，不同句子的实体之间没有任何边。初始图是 N 个互不相连的句子级子图，Leiden 永远无法把不同句子的实体合并到同一社区——因为它们之间没有路径。

**v5 修复**：新增 `EdgeSchedule` 分层加边机制——Level 1 注入同段落跨句边，Level 2 注入同文档跨段落边，Level 3 注入跨文档边，逐层扩展图的连通性。

### 缺陷二：结构熵恒为 0

v3 中每个节点的 `chunk_ids = frozenset([sent_id])`，而 sent_id 全局唯一。δH 只取决于社区大小，与节点物理来源无关，λ·δH 在比较不同候选社区时完全抵消。结果：**λ 对节点分配决策没有任何实际影响力。**

**v5 修复**：将 `chunk_ids` 从 sent_id 改为 para_id（同段落节点共享），使同段落合并时 H 不增，跨段落合并时 H 增加，λ·δH 产生真实的差异化惩罚。

### 缺陷三：λ 控制的是层数而非合并节奏

终止条件 `if lambda_val < 1e-6: break` 导致 λ=0 只跑 1 层（22,973 社区），λ=1000 跑 3 层（63,497 社区）。P@5 +23.2% 的提升实际来自搜索空间差异，不是结构熵约束的功劳。

**v5 修复**：移除该终止条件，层数由图的收敛性自然决定，确保评估公平。

---

## 已验证的实验数据（v4）

> 数据来源：`experiments/results/multihop_results_n200.json`，169 条有效 QA。

### 四组消融实验

| 指标 | [0] λ=0 基线 | [1] λ=1000 | [2] +Path A | [3] +Path A+B |
|------|:-----------:|:----------:|:-----------:|:-------------:|
| MRR | 0.3492 | 0.3552 | 0.3552 | 0.3552 |
| P@5 | 0.1325 | 0.1633 (+23.2%) | 0.1633 | 0.1633 |
| NDCG@10 | 0.2737 | 0.2792 | 0.2792 | 0.2792 |
| 层数 | 1 | 3 | 3 | 3 |
| 社区数 | 22,973 | 63,497 | 63,497 | 63,497 |
| 结构熵 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

**解读**：结构熵全为 0 证实了缺陷二；层数差异（1 vs 3）导致社区数差异（23K vs 63K），这是 P@5 提升的真实原因（缺陷三）；Path A（文档内消解边）和 Path B（噪声过滤）均无额外效果。

### 与 Naive RAG 基线对比

| 指标 | Naive RAG (TF-IDF) | GraphRAG Improved [1] | 差距 |
|------|:------------------:|:--------------------:|:----:|
| MRR | **0.6389** | 0.3552 | -44% |
| P@5 | **0.2568** | 0.1633 | -36% |
| NDCG@10 | **0.5216** | 0.2792 | -46% |

Naive RAG 全面大幅领先。缩小这一差距是后续工作的首要目标。

---

## 待运行的 v5 实验（六组消融）

v5 代码已就绪，以下六组实验待运行以验证修复效果：

| 组号 | 名称 | anchor | EdgeSchedule | 验证目标 |
|:----:|------|:------:|:------------:|---------|
| [0] | Baseline | sent | ✗ | 起始对照（复现 v4 的 [0]） |
| [1] | 仅改锚点 | **para** | ✗ | 验证锚点修复后结构熵是否非零 |
| [2] | 仅分层加边 | sent | **✓** | 验证 EdgeSchedule 是否解决图碎片化 |
| [3] | 锚点+加边 | **para** | **✓** | v5 核心方案，预期结构熵非零且 λ 真正影响合并 |
| [4] | +Path A | **para** | **✓** | 加入文档内消解边 |
| [5] | +跨文档边 | **para** | **✓**(含跨文档) | 加入跨文档同名实体边 |

运行命令：

```bash
# 运行全部六组
python -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --output-dir ./experiments/results

# 只运行指定组（如 Baseline 和 v5 核心组）
python -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag --groups 0,3
```

---

## 系统架构

### 四阶段 Pipeline

```
原始文档
  → Ingest（三级切分：Document → Paragraph → Sentence，分配 doc_id/para_id/sent_id）
  → Extract（spaCy 依存句法三元组抽取，句内 co_occurs 边，无实体消解）
  → Community Detection（结构熵约束 Leiden + λ 退火 + EdgeSchedule 分层加边）
  → Save（CSV / Parquet / HTML 报告）
```

### 项目结构

```
graphrag_improved/
├── main.py                         # CLI 入口
├── run.py                          # Pipeline 编排
├── pipeline_config.py              # 配置加载
├── config.yaml                     # 项目配置
│
├── data/ingestion.py               # 三级切分（doc → para → sent）
├── extraction/extractor.py         # spaCy 依存句法三元组抽取
├── proposition/transformer.py      # 命题转换预处理（共指消解 + 命题原子化）
│
├── constrained_leiden/             # 核心算法
│   ├── leiden_constrained.py       #   结构熵约束 Leiden（J = Q − λ·H）
│   ├── physical_anchor.py          #   PhysicalNode + 结构熵计算
│   ├── annealing.py                #   λ 退火调度（4 种曲线）
│   ├── graphrag_workflow.py        #   GraphRAG 兼容接口（支持 anchor_granularity）
│   └── edge_scheduler.py           #   分层加边调度（v5 新增）
│
├── retrieval/retriever.py          # U-Retrieval 双轨检索（TopDown + BottomUp）
├── evaluation/evaluator.py         # 三维评估（社区质量 + 检索质量 + 文本匹配）
├── output/reporter.py              # HTML 报告 + Force-directed 图可视化
│
├── experiments/
│   └── run_multihop_eval.py        # 六组消融实验脚本
├── baselines/
│   ├── naive_rag/                  # Naive RAG 基线（sentence-transformers + 余弦相似度）
│   └── graphrag_official/          # Microsoft GraphRAG 官方基线（未完成全量评估）
│
├── PROJECT_STATUS.md               # 项目现状全面盘点
├── PROJECT_PLAN.md                 # v5 改进计划书
├── REFACTOR_PROMPT.md              # v5 重构技术指引（面向大模型）
└── CHANGELOG.md                    # 版本变更日志
```

### 关键概念速查

| 概念 | 含义 |
|------|------|
| **Leiden 算法** | 社区检测算法（Louvain 的改进版），通过优化模块度 Q 将图中的节点划分为社区 |
| **模块度 Q** | 衡量社区内部连接紧密程度的指标，Q 越高社区结构越好 |
| **结构熵 H** | 本项目定义：社区内节点物理来源分布的 Shannon 熵。H=0 表示所有节点来自同一物理位置，H 越大越分散 |
| **物理锚点** | 每个节点携带的物理来源标识（v5 中为 para_id），用于计算结构熵 |
| **三级 ID** | doc_id（文档）→ para_id（段落，`{doc_id}-p{NNN}`）→ sent_id（句子，`{para_id}-s{NNN}`），NNN 为零填充 3 位序号 |
| **EdgeSchedule** | v5 分层加边机制：Level 0 仅句内边，Level 1 同段落跨句边，Level 2 同文档跨段边，Level 3 跨文档边 |
| **U-Retrieval** | 双轨检索：Top-Down 从高层社区向下导航 + Bottom-Up 通过物理锚点直接定位原始句子，结果融合 |
| **Path A** | 同文档内同名实体的消解边（soft edge, weight=0.5），连接不同句子中的同一实体 |

---

## 快速开始

### 安装

```bash
pip install networkx pandas pyyaml

# 必须：spaCy（当前唯一的抽取后端）
pip install spacy && python -m spacy download en_core_web_sm

# 可选
pip install pypdf        # PDF 支持
pip install rank-bm25    # BM25 检索后端
pip install pyarrow      # Parquet 导出
```

### 运行

```bash
# 使用内置示例数据
python -m graphrag_improved.main

# 指定数据目录
python -m graphrag_improved.main --data-dir ./my_papers

# 调整约束强度和退火曲线
python -m graphrag_improved.main --lambda-init 500 --schedule cosine
```

### 代码示例

```python
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection

# v5 推荐：para 粒度 + EdgeSchedule
communities_df = run_constrained_community_detection(
    entities_df, relationships_df,
    anchor_granularity="para",
    use_edge_schedule=True,
)
```

---

## 文档导航

| 文档 | 适合谁 | 内容 |
|------|-------|------|
| **README.md**（本文件） | 所有人 | 项目概览、状态速览、快速上手 |
| **PROJECT_PLAN.md** | 想深入理解技术方案的人 | 完整技术方案、改进设计、实验计划、实施步骤 |
| **PROJECT_STATUS.md** | 想了解详细进展的人 | 各版本实验数据、诊断分析、待解决问题 |
| **REFACTOR_PROMPT.md** | 参与开发的人 / 大模型 | v5 重构的精确代码指引，含代码片段和验证清单 |
| **CHANGELOG.md** | 跟踪变更的人 | 各版本的具体变更记录 |

## License

MIT
