# SP-GraphRAG: Structural-entropy Penalized GraphRAG

> 在 GraphRAG 的 Leiden 社区检测中引入结构熵惩罚项，通过物理锚点约束实现语义-结构对齐的层次化社区发现。

## 核心思想

标准 GraphRAG 使用 Leiden 算法发现文本知识图谱中的层次社区，但其目标函数仅优化模块度（Q），完全不感知节点的物理文档来源。SP-GraphRAG 在 Leiden 目标函数中加入结构熵惩罚项，使社区检测同时尊重语义关联和文档物理结构。

**目标函数**：

```
J = Q_leiden − λ · H_structure
```

- `Q_leiden`：标准模块度增益，衡量社区内部边密度超过随机期望
- `H_structure`：社区内节点物理来源分布的 Shannon 熵，`H = −Σ p_i log(p_i)`
- `λ`：退火系数，通过指数衰减 `λ_t = λ_0 · e^{-αt}` 控制约束强度——底层强约束保纯净，高层弱约束允许跨文档合并

## 项目结构

```
SP-GraphRAG/
├── graphrag_improved/              # 核心源码（v11b 当前版本）
│   ├── constrained_leiden/         # 核心算法模块
│   │   ├── leiden_constrained.py   # 结构熵约束 Leiden（CommunityEntropyState + O(1) 增量 ΔH）
│   │   ├── annealing.py            # λ 退火调度（指数/线性/余弦/阶梯 4 种曲线）
│   │   ├── edge_scheduler.py       # EdgeSchedule 三级同名实体边注入
│   │   ├── graphrag_workflow.py    # 主工作流编排 + 图构建 + 社区 DataFrame 生成
│   │   └── physical_anchor.py      # PhysicalNode 数据类 + 结构熵计算
│   ├── data/ingestion.py           # 文档 → 段落 → 句子三级物理切分
│   ├── extraction/extractor.py     # spaCy 依存句法三元组抽取（instance-level，无实体消解）
│   ├── retrieval/retriever.py      # U-Retrieval（TopDown 社区导航 + BottomUp TF-IDF 检索，v8 已修复锚点 bug）
│   ├── evaluation/evaluator.py     # P@K, R@K, MRR, NDCG@K, Bootstrap CI, Para-Hit 指标
│   ├── experiments/                # v6/v7/v8 消融实验脚本与结果
│   └── config.yaml                 # 项目配置
│
├── experiments/                    # 早期实验（v4/v5）
│   ├── scripts/                    # 实验与诊断脚本
│   └── results*/                   # 历史实验结果
│
├── baselines/                      # 基线对比
│   ├── naive_rag/                  # Naive RAG 基线
│   ├── graphrag_official/          # 微软官方 GraphRAG 基线
│   └── eval_results/               # 基线评估结果
│
├── data/multihop_rag/              # MultiHop-RAG 数据集（609 篇文章, 2556 QA）
├── research-sp-graphrag-v2/        # 研究论文（v2 最终版，含完整论文）
├── research-v1-archive/            # 研究论文（v1 初版草稿，已归档）
├── reports/                        # 实验报告
├── docs/                           # 项目文档
├── sample_data/                    # 冒烟测试样例数据
├── TECHNICAL_SPEC.md               # 完整技术规格书（面向 LLM）
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

## 快速开始

### 环境安装

```bash
cd SP-GraphRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 运行完整管线

```bash
cd graphrag_improved
python3 run.py --config config.yaml --data-dir ../sample_data
```

### 运行六组消融实验（v8）

```bash
cd graphrag_improved
python3 -m experiments.run_multihop_eval \
    --data-dir ../data/multihop_rag \
    --n-qa 500 \
    --output-dir experiments/results_v8
```

## 核心算法组件

**1. 三级物理ID体系**：`doc_id → para_id({doc_id}-p{NNN}) → sent_id({para_id}-s{NNN})`，节点 `node_id = {sent_id}-{entity_name}`。同名实体在不同句子中是不同节点（instance-level），不做实体消解。

**2. EdgeSchedule 三级边注入**：打破句内孤岛森林，在 Level 0 一次性注入段落内跨句子（w=2.0）、文档内跨段落（w=1.5）、跨文档（w=1.0）三类同名实体边。采用链式连接 O(n)，含停用词过滤。

**3. CommunityEntropyState**：O(1) 增量结构熵计算。维护每个社区的 `{chunk_id: weight}` 分布，节点移动时仅增量更新 ΔH，避免全量遍历。

**4. 指数退火**：`λ_t = λ_0 · exp(−0.5 · t)`。底层（Level 0）λ 最大，约束社区物理纯净；高层 λ→0，允许语义驱动的跨文档合并。

**5. U-Retrieval**：TopDown（从高层社区逐层导航到底层）+ BottomUp（TF-IDF 直接检索物理文本块），按 alpha 参数分配字符预算后融合。

## 实验结果（v11b，向量检索，n=1000，881 有效 QA）⭐ 最新

**核心对照**（唯一变量 = 社区检测算法，95% Bootstrap CI 完全不重叠）：

| 方法 | MRR | P@5 | R@5 | EM | MRR 95%CI |
|------|:---:|:---:|:---:|:--:|:---------:|
| A+VS 标准 Leiden（GraphRAG 复现） | 0.440 | 0.116 | 0.247 | 0.089 | [0.410, 0.471] |
| **B3+VS 约束 Leiden λ=0.003（本方法）** | **0.489** | **0.159** | **0.329** | **0.124** | **[0.458, 0.516]** |
| 提升 | **+10.9%** | **+37.4%** | **+33.6%** | **+39.3%** | **统计显著** |

**传导机制（E3b）**：约束 Leiden 从 LLM 摘要中获得的 MRR 收益（+10.9%）是标准 Leiden（+2.8%）的 3.9 倍，证明"物理纯净社区 → 更连贯摘要 → 向量检索提升"的因果链。

**消融结论（v7/v8）**：结构熵随 λ 单调递减，Level-0 社区物理纯净率 100%，EdgeSchedule 是约束生效的必要条件。

## 版本演进

| 版本 | 关键特性 | 状态 |
|------|---------|------|
| v5/v6 | anchor_granularity="para" + EdgeSchedule 权重校准 | 已完成 |
| v7/v8 | 消融验证（熵可控）+ Bootstrap CI + BottomUp bug 修复 | 已完成 |
| v9/v10 | LLM 摘要接入（TF-IDF 框架），发现稀释效应 | 已完成 |
| v11 | **向量检索框架，n=429，MRR +16.1%，CI 不重叠** | 已完成 |
| **v11b** | **向量检索大样本，n=881，MRR +10.9%，统计显著** | **当前版本** |

## 已知局限

1. **实体抽取**：使用 spaCy（非 LLM），为控制变量的合理选择；LLM 抽取预期效果更显著
2. **单一数据集**：MultiHop-RAG（新闻领域），其他领域泛化性待验证
3. **向量检索依赖**：TF-IDF 框架下约束效果被稀释效应抵消，需配合向量检索才能体现价值

## 技术文档

- [`TECHNICAL_SPEC.md`](./TECHNICAL_SPEC.md)：完整技术规格书（算法、数据结构、代码映射）
- [`docs/EXPERIMENT_RESULTS.md`](./docs/EXPERIMENT_RESULTS.md)：所有实验结果权威记录（v7-v11b）
- [`reports/thesis_structure.md`](./reports/thesis_structure.md)：硕士论文结构规划
- [`reports/SP-GraphRAG_Academic_Report_v11.md`](./reports/SP-GraphRAG_Academic_Report_v11.md)：v11 实验完整报告

## License

本项目仅供学术研究使用。
