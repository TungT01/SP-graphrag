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
├── graphrag_improved/              # 核心源码（v8 当前版本）
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

## 实验结果（v8, MultiHop-RAG, n=500, 429 有效 QA）

六组消融实验，含 Bootstrap CI（1000 resamples, 95%）和 Para-Hit 指标：

| 配置 | λ_0 | avg_H | MRR | MRR 95% CI | Para-MRR | R@5 | NDCG@10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline（无 ES, λ=0） | 0 | 0.000 | 0.4478 | [0.409, 0.484] | 0.4275 | 0.399 | 0.3624 |
| EdgeSchedule only | 0 | 0.137 | 0.4469 | [0.407, 0.485] | 0.4275 | 0.401 | 0.3607 |
| Weak (λ=0.001) | 0.001 | 0.115 | 0.4478 | [0.408, 0.485] | 0.4275 | 0.402 | 0.3612 |
| **Med (λ=0.003, 推荐)** | **0.003** | **0.089** | 0.4463 | [0.407, 0.485] | 0.4275 | 0.398 | 0.3605 |
| +PathA | 0.001 | 0.120 | 0.4464 | [0.407, 0.485] | 0.4275 | 0.399 | 0.3613 |
| +CrossDoc | 0.001 | 0.149 | 0.4428 | [0.403, 0.481] | 0.4275 | 0.397 | 0.3596 |

**核心结论**：

- 结构熵随 λ 增大单调递减（0.137→0.089），验证惩罚项有效控制社区物理纯净度
- 所有 6 组的 MRR 95% CI 完全重叠 → 检索质量无统计显著差异，结构熵惩罚「无损」
- Para-MRR 全组恒定（0.4275）→ 段落级检索表现不受社区检测参数影响（BottomUp bug 已修复但段落匹配路径一致）
- 跨文档边引入噪声（avg_H 反升至 0.149，MRR 微降），不推荐

## 版本演进

| 版本 | 关键特性 | 状态 |
|------|---------|------|
| v3 | 物理优先架构：instance-level 节点，ID={sent_id}-{entity} | 已完成 |
| v4 | 四组消融实验：Path A + Path B 对照 | 已完成 |
| v5 | anchor_granularity="para" + EdgeSchedule + 移除提前终止 | 已完成 |
| v6 | EdgeSchedule 权重校准（2.0/1.5/1.0）+ 噪声过滤 | 已完成 |
| v7 | 六组消融实验完整验证（n=200）：熵可控 + 检索无损 | 已完成 |
| v8 | 修复 BottomUp 锚点 bug + Bootstrap CI + Para-Hit 指标 + 大样本验证（n=500, 429 有效） | **当前版本** |

## 已知问题

1. **社区摘要缺失**：当前管线不生成 community_summary（需 LLM），TopDown 检索仅用实体列表
2. **跨文档边噪声**：同名异义实体被错误连接，需引入轻量实体消歧

## 技术文档

- [`TECHNICAL_SPEC.md`](./TECHNICAL_SPEC.md)：完整技术规格书，面向 LLM / AI Agent，包含所有算法细节、数据结构定义、代码映射
- [`research-sp-graphrag-v2/11-paper-final.md`](./research-sp-graphrag-v2/11-paper-final.md)：完整研究论文
- [`reports/`](./reports/)：v7 实验报告
- [`docs/`](./docs/)：项目计划、变更日志、文件清单等

## License

本项目仅供学术研究使用。
