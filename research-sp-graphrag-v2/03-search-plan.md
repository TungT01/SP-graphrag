# 文献搜索策略 — SP-GraphRAG

## 搜索目标

基于问题分解树的 5 个子问题，制定系统化的文献搜索策略，覆盖以下研究领域：
1. GraphRAG 与知识图谱增强检索
2. 社区检测算法（特别是 Leiden/Louvain 的约束变体）
3. 结构熵与信息论在图分析中的应用
4. 文档物理结构在 NLP/IR 中的利用
5. 多粒度检索与溯源

---

## 搜索关键词组

### 组1: GraphRAG 核心文献
| # | 查询 | 目标 |
|---|------|------|
| 1.1 | GraphRAG knowledge graph retrieval augmented generation | 核心方法论 |
| 1.2 | GraphRAG community detection retrieval | 社区与检索关联 |
| 1.3 | Microsoft GraphRAG 2024 survey | 官方实现与综述 |
| 1.4 | graph-based RAG benchmark comparison | 基线对比 |
| 1.5 | RAG knowledge graph multi-hop reasoning | 多跳推理 |

### 组2: 约束社区检测
| # | 查询 | 目标 |
|---|------|------|
| 2.1 | constrained community detection Leiden algorithm | 约束 Leiden |
| 2.2 | Leiden algorithm modularity optimization variant | Leiden 变体 |
| 2.3 | semi-supervised community detection with constraints | 半监督约束 |
| 2.4 | community detection label propagation constraint | 标签约束 |
| 2.5 | hierarchical community detection with prior knowledge | 先验知识引导 |

### 组3: 结构熵与图信息论
| # | 查询 | 目标 |
|---|------|------|
| 3.1 | structural entropy graph community | 结构熵社区 |
| 3.2 | Shannon entropy community detection penalty | 信息熵惩罚 |
| 3.3 | information theoretic graph partitioning | 信息论图分割 |
| 3.4 | structural information graph encoding tree | Li Angsheng 结构信息理论 |
| 3.5 | entropy regularized graph clustering | 熵正则化聚类 |

### 组4: 文档结构与检索
| # | 查询 | 目标 |
|---|------|------|
| 4.1 | document structure aware retrieval NLP | 文档结构检索 |
| 4.2 | hierarchical document representation RAG | 层次文档表示 |
| 4.3 | passage retrieval physical layout | 段落检索布局 |
| 4.4 | multi-granularity text retrieval | 多粒度检索 |
| 4.5 | provenance tracking knowledge graph | 溯源追踪 |

### 组5: 相关基线与对比方法
| # | 查询 | 目标 |
|---|------|------|
| 5.1 | MultiHop-RAG dataset benchmark evaluation | 数据集评估 |
| 5.2 | naive RAG vs graph RAG comparison | 基线对比 |
| 5.3 | LightRAG 2024 graph retrieval | 最新方法 |
| 5.4 | RAPTOR recursive tree retrieval | 树结构检索 |
| 5.5 | HippoRAG knowledge graph retrieval 2024 | 类似方法 |

---

## 搜索策略

### 搜索源优先级
1. **arXiv**（预印本，最新研究）— 限定 site:arxiv.org
2. **Semantic Scholar / Google Scholar**（引用追踪）
3. **ACL Anthology**（NLP/IR 顶会论文）
4. **Web 通用搜索**（技术博客、GitHub README 等补充资料）

### 执行策略
- 第一轮：每组取前 3 个关键词执行搜索，快速覆盖核心文献
- 第二轮：根据第一轮结果调整关键词，补充漏网文献
- 第三轮：追踪高引文献的引用链（搜索标题 + "cited by"）

### 筛选标准
- relevance_score ≥ 0.6（与本研究直接相关）
- quality_score ≥ 0.5（方法论清晰、有实验验证）
- 优先保留 2022-2025 年的最新工作
- 经典方法（如 Leiden 原始论文）不限年份
