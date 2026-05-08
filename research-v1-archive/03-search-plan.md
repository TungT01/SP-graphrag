# 文献搜索计划

## 搜索策略总览

基于问题分解树的 5 个子问题，设计以下搜索关键词组。每个子问题对应 3-5 组搜索查询，包含原始查询及其 survey/benchmark/comparison 变体。

---

## 关键词组 1：GraphRAG 基础与演进（对应 SP1-SP5 全局背景）

| 编号 | 查询 | 目标 |
|------|------|------|
| Q1.1 | GraphRAG knowledge graph retrieval augmented generation | 核心概念 |
| Q1.2 | GraphRAG community detection Leiden algorithm | 聚类机制 |
| Q1.3 | GraphRAG survey benchmark comparison | 综述对比 |
| Q1.4 | Microsoft GraphRAG global local search | 微软原始论文 |
| Q1.5 | graph-based RAG limitations hallucination | 已知局限 |

## 关键词组 2：物理文档结构与知识图谱（对应 SP1）

| 编号 | 查询 | 目标 |
|------|------|------|
| Q2.1 | document structure preservation knowledge graph | 文档结构保持 |
| Q2.2 | proposition extraction knowledge graph triple | 命题转换 |
| Q2.3 | provenance tracking knowledge graph RAG | 出处追踪 |
| Q2.4 | chunk-level attribution retrieval augmented | 块级归因 |
| Q2.5 | document layout structure information extraction | 文档布局感知 |

## 关键词组 3：社区检测与结构约束聚类（对应 SP2）

| 编号 | 查询 | 目标 |
|------|------|------|
| Q3.1 | Leiden algorithm community detection modularity | Leiden 算法 |
| Q3.2 | structure entropy graph clustering constraint | 结构熵聚类 |
| Q3.3 | hierarchical community detection knowledge graph | 层级社区检测 |
| Q3.4 | constrained graph clustering document structure | 约束聚类 |
| Q3.5 | annealing optimization graph partitioning | 退火优化 |

## 关键词组 4：专业领域 RAG 与医疗/科研场景（对应 SP4, SP5）

| 编号 | 查询 | 目标 |
|------|------|------|
| Q4.1 | MedGraphRAG medical knowledge graph retrieval | 医疗 GraphRAG |
| Q4.2 | scientific literature RAG knowledge extraction | 科研文献 RAG |
| Q4.3 | domain-specific RAG biomedical legal | 领域特定 RAG |
| Q4.4 | RAG faithfulness attribution source tracking | RAG 忠实性 |
| Q4.5 | retrieval augmented generation hallucination survey | RAG 幻觉综述 |

## 关键词组 5：检索策略与多粒度检索（对应 SP3）

| 编号 | 查询 | 目标 |
|------|------|------|
| Q5.1 | hierarchical retrieval multi-granularity RAG | 多粒度检索 |
| Q5.2 | top-down bottom-up retrieval knowledge graph | 双向检索 |
| Q5.3 | hybrid retrieval dense sparse knowledge graph | 混合检索 |
| Q5.4 | graph traversal retrieval augmented generation | 图遍历检索 |
| Q5.5 | multi-hop reasoning knowledge graph QA | 多跳推理 |

---

## 搜索源优先级

1. **arXiv**（最高优先级）：最新预印本，覆盖 GraphRAG、RAG、KG 领域
2. **Google Scholar**：索引广泛，含已发表会议/期刊论文
3. **Semantic Scholar**：提供引用关系和影响力指标
4. **ACL Anthology**：NLP 领域顶会论文
5. **一般 Web 搜索**：补充技术博客和项目文档（如微软 GraphRAG 官方文档）

## 搜索执行计划

- 第一轮：核心关键词搜索（Q1.1-Q1.4, Q4.1, Q4.2），建立文献基础
- 第二轮：扩展搜索（Q2.x, Q3.x, Q5.x），补充技术细节
- 第三轮：引文追踪——对第一轮找到的核心论文，搜索其引用文献和被引文献
- 去重与筛选：DOI/标题去重，relevance ≥ 0.6 保留
