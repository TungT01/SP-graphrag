# 文献采集结果 — SP-GraphRAG

## 文献列表

---

### [1] From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **作者**: Darren Edge, Ha Trinh, Newman Cheng, et al. (Microsoft Research)
- **年份**: 2024
- **来源**: arXiv:2404.16130 (后发表于 NeurIPS 2024 Workshop)
- **摘要**: 提出 GraphRAG 方法，通过从文本中抽取知识图谱、使用 Leiden 社区检测构建层次化社区、为每个社区生成 LLM 摘要报告，实现全局性问答。在 query-focused summarization 任务上显著优于 naive RAG。
- **相关性**: 1.0 — 本研究的直接基线和改进目标
- **质量**: 0.9 — 完整的系统设计和评估

---

### [2] From Louvain to Leiden: Guaranteeing Well-Connected Communities
- **作者**: V.A. Traag, L. Waltman, N.J. van Eck
- **年份**: 2019
- **来源**: Scientific Reports, 9:5233 (Nature)
- **DOI**: 10.1038/s41598-019-41695-z
- **摘要**: 提出 Leiden 算法，改进 Louvain 算法的社区检测质量和速度。证明 Leiden 算法保证社区的连通性，并在 local moving + refinement + aggregation 三阶段迭代中收敛。
- **相关性**: 1.0 — 本研究修改的核心算法
- **质量**: 1.0 — 高引经典论文

---

### [3] Structural Information and Dynamical Complexity of Networks
- **作者**: Angsheng Li, Yicheng Pan
- **年份**: 2016
- **来源**: IEEE Transactions on Information Theory, 62(6):3290-3339
- **DOI**: 10.1109/TIT.2016.2555904
- **摘要**: 提出图的结构信息（structural information）理论，定义了基于编码树（encoding tree）的 K 维结构熵，将香农信息论从随机变量扩展到图结构。结构熵最小化原则可用于最优社区分割。
- **相关性**: 0.9 — 本研究的理论基础（结构熵概念）
- **质量**: 1.0 — 该领域奠基性论文

---

### [4] A Survey of Structural Entropy: Theory, Methods, and Applications
- **作者**: Hao Peng et al.
- **年份**: 2025
- **来源**: IJCAI 2025 / arXiv
- **摘要**: 全面综述结构熵理论，涵盖计算方法、学习范式和跨领域应用（从生物信息学到模式识别）。强调结构熵在推进图分析和理解方面的潜力。
- **相关性**: 0.85 — 提供结构熵的最新研究全景
- **质量**: 0.9 — 顶会综述论文

---

### [5] LightRAG: Simple and Fast Retrieval-Augmented Generation
- **作者**: Zirui Guo, Lianghao Xia, et al. (HKUDS)
- **年份**: 2024
- **来源**: arXiv:2410.05779 (EMNLP 2025 Findings)
- **摘要**: 提出 LightRAG，将图结构整合到文本索引和检索中，采用双层检索系统（low-level 实体级 + high-level 主题级）。相比 GraphRAG 更快速、成本更低。
- **相关性**: 0.8 — 重要对比方法，代表 GraphRAG 的轻量化趋势
- **质量**: 0.85 — EMNLP Findings

---

### [6] HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
- **作者**: Bernal Jiménez Gutiérrez, Yiheng Shu, et al. (OSU NLP Group)
- **年份**: 2024
- **来源**: NeurIPS 2024 (arXiv:2405.14831)
- **摘要**: 受海马体索引理论启发，使用知识图谱 + Personalized PageRank 实现多跳信息检索。在多跳 QA 任务上显著优于标准 RAG。
- **相关性**: 0.8 — 同为知识图谱增强检索，但使用 PPR 而非社区检测
- **质量**: 0.95 — NeurIPS 主会

---

### [7] RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- **作者**: Parth Sarthi, Salman Abdullah, et al.
- **年份**: 2024
- **来源**: ICLR 2024 (arXiv:2401.18059)
- **摘要**: 通过递归嵌入、聚类和摘要文本块，从底向上构建层次树结构。检索时在不同抽象层级整合信息。
- **相关性**: 0.75 — 多粒度层次化检索的对比方法
- **质量**: 0.95 — ICLR 2024 主会

---

### [8] MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries
- **作者**: Yixuan Tang, Yi Yang
- **年份**: 2024
- **来源**: arXiv:2401.15391
- **摘要**: 构建了首个专注于多跳查询的 RAG benchmark 数据集，包含 609 篇新闻文章和 2556 个多跳 QA 对。每个问题需要跨 2-4 篇文档推理。
- **相关性**: 1.0 — 本研究使用的评估数据集
- **质量**: 0.85 — 广泛使用的 benchmark

---

### [9] Community Detection in Large-Scale Complex Networks via Structural Entropy (CoDeSEG)
- **作者**: (arXiv:2501.15130)
- **年份**: 2025
- **来源**: arXiv:2501.15130
- **摘要**: 提出 CoDeSEG 算法，通过最小化网络的二维结构熵来识别社区，采用势博弈框架。解决了传统方法在大规模网络上的可扩展性问题。
- **相关性**: 0.85 — 结构熵+社区检测的最新工作，方向最接近本研究
- **质量**: 0.8 — 预印本

---

### [10] Retrieval-Augmented Generation with Graphs (GraphRAG Survey)
- **作者**: (蚂蚁集团 & 多所高校联合)
- **年份**: 2025
- **来源**: arXiv:2501.00309
- **摘要**: 提出了一个全面的 GraphRAG 框架定义，将其分解为 query processor、retriever、organizer、generator、data source 五个组件。综述了近 100 篇 GraphRAG 相关论文。
- **相关性**: 0.85 — 定位本研究在 GraphRAG 全景中的位置
- **质量**: 0.9 — 综合性综述

---

### [11] Entropy-Guided Graph Clustering via Rényi Optimization
- **作者**: (Springer 2025)
- **年份**: 2025
- **来源**: Springer LNCS
- **摘要**: 提出基于 Rényi 熵的可微分图聚类框架，引入 masked entropy loss 鼓励信息性节点表示并尊重图拓扑。
- **相关性**: 0.7 — 信息论+图聚类的最新对比方法
- **质量**: 0.75

---

### [12] Document Segmentation Matters for Retrieval-Augmented Generation
- **作者**: (ACL 2025 Findings)
- **年份**: 2025
- **来源**: ACL 2025 Findings
- **摘要**: 系统研究了文档分割（chunking）对 RAG 性能的影响。提出利用文档摘要作为伪指令指导分块，通过计算句子与摘要的语义相似度优化分割。
- **相关性**: 0.7 — 文档结构对检索的影响
- **质量**: 0.8

---

### [13] A Comprehensive Review of Community Detection in Graphs
- **作者**: (arXiv:2309.11798)
- **年份**: 2023
- **来源**: arXiv:2309.11798
- **摘要**: 全面综述图社区检测方法，涵盖模块度方法、谱聚类、概率模型和深度学习。详细对比各类方法的优缺点。
- **相关性**: 0.7 — 提供社区检测方法的全景背景
- **质量**: 0.85

---

### [14] Network Optimization Approach to Delineating Health Care Service Areas
- **作者**: (PMC, 2021)
- **年份**: 2021
- **来源**: PMC
- **摘要**: 实现了带约束的社区检测（空间连通性约束 + 最小区域尺寸阈值），将额外约束引入 Leiden/Louvain 算法中。
- **相关性**: 0.75 — 约束社区检测的实际应用案例
- **质量**: 0.7

---

### [15] StructuGraphRAG: Structured Document-Informed Knowledge Graphs for RAG
- **作者**: (AAAI 2025 Spring Symposium)
- **年份**: 2025
- **来源**: AAAI-SS 2025
- **摘要**: 利用文档结构信息来指导知识图谱抽取过程，构建结构感知的知识图谱用于 RAG。
- **相关性**: 0.75 — 文档结构+知识图谱+RAG 的交叉工作
- **质量**: 0.7

---

## 文献覆盖度评估

| 子问题 | 核心文献数 | 覆盖状况 |
|--------|-----------|----------|
| SP1 物理锚定 | 3 篇 ([1], [12], [15]) | 充足 |
| SP2 结构熵目标函数 | 4 篇 ([3], [4], [9], [11]) | 充足 |
| SP3 退火策略 | 2 篇 ([2], [14]) | 该方向文献有限，退火在社区检测中的应用较少见 |
| SP4 EdgeSchedule | 2 篇 ([2], [13]) | 需从 Leiden 原始机制出发论述 |
| SP5 检索保持性 | 5 篇 ([1], [5], [6], [7], [8]) | 充足 |

**总计**：15 篇高相关性文献，覆盖了研究的所有核心方向。文献数量充足，可进入下一阶段。
