# 知识卡片 — SP-GraphRAG

---

### From Local to Global: A Graph RAG Approach (Edge et al., 2024)
- **cite_key**: edge2024graphrag
- **核心问题**: 传统 RAG 无法回答全局性问题（如"数据集的主要主题是什么？"）
- **方法**: 从文本抽取知识图谱 → Leiden 社区检测构建层次化社区 → LLM 为每个社区生成摘要报告 → 基于社区摘要进行 map-reduce 全局问答
- **数据**: 私有文本语料库（新闻、播客转录等）
- **关键指标**: 在 comprehensiveness 和 diversity 指标上显著优于 naive RAG（约 70-80% 的 win rate）
- **发现**: 社区层次结构提供了不同粒度的语料库理解；中间层级（C1-C2）通常效果最好
- **局限**: Leiden 社区检测仅优化拓扑模块度，不考虑节点的物理来源；社区摘要依赖 LLM 质量；索引成本高
- **与本研究的关联**: 直接基线。本研究在 Leiden 步骤中引入结构熵惩罚，保留其层次化社区框架但增加物理结构感知性

---

### From Louvain to Leiden (Traag et al., 2019)
- **cite_key**: traag2019leiden
- **核心问题**: Louvain 算法可能产生断开的（disconnected）社区
- **方法**: 三阶段迭代——local moving（节点移动优化模块度）→ refinement（细化社区内部）→ aggregation（聚合为新图进入下一层）。使用快速本地移动和随机邻居移动加速
- **数据**: 多种 benchmark 网络和真实网络
- **关键指标**: 模块度持平或优于 Louvain，同时保证社区连通性，速度更快
- **发现**: 连通性保证是 Leiden 对 Louvain 最重要的改进
- **局限**: 仍然仅优化模块度 Q，不考虑任何外部约束或领域先验
- **与本研究的关联**: 本研究在 local moving 阶段修改节点移动决策为 ΔJ = ΔQ - λ·ΔH，保留 Leiden 的 refinement 和 aggregation 不变

---

### Structural Information and Dynamical Complexity of Networks (Li & Pan, 2016)
- **cite_key**: li2016structural
- **核心问题**: 如何量化图的结构复杂性（超越传统的随机变量信息论）
- **方法**: 定义 K 维结构信息：给定图 G 和高度为 K 的编码树 T，结构熵 H^K(G) = 最小化 T 下的编码长度。1 维 = 无社区的全局熵，2 维 = 最优二层社区分割下的熵
- **数据**: 理论证明 + 算法实验
- **关键指标**: 提出结构熵最小化原则作为最优社区分割的理论依据
- **发现**: 结构熵可以量化"真正的"社区结构信息量，区别于模块度等启发式指标
- **局限**: 精确计算 K 维结构熵为 NP-hard，需要近似算法
- **与本研究的关联**: 为 SP-GraphRAG 的 H_structure 惩罚项提供理论动机——本研究使用简化版的 Shannon 熵作为物理来源混杂度的度量

---

### A Survey of Structural Entropy (Peng et al., 2025)
- **cite_key**: peng2025survey_se
- **核心问题**: 全面综述结构熵的理论、方法和应用
- **方法**: 涵盖精确算法、近似算法、基于 GNN 的学习方法
- **关键指标**: 在社区检测、图分类、异常检测等多个任务上均有应用
- **发现**: 结构熵作为全局信息度量，相比模块度能更好地捕获层次化结构
- **局限**: 大规模图上的计算效率仍是挑战
- **与本研究的关联**: 确认了结构熵在图分析中的理论价值和实际应用潜力

---

### LightRAG (Guo et al., 2024)
- **cite_key**: guo2024lightrag
- **核心问题**: GraphRAG 索引成本高、查询慢（需遍历数百个社区）
- **方法**: 双层检索系统——low-level（实体和关系的 KV 存储）+ high-level（主题级聚合）；增量更新算法
- **数据**: 多个领域文本数据集
- **关键指标**: 检索准确率和效率均优于 GraphRAG，显著降低 API 调用次数
- **发现**: 图结构 + 向量表示的结合比纯社区摘要更高效
- **局限**: 不保留文档的物理层次结构；实体消解依赖 LLM
- **与本研究的关联**: 代表 GraphRAG 轻量化趋势的对比方法；其双层检索与 SP-GraphRAG 的 U-Retrieval 有概念相似性

---

### HippoRAG (Gutiérrez et al., 2024)
- **cite_key**: gutierrez2024hipporag
- **核心问题**: RAG 系统缺乏类似人类长期记忆的知识整合能力
- **方法**: 模拟海马体索引理论——使用 KG 作为索引结构，Personalized PageRank 从查询种子在图上传播以整合多跳信息
- **数据**: MultiHop-RAG 等多跳 QA 数据集
- **关键指标**: 在多跳检索上显著优于标准 RAG（+20% recall）
- **发现**: 图结构的传播检索比单次向量匹配更适合多跳推理
- **局限**: PPR 传播可能引入噪声；对 KG 质量敏感
- **与本研究的关联**: 同为 KG 增强检索，但使用 PPR 而非社区检测；可作为未来替代检索策略的参考

---

### RAPTOR (Sarthi et al., 2024)
- **cite_key**: sarthi2024raptor
- **核心问题**: 传统 RAG 只能检索短的连续文本片段，无法理解完整文档
- **方法**: 递归嵌入 + 聚类 + 摘要构建层次树；检索时在不同抽象层级访问节点
- **数据**: NarrativeQA, QASPER, QuALITY 等长文档 QA
- **关键指标**: 在多种长文档 QA 任务上优于 flat retrieval（+5-15%）
- **发现**: 不同查询类型受益于不同层级的摘要节点
- **局限**: 聚类质量依赖嵌入模型；摘要可能丢失细节
- **与本研究的关联**: 层次化组织检索单元的思路与 SP-GraphRAG 的社区层次有概念相似性；RAPTOR 的聚类不保留物理结构

---

### MultiHop-RAG (Tang & Yang, 2024)
- **cite_key**: tang2024multihoprag
- **核心问题**: 缺少专注多跳查询的 RAG benchmark
- **方法**: 基于 609 篇新闻文章构建知识库，生成 2556 个多跳 QA 对（跨 2-4 篇文档推理）
- **数据**: 4 种查询类型：推理、比较、时序、无效查询
- **关键指标**: 现有 RAG 系统在多跳场景下性能大幅下降
- **发现**: 88% 的问题需要跨文档证据整合；检索和生成阶段均面临挑战
- **局限**: 仅基于新闻文章，领域单一
- **与本研究的关联**: 本研究的评估数据集，所有实验在此数据集上进行

---

### CoDeSEG (arXiv:2501.15130, 2025)
- **cite_key**: codeseg2025
- **核心问题**: 传统社区检测在大规模网络上的可扩展性和质量
- **方法**: 通过最小化网络的二维结构熵，在势博弈框架下识别社区
- **数据**: 大规模合成和真实网络
- **关键指标**: 在 LFR benchmark 上社区检测质量优于 Louvain/Leiden
- **发现**: 结构熵最小化可以直接发现社区，无需预设社区数
- **局限**: 未与领域先验（如物理结构）结合
- **与本研究的关联**: 最接近本研究的方法——将结构熵用于社区检测优化。区别在于 CoDeSEG 最小化全局结构熵以发现社区，而 SP-GraphRAG 将物理来源的局部熵作为惩罚项引入 Leiden

---

### GraphRAG Survey (2025)
- **cite_key**: graphrag_survey2025
- **核心问题**: 对 GraphRAG 领域进行系统化分类和综述
- **方法**: 定义 GraphRAG 框架的五大组件，综述近 100 篇相关论文
- **关键指标**: N/A（综述论文）
- **发现**: GraphRAG 领域快速发展，但社区检测步骤的优化是被忽视的研究方向
- **与本研究的关联**: 定位本研究在 GraphRAG 全景中的位置

---

### Entropy-Guided Graph Clustering via Rényi Optimization (2025)
- **cite_key**: renyi2025graph
- **核心问题**: 图聚类中如何利用信息论引导学习
- **方法**: 基于可微分 Rényi 熵的 masked entropy loss
- **发现**: 熵正则化可以鼓励更均匀、更具信息量的节点表示
- **与本研究的关联**: 信息论+图聚类的方法论对比

---

### Document Segmentation Matters for RAG (ACL 2025)
- **cite_key**: docseg2025
- **核心问题**: 文档分割策略对 RAG 性能的影响
- **方法**: 利用文档摘要作为伪指令指导语义分块
- **发现**: 语义感知分块比固定长度分块在 RAG 上表现更好
- **与本研究的关联**: 强调了文档物理结构对检索质量的重要性

---

### Constrained Community Detection for Health Care Service Areas (2021)
- **cite_key**: constrained_cd2021
- **核心问题**: 医疗服务区域划分需要满足空间连通性和最小区域尺寸约束
- **方法**: 在 Leiden/Louvain 中加入空间约束
- **发现**: 约束社区检测可以产出更有实际意义的划分结果
- **与本研究的关联**: 约束社区检测的先例——证明在 Leiden 中加入外部约束是可行的

---

### StructuGraphRAG (AAAI-SS 2025)
- **cite_key**: structugraphrag2025
- **核心问题**: 如何利用文档结构信息改进知识图谱抽取和 RAG
- **方法**: 利用文档层次结构指导实体和关系抽取
- **发现**: 结构感知的 KG 构建比 flat 抽取质量更高
- **与本研究的关联**: 文档结构 + KG + RAG 的交叉工作，但关注抽取阶段而非社区检测阶段

---

### A Comprehensive Review of Community Detection in Graphs (2023)
- **cite_key**: cd_review2023
- **核心问题**: 图社区检测方法全景综述
- **方法**: 涵盖模块度方法、谱聚类、概率模型和深度学习
- **发现**: 模块度最大化仍是最广泛使用的方法；深度学习方法在有标注数据时更优
- **与本研究的关联**: 提供社区检测的研究全景背景
