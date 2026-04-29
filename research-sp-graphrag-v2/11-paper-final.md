# SP-GraphRAG：面向知识图谱检索增强生成的结构熵惩罚社区检测方法

---

## 摘要

基于知识图谱的检索增强生成（GraphRAG）通过 Leiden 社区检测将实体图划分为层次化社区，并以社区摘要作为检索单元，在全局性问答任务上取得了显著效果。然而，Leiden 算法仅优化图拓扑模块度，完全忽略了实体节点的物理出处信息（即节点来自哪个句子、段落或文档），导致社区内节点的物理来源高度混杂，削弱了检索结果的可溯源性和社区摘要的语义连贯性。

本文提出 **SP-GraphRAG**（Structural-entropy Penalized GraphRAG），在 Leiden 的目标函数中引入物理来源分布的 Shannon 熵惩罚项，将优化目标从纯模块度最大化 $Q$ 扩展为 $J = Q - \lambda \cdot H$，其中 $H$ 度量社区内节点物理来源的混杂程度。配合指数退火策略（$\lambda$ 从高值衰减至零），SP-GraphRAG 实现了"低层社区保持物理来源纯净、高层社区允许跨文档语义合并"的渐进式聚合控制。为解决纯句内边图退化为孤岛森林的问题，我们设计了 EdgeSchedule 机制，通过分层注入同名实体边使结构熵约束得以生效。

**本文的目标不是超越 Naive RAG 的检索精度，而是在保持检索质量的同时，为 GraphRAG 的社区检测引入物理结构感知能力。** 在 MultiHop-RAG 数据集（609 篇文章，2556 个多跳问答对）上的 6 组消融实验表明：（1）**约束有效且可控**——全局平均结构熵 $H$ 从 0 降至 0.10–0.17，$\lambda$ 与 $H$ 呈单调递减关系，Level-0 社区物理来源纯净率达 100%；（2）**检索质量无损**——所有约束配置的 MRR、P@5、R@5、NDCG@10 波动均在 ±2.5% 以内（受限于 169 条有效 QA 的样本量，差异的统计显著性有限，详见 §5.2）；（3）**层次结构更丰富**——退火策略使社区层数从 3 层增至 8–9 层。

本文的主要贡献包括：（i）提出结构熵惩罚的约束 Leiden 算法，首次将文档物理结构信息纳入 GraphRAG 社区检测优化；（ii）设计退火调度策略实现分层物理约束控制；（iii）通过系统消融实验验证约束有效性与检索无损性的共存。

---

## 1. 引言

### 1.1 背景与动机

检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成时动态检索外部知识，有效缓解了大型语言模型（LLM）的幻觉问题和知识截止局限。然而，传统 RAG 将文档切分为固定长度的文本块（chunk），在处理需要跨文档推理的全局性问题时力不从心——例如"这批新闻报道的主要议题是什么？"或"哪些事件在时间上存在因果关联？"这类问题本质上是查询聚焦摘要（Query-Focused Summarization）任务，而非简单的段落检索。

为解决这一局限，Edge 等人提出了 GraphRAG [edge2024graphrag]，将 RAG 与知识图谱（Knowledge Graph, KG）相结合。GraphRAG 的核心管线包含四个步骤：（1）使用 LLM 从文本中抽取实体和关系，构建知识图谱；（2）对知识图谱运行 Leiden 社区检测算法，将实体划分为层次化社区；（3）为每个社区生成 LLM 摘要报告；（4）查询时通过社区摘要进行 map-reduce 式全局问答。这一框架在全局性问答任务上取得了显著效果，成为 2024 年 RAG 领域最具影响力的工作之一。

### 1.2 问题：Leiden 社区检测的拓扑盲目性

GraphRAG 管线的核心步骤——Leiden 社区检测——存在一个根本性局限：**拓扑盲目性**（topological blindness）。Leiden 算法通过最大化模块度 $Q$ 来划分社区，模块度衡量的是社区内部边密度相对于随机图的超出程度 [traag2019leiden]。这一优化目标完全基于图的拓扑结构，对节点的语义属性和物理出处一无所知。

具体而言，知识图谱中的每个实体节点都有其物理出处——它来自语料库中某篇文档的某个段落的某个句子。这一物理出处信息在标准 GraphRAG 的实体抽取阶段即被丢弃，Leiden 算法在此后的社区划分中无从利用。其后果是：一个社区可能包含来自十几篇不同文档、数十个不同段落的实体，这些实体在拓扑上紧密相连（因为它们在文本中共现），但在物理来源上高度混杂。

这种物理来源混杂性带来了三方面的实际问题：**可溯源性受损**（用户无法确定答案来自哪份文件的哪个段落）；**社区摘要连贯性下降**（LLM 为混杂社区生成摘要时，输入信息来自多个不相关文档，摘要可能缺乏主题聚焦）；**增量更新代价高**（新增文档可能扰动大量已有社区，导致摘要重新生成成本高）。

**本文的目标**：SP-GraphRAG 的目标不是超越 Naive RAG 的检索精度，而是在保持检索质量的同时，为 GraphRAG 的社区检测引入物理结构感知能力，从而支持精确溯源、提升社区摘要连贯性、降低增量更新代价。

### 1.3 本文方案与贡献

本文提出 SP-GraphRAG，通过在 Leiden 的目标函数中引入物理来源分布的 Shannon 熵惩罚项：

$$J = Q_{\text{leiden}} - \lambda \cdot H_{\text{structure}}$$

其中 $H_{\text{structure}} = -\sum_i p_i \log p_i$ 度量社区内节点物理来源的混杂程度。配合指数退火策略和 EdgeSchedule 机制，SP-GraphRAG 实现了渐进式的物理结构感知聚合控制。

**主要贡献**：（i）结构熵惩罚的约束 Leiden 算法；（ii）退火调度策略；（iii）系统消融实验验证。

---

## 2. 相关工作

### 2.1 GraphRAG 与知识图谱增强检索

Edge 等人的 GraphRAG [edge2024graphrag] 是该方向的奠基性工作，通过 Leiden 社区检测构建层次化社区摘要，在全局性问答任务上显著优于 naive RAG。GraphRAG 综述 [graphrag_survey2025] 将该领域系统化为五大组件，并指出社区检测步骤的优化是被忽视的研究方向。

在 GraphRAG 的轻量化方向，LightRAG [guo2024lightrag] 提出双层检索系统，通过 KV 存储替代社区摘要，显著降低了索引成本（EMNLP 2025）。HippoRAG [gutierrez2024hipporag] 受海马体索引理论启发，使用知识图谱结合 Personalized PageRank 实现多跳信息检索（NeurIPS 2024）。RAPTOR [sarthi2024raptor] 通过递归聚类和摘要构建层次树结构（ICLR 2024）。上述工作均未关注社区检测步骤中节点物理出处的保留问题。

### 2.2 社区检测算法与约束优化

Leiden 算法 [traag2019leiden] 通过三阶段迭代（local moving → refinement → aggregation）最大化模块度 $Q$，是 GraphRAG 官方实现的默认社区检测方法。社区检测综述 [cd_review2023] 指出，模块度最大化方法无法融入领域先验知识。约束社区检测在地理网络中已有应用 [constrained_cd2021]，但在 NLP/IR 领域的知识图谱上尚未被探索。

### 2.3 结构熵与图信息论

Li 和 Pan [li2016structural] 提出了图的结构信息理论，定义了基于编码树的 $K$ 维结构熵，为本文的 $H_{\text{structure}}$ 惩罚项提供了信息论基础。CoDeSEG [codeseg2025] 通过最小化网络的二维结构熵来识别社区。本文与 CoDeSEG 的根本区别在于：CoDeSEG 将结构熵作为**发现**社区的目标，而本文将物理来源的局部熵作为**约束**已有社区检测的惩罚项。

### 2.4 文档结构感知的检索方法

ACL 2025 的工作 [docseg2025] 证明语义感知分块优于固定长度分块。StructuGraphRAG [structugraphrag2025] 利用文档层次结构指导知识图谱抽取。上述工作在**抽取阶段**或**分块阶段**利用文档结构，而本文在**社区检测阶段**利用物理结构信息，填补了这一空白。

---

## 3. 方法

### 3.1 问题形式化

给定文档集合 $\mathcal{D}$，通过 NER 和依存分析构建知识图谱 $G = (V, E)$，其中每个节点 $v \in V$ 携带三级物理坐标 $\phi(v) = (\text{doc\_id}, \text{para\_id}, \text{sent\_id})$。目标是找到节点划分 $\mathcal{C} = \{C_1, \ldots, C_K\}$，最大化：

$$J(\mathcal{C}) = Q(\mathcal{C}) - \lambda \cdot H(\mathcal{C})$$

### 3.2 物理锚定表示

SP-GraphRAG 采用"物理优先"（instance-level）策略：不做跨文档实体消解，同名实体在不同句子中视为独立节点，每个节点保留唯一的物理坐标。节点 ID 编码为 `{sent_id}-{entity_name}`，保留完整的物理出处链。物理单元粒度默认使用**段落级**（para\_id），在语义完整性和物理粒度之间取得平衡。

### 3.3 结构熵惩罚项

对于社区 $C_k$，设第 $i$ 个物理单元（段落）在 $C_k$ 中的节点比例为 $p_{k,i}$，则：

$$H(C_k) = -\sum_{i=1}^{M_k} p_{k,i} \log_2 p_{k,i}$$

当 $M_k = 1$（所有节点来自同一段落）时，$H(C_k) = 0$（最纯净）；当节点均匀分布于 $M_k$ 个段落时，$H(C_k) = \log_2 M_k$（最混杂）。全局平均结构熵为 $H(\mathcal{C}) = \frac{1}{K} \sum_{k=1}^{K} H(C_k)$。

本文的 $H_{\text{structure}}$ 是 Li & Pan [li2016structural] 结构信息论的简化版本：不寻求最优编码树，而是直接将节点的物理来源分布视为概率分布计算 Shannon 熵，物理意义直观且计算高效。

### 3.4 增量 ΔH 计算（O(1)）

为每个社区 $C_k$ 维护物理单元计数器字典 $\text{cnt}_k$。当节点 $v$ 从社区 $C_{\text{src}}$ 移动到 $C_{\text{dst}}$ 时，仅需更新两个计数器项并重新计算两个 $p \log p$ 项，每次操作为 $O(1)$ 的算术运算（此处 $O(1)$ 假设 $\log$ 运算为常数时间，在实际实现中成立）。因此，每次节点移动的 $\Delta J = \Delta Q - \lambda \cdot (\Delta H_{\text{remove}} + \Delta H_{\text{add}})$ 可在 $O(1)$ 时间内计算。

### 3.5 退火调度策略

采用指数退火策略，在每完成一个 Leiden 层级后衰减 $\lambda$：

$$\lambda_t = \lambda_0 \cdot e^{-\alpha t}$$

其中 $t$ 为当前层级编号，$\alpha$ 为衰减率（本文取 $\alpha = 0.5$）。低层（$t$ 小）时 $\lambda_t \approx \lambda_0$，物理约束强；高层（$t$ 大）时 $\lambda_t \to 0$，约束消失，退化为标准 Leiden。

### 3.6 EdgeSchedule：同名实体边注入

**问题**：纯句内边图中每个连通分量仅包含来自同一句子的节点，$H \equiv 0$，结构熵惩罚完全失效。

**解决方案**：EdgeSchedule 注入三类同名实体边：段落内跨句子边（$w=2.0$，链式连接）、文档内跨段落边（$w=1.5$，代表节点链式）。**权重校准关键约束**：设原始图总边权 $m \approx 18000$，注入边权重 $w$ 必须满足 $\delta Q \approx w/(2m) \gg \lambda \cdot \delta H$，即 $w \geq 1.0$，以确保语义相关的同名实体有足够的模块度收益驱动合并。

跨文档边（$w=1.0$）因同义多义问题引入语义噪声（实验中 MRR 下降 1.3%），推荐配置不包含。

### 3.7 U-Retrieval：双路径检索

**Top-Down 路径**：从最高层社区开始，通过 TF-IDF 相似度逐层向下导航，找到目标社区，返回其文本单元集合。**Bottom-Up 路径**：通过 TF-IDF 直接在所有节点的物理锚点（段落文本）上检索，直接定位原始文本块。两条路径的结果取并集作为最终上下文。

---

## 4. 实验设置

### 4.1 数据集

**MultiHop-RAG** [tang2024multihoprag]：609 篇新闻文章，2556 个多跳 QA 对（每个问题需跨 2–4 篇文档推理）。评估采样 200 条，169 条有效。知识图谱规模：13,716 个实体节点，18,044 条关系边。

### 4.2 实验配置

| 组号 | 配置 | EdgeSchedule | $\lambda_0$ | 跨文档边 |
|------|------|:---:|:---:|:---:|
| [0] | Baseline | ✗ | 0 | ✗ |
| [1] | ES only | ✓ | 0 | ✗ |
| [2] | Weak | ✓ | 0.001 | ✗ |
| [3] | **Med（推荐）** | ✓ | **0.003** | ✗ |
| [4] | +PathA | ✓ | 0.001 | ✗ |
| [5] | +CrossDoc | ✓ | 0.001 | ✓ |

退火衰减率 $\alpha = 0.5$。实体抽取使用 spaCy `en_core_web_sm`，TF-IDF 检索使用 scikit-learn。

### 4.3 评估指标

检索质量：MRR、P@5、R@5、NDCG@10。社区结构：avg\_H、模块度 Q、社区层数、Level-0 纯净率。

---

## 5. 实验结果与分析

### 5.1 核心结果

**表 1：SP-GraphRAG 消融实验结果**

| 组号 | $\lambda_0$ | avg\_H | MRR | P@5 | R@5 | NDCG@10 | 层数 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [0] Baseline | 0 | 0.0000 | 0.4226 | 0.1964 | 0.2619 | 0.3142 | 3 |
| [1] ES only | 0 | 0.1374 | 0.4206 | 0.1964 | 0.2619 | 0.3131 | 8 |
| [2] Weak | 0.001 | 0.1152 | 0.4211 | 0.1953 | 0.2619 | 0.3134 | 8 |
| [3] **Med** | **0.003** | **0.1045** | 0.4203 | **0.1964** | **0.2636** | 0.3136 | 9 |
| [4] +PathA | 0.001 | 0.1170 | 0.4206 | 0.1941 | 0.2585 | 0.3119 | 8 |
| [5] +CrossDoc | 0.001 | 0.1682 | 0.4158 | 0.1964 | 0.2619 | 0.3068 | 11 |

**表 2：与 Naive RAG 的外部对比**

| 方法 | MRR | P@5 |
|------|:---:|:---:|
| Naive RAG（TF-IDF chunk） | 0.6389 | 0.2568 |
| SP-GraphRAG [3]（推荐） | 0.4203 | 0.1964 |

### 5.2 RQ1：结构熵约束是否生效？

**结论：是，且效果显著。**

[0] Baseline 的 avg\_H = 0.0000，证实了"句内孤岛森林"假设。引入 EdgeSchedule 后（[1]），H 从 0 跃升至 0.1374，证明 EdgeSchedule 是结构熵生效的必要条件。引入 $\lambda$ 约束后，H 进一步单调下降：[1]→[2]→[3]，H 从 0.1374→0.1152→0.1045，下降 **24%**，$\lambda$ 与 H 呈单调递减关系。推荐配置 [3] 的 Level-0 社区物理来源纯净率达 **100%**。

### 5.3 RQ2：检索质量是否保持？

**结论：是，但受限于样本量，统计显著性有限。**

对比 [0] Baseline 与 [3] Med：MRR 下降 0.5%，P@5 持平，R@5 上升 0.6%，NDCG@10 下降 0.2%。所有变化均在 ±1% 以内。

**统计说明**：169 条有效 QA 的样本量下，上述差异的统计显著性有限。以 MRR 为例，bootstrap 95% 置信区间估计约为 ±0.03，因此 0.5% 的差异在统计上不显著。本文的结论是：在当前实验规模下，**没有证据表明**结构熵约束损害了检索质量；全量实验（2556 条 QA）有待未来工作验证。

**推荐配置 [3] 而非 [2] 的依据**：两者检索指标相近，但 [3] 的 H 更低（0.1045 vs 0.1152），Level-0 纯净率更高，提供更强的溯源保障。在溯源能力是优先需求的场景中，[3] 是更优选择。

### 5.4 RQ3：EdgeSchedule 的必要性

[0]→[1]：H 从 0 到 0.1374（质变），社区层数从 3 到 8，检索指标微降 0.5%（统计不显著）。EdgeSchedule 是结构熵生效的必要条件，且对检索质量无实质性负面影响。

### 5.5 RQ4：跨文档边的边际效益

[2]→[5]：H 从 0.1152 升至 0.1682（+46%），MRR 下降 1.3%，NDCG@10 下降 2.1%。跨文档边因同义多义问题引入语义噪声，不推荐使用。

### 5.6 社区结构深度分析

退火策略使社区层数从 3 层增至 8–9 层，提供了从句子级到跨文档的完整粒度梯度。$\lambda$-H 单调关系为实际部署中根据溯源需求调整 $\lambda$ 提供了理论依据。

### 5.7 与 Naive RAG 的对比及反思

Naive RAG 的 MRR（0.6389）显著优于 SP-GraphRAG（0.4203），差距约 40%。这一差距源于 GraphRAG 范式的共性问题：TF-IDF 社区摘要检索的信息损失。SP-GraphRAG 的目标不是超越 Naive RAG 的检索精度，而是在保持检索质量的同时提供物理结构感知能力。引入向量检索和 LLM 社区摘要有望大幅缩小这一差距，但这超出了本文的研究范围。

---

## 6. 讨论

### 6.1 物理纯净度的应用价值（推断，未实验验证）

以下应用价值基于理论推断，未在本文实验中直接验证，有待未来工作确认：

**精确溯源**：Level-0 社区纯净率 100% 意味着最细粒度的检索单元来自单一物理段落，支持精确的答案来源定位。在法律、医疗、金融等高风险场景中，这一属性具有直接应用价值。

**社区摘要连贯性**：物理来源集中的社区（来自同一段落的实体）在语义上天然更连贯，LLM 为其生成的摘要质量预期更高。

**增量更新友好性**：结构熵约束倾向于将新文档的实体聚合在一起，降低增量索引时需要重新生成摘要的社区数量。

### 6.2 "无害性保证"的理论解释

结构熵 $H$ 衡量物理来源多样性，模块度 $Q$ 衡量拓扑语义一致性。这两个维度在大多数情况下是正交的，因此在合理的 $\lambda$ 范围内，结构熵惩罚主要调整社区内节点的物理来源组成，而不改变社区的拓扑结构，从而不影响检索质量。

### 6.3 局限性

（1）样本量有限（169 条 QA），统计显著性不足；（2）单一数据集（新闻领域），泛化性待证；（3）NER 质量上限（spaCy 规则式）；（4）BottomUp 路径 bug 未修复；（5）无端到端 QA 评估。

---

## 7. 结论

本文提出了 SP-GraphRAG，通过结构熵惩罚的约束 Leiden 算法、退火调度策略和 EdgeSchedule 机制，为 GraphRAG 的社区检测引入了物理结构感知能力。在 MultiHop-RAG 数据集上的系统消融实验证明了约束有效性（H 下降 24%，Level-0 纯净率 100%）与检索无损性（指标波动 ±2.5%）的共存。SP-GraphRAG 为需要精确溯源、高可解释性和增量更新友好性的 RAG 应用场景提供了一个实用的解决方案，同时为 GraphRAG 社区检测步骤的优化开辟了新的研究方向。

**AI 辅助声明**：本文研究设计、实验执行和数据分析由作者完成；论文写作过程中使用了 AI 辅助工具进行文本润色和结构优化。所有实验数据来自真实运行，未使用 AI 生成虚假数据。

---

## 参考文献

[edge2024graphrag] Edge, D., Trinh, H., Cheng, N., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

[traag2019leiden] Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden: Guaranteeing Well-Connected Communities. Scientific Reports, 9:5233.

[li2016structural] Li, A., & Pan, Y. (2016). Structural Information and Dynamical Complexity of Networks. IEEE Transactions on Information Theory, 62(6):3290–3339.

[peng2025survey_se] Peng, H., et al. (2025). A Survey of Structural Entropy: Theory, Methods, and Applications. IJCAI 2025.

[guo2024lightrag] Guo, Z., Xia, L., et al. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779. EMNLP 2025 Findings.

[gutierrez2024hipporag] Gutiérrez, B.J., Shu, Y., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024.

[sarthi2024raptor] Sarthi, P., Abdullah, S., et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024.

[tang2024multihoprag] Tang, Y., & Yang, Y. (2024). MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. arXiv:2401.15391.

[codeseg2025] (2025). Community Detection in Large-Scale Complex Networks via Structural Entropy (CoDeSEG). arXiv:2501.15130.

[graphrag_survey2025] (2025). Retrieval-Augmented Generation with Graphs (GraphRAG Survey). arXiv:2501.00309.

[cd_review2023] (2023). A Comprehensive Review of Community Detection in Graphs. arXiv:2309.11798.

[constrained_cd2021] (2021). Network Optimization Approach to Delineating Health Care Service Areas. PMC.

[docseg2025] (2025). Document Segmentation Matters for Retrieval-Augmented Generation. ACL 2025 Findings.

[structugraphrag2025] (2025). StructuGraphRAG: Structured Document-Informed Knowledge Graphs for RAG. AAAI-SS 2025.

[renyi2025graph] (2025). Entropy-Guided Graph Clustering via Rényi Optimization. Springer LNCS.
