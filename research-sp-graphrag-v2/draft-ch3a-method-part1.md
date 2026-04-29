# 第三章 方法（上）：问题形式化与核心算法

## 3.1 问题形式化

**知识图谱构建**。给定文档集合 $\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$，每篇文档 $d_i$ 被分割为段落序列，每个段落进一步分割为句子序列。对每个句子 $s$ 运行命名实体识别（NER）和依存句法分析，抽取实体集合 $V_s$ 和关系集合 $E_s$（包含共现边和依存三元组边）。所有句子的实体和关系聚合形成知识图谱 $G = (V, E)$。

**物理坐标**。每个实体节点 $v \in V$ 携带三级物理坐标 $\phi(v) = (\text{doc\_id}, \text{para\_id}, \text{sent\_id})$，表示该实体来自哪篇文档的哪个段落的哪个句子。节点 ID 编码为 $\text{id}(v) = \texttt{\{sent\_id\}-\{entity\_name\}}$，保留完整的物理出处链。

**社区检测目标**。给定知识图谱 $G$，目标是找到节点划分 $\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$，使得：

$$J(\mathcal{C}) = Q(\mathcal{C}) - \lambda \cdot H(\mathcal{C})$$

其中 $Q(\mathcal{C})$ 为 Leiden 模块度，$H(\mathcal{C})$ 为全局平均结构熵，$\lambda \geq 0$ 为惩罚系数。

## 3.2 物理锚定表示

**设计原则**。标准 GraphRAG 在实体抽取后即丢弃物理出处信息，仅保留实体名称和关系类型。SP-GraphRAG 采用"物理优先"（instance-level）策略：不做跨文档实体消解，同名实体在不同句子中视为不同节点，每个节点保留其唯一的物理坐标。

**三级坐标系统**。节点的物理坐标 $\phi(v)$ 构成一个三层层次结构：

$$\text{sent\_id} \subset \text{para\_id} \subset \text{doc\_id}$$

例如，节点 `doc1-p0-s2-Apple` 表示来自文档 `doc1` 第 0 段第 2 句的实体 `Apple`。这一编码方案使得从节点 ID 即可直接解析物理出处，无需额外的元数据查询。

**物理单元定义**。在计算结构熵时，"物理单元"（physical unit）的粒度可以是句子级、段落级或文档级。本文默认使用**段落级**（para\_id）作为物理单元，因为段落是语义完整性和物理粒度之间的最佳平衡点——句子级粒度过细（导致 $H$ 过高），文档级粒度过粗（无法区分同文档不同段落的实体）。

## 3.3 结构熵惩罚项

**定义**。对于社区 $C_k$，设其包含 $|C_k|$ 个节点，来自 $M_k$ 个不同的物理单元（段落）。设第 $i$ 个物理单元在 $C_k$ 中的节点比例为 $p_{k,i} = n_{k,i} / |C_k|$，其中 $n_{k,i}$ 为来自第 $i$ 个物理单元的节点数。则社区 $C_k$ 的结构熵为：

$$H(C_k) = -\sum_{i=1}^{M_k} p_{k,i} \log_2 p_{k,i}$$

当 $M_k = 1$（所有节点来自同一物理单元）时，$H(C_k) = 0$（最纯净）；当节点均匀分布于 $M_k$ 个物理单元时，$H(C_k) = \log_2 M_k$（最混杂）。

全局平均结构熵为：

$$H(\mathcal{C}) = \frac{1}{K} \sum_{k=1}^{K} H(C_k)$$

**与 Li & Pan 结构信息论的关系**。Li 和 Pan [li2016structural] 定义的结构熵基于编码树，度量图的全局结构复杂性。本文的 $H_{\text{structure}}$ 是其简化版本：我们不寻求最优编码树，而是直接将节点的物理来源分布视为概率分布，计算其 Shannon 熵。这一简化使得计算高效（$O(|C_k|)$），且物理意义直观——$H(C_k)$ 直接度量社区 $C_k$ 内节点物理来源的混杂程度。

## 3.4 增量 ΔH 计算

**挑战**。Leiden 的 local moving 阶段需要对每个节点评估将其移入/移出每个邻居社区的收益 $\Delta J$。若每次移动后重新计算 $H$，时间复杂度为 $O(|C_k|)$，在大规模图上不可接受。

**解决方案：计数器维护**。为每个社区 $C_k$ 维护一个物理单元计数器字典 $\text{cnt}_k$，其中 $\text{cnt}_k[u]$ 记录来自物理单元 $u$ 的节点数。当节点 $v$（来自物理单元 $\phi(v)$）从社区 $C_{\text{src}}$ 移动到社区 $C_{\text{dst}}$ 时，增量 $\Delta H$ 的计算分为两步：

**移出操作**（$C_{\text{src}}$ 的熵变化）：

$$\Delta H_{\text{remove}}(C_{\text{src}}, v) = H(C_{\text{src}} \setminus \{v\}) - H(C_{\text{src}})$$

设 $n = |C_{\text{src}}|$，$c = \text{cnt}_{\text{src}}[\phi(v)]$，则：

$$\Delta H_{\text{remove}} = \frac{n}{n-1} H(C_{\text{src}}) - \frac{1}{n-1}\left[c \log_2 c - (c-1)\log_2(c-1)\right] \cdot \frac{1}{n-1}$$

实际实现中，通过直接更新计数器并重新计算受影响项，每次操作仅需 $O(1)$ 的算术运算（更新两个计数器项，重新计算两个 $p \log p$ 项）。

**移入操作**（$C_{\text{dst}}$ 的熵变化）：对称地，$\Delta H_{\text{add}}(C_{\text{dst}}, v)$ 同样为 $O(1)$。

因此，每次节点移动的 $\Delta J = \Delta Q - \lambda \cdot (\Delta H_{\text{remove}} + \Delta H_{\text{add}})$ 可在 $O(1)$ 时间内计算，保证了约束 Leiden 算法与原始 Leiden 相同的时间复杂度量级。
