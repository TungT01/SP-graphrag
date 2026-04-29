# 第四章 实验设置

## 4.1 数据集

本文使用 **MultiHop-RAG** [tang2024multihoprag] 作为评估数据集。MultiHop-RAG 是首个专注于多跳查询的 RAG benchmark，包含 609 篇新闻文章构成的知识库和 2556 个多跳问答对，每个问题的答案需要跨 2–4 篇文档推理。数据集涵盖四种查询类型：推理型、比较型、时序型和无效查询，全面考察 RAG 系统的多跳检索能力。

**评估规模**。受计算资源限制，本文从 2556 个问答对中随机抽样 200 条进行评估，其中 169 条有效（排除了 ground-truth 文档不在语料库中的问题）。

**知识图谱规模**。对全部 609 篇文章运行实体抽取后，得到：
- 实体节点：13,716 个
- 关系边（句内）：18,044 条
- 平均每篇文章实体数：22.5 个

## 4.2 知识图谱构建

**实体抽取**。使用 spaCy（`en_core_web_sm` 模型）进行命名实体识别（NER），抽取 PERSON、ORG、GPE、EVENT、PRODUCT 等类型的实体。同时运行依存句法分析，抽取主谓宾三元组作为关系边。应用 150+ 停用词过滤低质量实体（如代词、通用名词）。

**物理锚定**。每个实体节点的 ID 编码为 `{doc_id}-{para_id}-{sent_id}-{entity_name_normalized}`，保留三级物理坐标。不做跨文档实体消解，同名实体在不同句子中视为独立节点。

**EdgeSchedule 配置**。注入三类同名实体边：段落内跨句子边（$w=2.0$，链式连接）、文档内跨段落边（$w=1.5$，代表节点链式）。推荐配置不包含跨文档边（原因见 §5.5）。

## 4.3 实验配置

本文设计 6 组消融实验，系统验证各组件的贡献：

| 组号 | 名称 | EdgeSchedule | $\lambda_0$ | 跨文档边 | Path A |
|------|------|:---:|:---:|:---:|:---:|
| [0] | Baseline | ✗ | 0 | ✗ | ✗ |
| [1] | ES only | ✓ | 0 | ✗ | ✗ |
| [2] | Weak constraint | ✓ | 0.001 | ✗ | ✗ |
| [3] | Med constraint（推荐） | ✓ | 0.003 | ✗ | ✗ |
| [4] | +PathA | ✓ | 0.001 | ✗ | ✓ |
| [5] | +CrossDoc | ✓ | 0.001 | ✓ | ✗ |

其中 Path A 指在 EdgeSchedule 中额外注入基于路径相似度的边；+CrossDoc 指加入跨文档同名实体边（$w=1.0$，上限 5 条）。退火衰减率统一为 $\alpha = 0.5$。

## 4.4 评估指标

**检索质量指标**（主要）：
- **MRR**（Mean Reciprocal Rank）：衡量第一个相关文档的排名
- **P@5**（Precision at 5）：前 5 个检索结果中相关文档的比例
- **R@5**（Recall at 5）：前 5 个检索结果覆盖的相关文档比例
- **NDCG@10**（Normalized Discounted Cumulative Gain at 10）：综合考虑排名和相关性的指标

**社区结构指标**（次要）：
- **avg\_H**：全局平均结构熵，衡量社区物理来源混杂程度
- **模块度 Q**：社区检测质量的传统指标
- **社区层数**：层次化结构的深度
- **Level-0 纯净率**：最底层社区中物理来源单一（$H=0$）的社区比例

**Ground-truth 定义**。对于每个查询，ground-truth 为 MultiHop-RAG 标注的支持文档集合（supporting evidence documents）。检索结果中包含 ground-truth 文档的视为相关。

## 4.5 对比基线

**Naive RAG**：基于 TF-IDF 的 chunk-level 检索，将每篇文章切分为固定长度文本块（chunk size = 512 tokens，overlap = 50 tokens），构建 TF-IDF 索引，直接检索与查询最相关的文本块。Naive RAG 作为外部对比基线，不参与消融实验的内部对比。

**SP-GraphRAG [0] Baseline**：无 EdgeSchedule、$\lambda=0$ 的标准 Leiden 社区检测，作为消融实验的内部基线。

## 4.6 实现细节

所有实验在 Python 3.11 环境下运行，使用 `leidenalg` 库作为 Leiden 算法的基础实现，在其 `local_moving` 阶段注入结构熵惩罚项。TF-IDF 检索使用 `scikit-learn` 的 `TfidfVectorizer`。实验在配备 Apple M 系列芯片的 MacBook 上运行，单次完整实验（索引 + 200 QA 评估）耗时约 15–30 分钟。
