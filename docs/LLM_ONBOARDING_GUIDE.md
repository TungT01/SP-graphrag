# 如何让一个新 LLM 完整理解本项目

> 这份文档是给**你（项目作者）**看的操作指南，不是给 LLM 看的。
> 它告诉你：该准备哪些文件、按什么顺序喂、用什么提示词、怎么验证 LLM 真的理解了。

---

## 一、核心原则

**别一次性把所有文件扔进去。** 本项目全部文档 + 代码约 6 万 token，即使上下文窗口够大，一次性灌入会导致 LLM 对关键信息（如"结构熵为什么失效"）的注意力被大量胶水代码稀释。

正确做法是 **分三轮投喂，每轮给一个明确任务**。

---

## 二、文件清单与分组

### 第一轮：建立全局认知（~7K token）

| 文件 | 作用 | 必须 |
|---|---|---|
| `docs/LLM_REFERENCE.md` | 单一权威参考：公式、API、模块、最新实验数字 | ✅ |
| `docs/EXPERIMENT_RESULTS.md` | 所有实验结果权威记录（v7-v11b+E3） | ✅ |
| `config.yaml` | 全部配置参数 | ✅ |
| `constrained_leiden/__init__.py` | 公开 API 清单，理解模块边界 | ✅ |
| `experiments/results_v11b/multihop_results_n1000.json` | v11b 核心实验结果（n=881，当前最终数据） | ✅ |

### 第二轮：理解核心算法（~14K token）

| 文件 | 作用 | 必须 |
|---|---|---|
| `constrained_leiden/physical_anchor.py` | PhysicalNode + 结构熵计算 | ✅ |
| `constrained_leiden/annealing.py` | λ 退火 4 种曲线 | ✅ |
| `constrained_leiden/leiden_constrained.py` | 层次化 Leiden 核心算法 | ✅ |
| `constrained_leiden/edge_scheduler.py` | v5 分层加边调度器 | ✅ |

### 第三轮：理解完整 Pipeline（~30K token，按需）

| 文件 | 作用 | 何时需要 |
|---|---|---|
| `constrained_leiden/graphrag_workflow.py` | 主入口，图构建 + 调用链 | 需要改接口时 |
| `extraction/extractor.py` | 实体/关系抽取 | 需要改抽取逻辑时 |
| `data/ingestion.py` | 文档→段落→句子切分 | 需要改数据流时 |
| `retrieval/retriever.py` | URetriever（TF-IDF/向量，6 种模式） | 需要改检索时 |
| `summarization/summarizer.py` | LLM 社区摘要生成（含本地缓存） | 需要改摘要时 |
| `evaluation/evaluator.py` | 检索评估（MRR/P@K + Bootstrap CI） | 需要加检索指标时 |
| `evaluation/qa_evaluator.py` | 端到端 QA 评估（EM/F1，需 LLM） | 需要 QA 评估时 |
| `evaluation/summary_quality_evaluator.py` | 摘要质量评估（LLM 打分） | 需要摘要质量分析时 |
| `experiments/run_multihop_eval.py` | 主实验脚本（4 组向量检索配置） | 需要跑实验时 |

### 不要喂的文件

| 文件 | 原因 |
|---|---|
| `pipeline_config.py` | config.yaml 解析器，无独占信息 |
| `docs/archive/*.md` | 均为 v5 时代过时文档，已归档 |
| `research-v1-archive/` | v1 论文草稿，已被 v2 取代 |

---

## 三、提示词模板

### 第一轮提示词：建立全局认知

```markdown
你即将学习一个研究项目——GraphRAG-Improved。这个项目的目标是改进 GraphRAG
的社区检测阶段，通过注入结构熵惩罚来提升多跳问答的检索效果。

我会给你 5 个文件。请按以下顺序阅读：

1. docs/LLM_REFERENCE.md — 单一权威参考，所有数字和 API 以它为准
2. docs/EXPERIMENT_RESULTS.md — 所有实验结果（v7-v11b+E3）
3. config.yaml — 项目默认配置
4. constrained_leiden/__init__.py — 核心模块公开 API 清单
5. experiments/results_v11b/multihop_results_n1000.json — 最终核心实验数据（n=881）

阅读完成后，请回答以下验证问题（不要猜测，如果信息不足请明确说"文档中未提及"）：

Q1: 项目的核心目标函数是什么？写出完整公式和每个符号的含义。
Q2: 为什么早期实验（v4/v7）中结构熵全部为 0？根本原因链是什么？
Q3: EdgeSchedule 的权重为什么必须 ≥ 1.0？用量级分析说明。
Q4: 在向量检索框架下，SP-GraphRAG（B3+VS）比原版 GraphRAG（A+VS）MRR 提升多少？该结论在多大样本下得到验证？
Q5: 为什么 TF-IDF 框架下约束无效，而向量检索框架下有效？
Q6: URetriever 支持哪 6 种 retrieval_mode？向量模式的前缀是什么？
Q7: 传导机制实验（E3b）说明了什么？约束 Leiden 的摘要收益是标准 Leiden 的几倍？

如果你能准确回答全部 7 个问题，说明你已经建立了对项目的全局认知。
如果有任何问题无法回答或不确定，请指出具体哪个问题以及缺少什么信息。
```

### 第二轮提示词：深入核心算法

```markdown
你已经对 GraphRAG-Improved 项目有了全局认知。现在我给你核心算法的完整实现代码
（4 个文件），请深入理解算法细节。

阅读顺序：
1. physical_anchor.py — 先理解 PhysicalNode 数据结构和结构熵的计算方式
2. annealing.py — 理解 λ 退火机制
3. leiden_constrained.py — 核心：带结构熵惩罚的层次化 Leiden 算法
4. edge_scheduler.py — v5 新增的分层加边调度器

阅读后请回答：

Q1: CommunityEntropyState 的 delta_entropy_if_add 方法的时间复杂度是多少？
    它是如何避免每次都重新计算整个社区熵的？
Q2: _local_moving_phase 中，节点移动的决策标准是 ΔJ = ΔQ - λ·ΔH。
    请描述当 ΔQ > 0 但 ΔH 也 > 0 时，λ 的值如何影响决策。
Q3: hierarchical_leiden_constrained 的三个终止条件分别是什么？
    v5 移除了哪个终止条件？为什么？
Q4: EdgeSchedule.build 在 Level 1/2/3 分别注入什么范围的边？
    为什么权重随层级递减（0.3 → 0.2 → 0.1）？
Q5: 如果让你改进这个算法，你会从哪里入手？给出至少 2 个具体方向和理由。
```

### 第三轮提示词：执行具体改进任务

```markdown
你已经完全理解了 GraphRAG-Improved 项目的设计和实现。现在我需要你帮我完成
一个具体任务：

[在这里描述你的具体需求，例如：]

任务：修复 BottomUpRetriever 的锚点匹配 Bug

背景（来自 LLM_REFERENCE.md §5.2）：
- _entity_chunks 存储 entity_title → Set[sent_id]
- 但 text_units 的 chunk_id = para_id
- sent_id 格式 "{doc_id}-p{NNN}-s{NNN}" ≠ para_id 格式 "{doc_id}-p{NNN}"
- 导致锚点加权 ×1.5 从不触发

要求：
1. 修改 retriever.py 中的相关逻辑
2. 保持向后兼容（不改变 URetriever 的公开接口）
3. 添加单元测试验证修复有效
4. 说明修复后预期对检索指标的影响

请先阅读 retriever.py，然后给出修改方案。
```

---

## 四、验证 LLM 是否真正理解

第一轮 7 个问题的**标准答案**（你可以用来判分）：

**Q1 标准答案**：`J = Q_leiden − λ · H_structure`。Q_leiden 是模块度增量，H_structure = −Σ p_i · log(p_i) 是社区内来源文档分布的信息熵，λ 是惩罚强度（由退火策略控制）。

**Q2 标准答案**：v4 使用 `anchor_granularity="sent"`，每个 PhysicalNode 的 chunk_ids 只含一个唯一 sent_id。社区内每个节点的 chunk 各不相同，绝大多数社区为单节点（Level 0 平均社区大小 2.06），导致 avg_structural_entropy 在四位小数精度下显示为 0.0000。结构熵惩罚项完全失效。

**Q3 标准答案**：(1) `anchor_granularity` 改为 `"para"`，同段落多节点共享 para_id 使 H > 0；(2) EdgeSchedule 分 3 级注入跨句/跨段/跨文档边，解决底层图断裂为 ~24,858 个连通分量的问题；(3) 移除 `lambda_val < 1e-6: break` 终止条件，防止 λ 衰减导致提前终止。

**Q4 标准答案**：Our Method 最佳组 MRR = 0.3552，Naive RAG MRR = 0.6389，差距 = 0.6389 − 0.3552 = 0.2837（落后约 44%）。

**Q5 标准答案**：`_entity_chunks` 存的是 sent_id 格式（如 `xxx-p001-s002`），但 text_units 的 chunk_id 是 para_id 格式（如 `xxx-p001`）。L438 处 `chunk_id in anchor_chunk_ids` 比较时，两种格式永远不相等，导致锚点命中的 ×1.5 加权从不触发，BottomUp 退化为纯 TF-IDF。

**Q6 标准答案**：Level 2 注入 intra_document 边（同文档跨段落的同名实体），默认权重 0.2。

**Q7 标准答案**：`run_constrained_community_detection` 的 `edge_schedule` 参数类型是 `Optional[EdgeSchedule]`（对象，不是 bool）。实验脚本 RunConfig 中的 `use_edge_schedule` 是 `bool`，在 `run_one()` 函数中做转换：`if cfg.use_edge_schedule: edge_schedule = EdgeSchedule.build(entities_df, include_cross_doc=cfg.edge_schedule_cross_doc)`。

---

## 五、常见踩坑与对策

**坑 1：LLM 引用了 README 或 PROJECT_STATUS 中的过期数字**
对策：在提示词中明确说"LLM_REFERENCE.md 是单一权威来源，与其他文档冲突时以它为准"。

**坑 2：LLM 混淆 Dataset A（60,439 实体）和实际实验数据（47,142 实体）**
对策：不要喂 PROJECT_STATUS.md，它包含历史错误的 Dataset A/B 分组描述。LLM_REFERENCE.md 已修正。

**坑 3：LLM 看到 config.yaml 中 `use_edge_schedule: true` 就以为参数是 bool**
对策：第一轮验证 Q7 就是为了检测这个误解。如果 LLM 答错，补充说明 §8.2 的两层转换。

**坑 4：LLM 读了代码但没理解"为什么 v4 结构熵全零"**
对策：这是理解深度的试金石。如果 Q2 答不上来，让 LLM 重新阅读 LLM_REFERENCE.md §1.3 和 §8.4，并追问"当 anchor_granularity='sent' 时，一个 3 节点社区的 H 值是多少？"

**坑 5：一次性喂太多代码，LLM 开始"创造性地"编造函数签名**
对策：永远先喂 LLM_REFERENCE.md（里面有精确签名），再喂代码。代码是用来补充理解的，不是用来替代文档的。

---

## 六、快速复制清单

如果你要把文件发给另一个 LLM 对话，按这个顺序复制粘贴：

```
第一轮（必须，~7K token）：
  ① docs/LLM_REFERENCE.md
  ② graphrag_improved/config.yaml
  ③ graphrag_improved/constrained_leiden/__init__.py
  ④ graphrag_improved/experiments/results_v11b/multihop_results_n1000.json  ← 核心数据
  ⑤ docs/EXPERIMENT_RESULTS.md

第二轮（理解算法，+14K token）：
  ⑥ graphrag_improved/constrained_leiden/physical_anchor.py
  ⑦ graphrag_improved/constrained_leiden/annealing.py
  ⑧ graphrag_improved/constrained_leiden/leiden_constrained.py
  ⑨ graphrag_improved/constrained_leiden/edge_scheduler.py

第三轮（完整 Pipeline，按需 +30K token）：
  ⑩ graphrag_improved/constrained_leiden/graphrag_workflow.py
  ⑪ 根据任务选择 extractor.py / ingestion.py / retriever.py / evaluator.py / qa_evaluator.py
```
