# SP-GraphRAG 实验数据审计报告

**审计日期：** 2026-05-28
**审计范围：** v7 / v8 / v8b / v9 / v10 / v11 / v11b / v11_lambda（全部实验结果文件）
**审计方法：** 直接读取 JSON 结果文件，与记忆文件及汇报文档逐项核对

---

## 一、审计结论总览

| 级别 | 问题数 | 处置方式 |
|------|------|--------|
| 🔴 数据错误（必须修正） | 3 | 已全部修正 |
| 🟡 方法论问题（需声明） | 2 | 已在文档中补充说明 |
| 🟢 已验证无误 | 全部核心指标 | — |

---

## 二、🔴 数据错误（已修正）

### 错误 1：meeting-report.md 中 EM 绝对值全部错误

**发现位置：** `meeting-report.md` v11/v11b 对比表

| 指标 | 错误值（原报告） | 正确值（文件实测） | 来源文件 |
|------|--------------|----------------|--------|
| A+VS EM (n=881) | 0.158 | **0.0885** | `results_v11b/multihop_results_n1000.json` |
| B3+VS EM (n=881) | 0.221 | **0.1237** | `results_v11b/multihop_results_n1000.json` |
| A+VS MRR (n=881) | 0.464 | **0.4404** | 同上 |
| B3+VS MRR (n=881) | 0.515 | **0.4886** | 同上 |

> **相对提升 +39.8% 是正确的**（错误仅在绝对值）。已修正 meeting-report.md。

---

### 错误 2：meeting-report.md E3b 表格绝对值错误

**发现位置：** `meeting-report.md` 阶段四 E3b 表格

| 条件 | 错误无摘要 MRR | 正确无摘要 MRR | 错误有摘要 MRR | 正确有摘要 MRR | 来源文件 |
|------|-------------|-------------|-------------|-------------|--------|
| 标准 Leiden (λ=0) | 0.464 | **0.4231** | 0.477 | **0.4347** | `e3_supplementary.json` |
| 约束 Leiden (λ=0.003) | 0.464 | **0.4464** | 0.515 | **0.4951** | `e3_supplementary.json` |

> **原报告将两组无摘要 MRR 均设为 0.464，实为编造。** 摘要收益比例（+2.8% / +10.9% / 3.9 倍）正确。已修正 meeting-report.md。
> 
> **新发现：** 约束 Leiden 无摘要（0.4464）已超过标准 Leiden 无摘要（0.4231），提升 +5.5%。这说明约束机制本身对检索有独立贡献，不依赖 LLM 摘要。

---

### 错误 3：memory 文件缺失 v11b 详细数据 / "双样本" 叙事不成立

**发现位置：** `memory/experiment_results.md`（缺失 v11b 详细行）；`memory/paper_strategy.md`（声称 "v11(n=429) + v11b(n=881) 双样本"）

**实际文件状态：**
- `results_v11/multihop_results_n500.json`：**仅有 D+V（BottomUp-only）一个组**，无 A+VS vs B3+VS 对比
- `results_v11b/multihop_results_n1000.json`：有 A+VS + B3+VS 完整对比

**结论：** 宣称的 "v11(n=429) 核心对比" 对应的结果文件已不存在（可能被覆盖），**论文中的主要统计依据只有 v11b（n=881）**，不存在双样本验证。

> 已修正 memory 文件，删除双样本叙事。meeting-report.md 中已将 "n=429（v11）" 列移除。

---

## 三、🟡 方法论问题（已在文档中声明）

### 问题 4：EM/Token-F1 的上下文长度不均等（潜在混淆变量）

| 配置 | avg_context_chars | 说明 |
|------|-----------------|------|
| A+VS（标准 GraphRAG） | **594** | 标准 Leiden 社区数少、摘要短，检索到的内容少 |
| B3+VS（SP-GraphRAG） | **1853** | 约束 Leiden 社区更细、摘要质量高，检索到的内容多 |

- 差值：B3+VS 获得约 **3.1 倍** 的 LLM 输入上下文
- **MRR 不受影响**（衡量检索排序，不调用 LLM）
- **EM/Token-F1 可能有部分提升来自"更多信息"而非"社区质量"**

**现有对照实验 (`token_budget_comparison.json`) 不能消除此混淆：** 该文件对比的是 "D+V BottomUp-only（等预算 1744 chars）vs B3+VS"，**并非 A+VS vs B3+VS 的等预算对比**。

**答辩建议：** 主要论据以 MRR（+10.9%，paired CI=[0.024, 0.080]，统计显著）为核心，EM/F1 作为辅助指标并主动声明此局限性。

---

### 问题 5：两个辅助文件命名误导

| 文件 | 文件名暗示 | 实际内容 |
|------|---------|--------|
| `f1_bootstrap_ci.json` | Token-F1 的 Bootstrap CI | 按题型分组的 MRR Bootstrap CI |
| `f1_question_type.json` | 按题型的 Token-F1 | 按题型分组的 MRR（不含 F1） |

> 不影响数据正确性，文件内容本身无误，但命名会导致混淆。已在 memory 中标注。

---

## 四、🟢 已验证无误的数据

| 数据 | 来源 | 验证结论 |
|------|------|--------|
| v11b MRR: A+VS=0.4404, B3+VS=0.4886 | `results_v11b/*.json` | ✅ |
| Paired Bootstrap MRR: diff=0.052, CI=[0.024, 0.080], significant | `paired_bootstrap.json` | ✅ |
| v11b Bootstrap CI: A+VS [0.410, 0.471], B3+VS [0.458, 0.516] | `results_v11b/*.json` | ✅ |
| v11b Token-F1: A+VS=0.1267, B3+VS=0.1844 (+45.5%) | `results_v11b/*.json` | ✅ |
| E3b 摘要收益比: 标准+2.8%, 约束+10.9%, 比值=3.9x | `e3_supplementary.json` | ✅ |
| v7/v8 六组消融 MRR/CI 数值 | `results_v7/`, `results_v8/` | ✅ |
| v8b TopDown vs U-Retrieval 对比 | `results_v8b/*.json` | ✅ |
| v9 LLM summary 实验数值 | `results_v9/*.json` | ✅ |
| v10 大样本 TF-IDF 结果 | `results_v10/*.json` | ✅ |
| valid_qa 一致性（429 / 881） | 跨文件核查 | ✅ |
| 所有文件 Bootstrap CI lower/upper/n 一致 | 跨文件核查 | ✅ |

---

## 五、不需要补跑的项目

| 项目 | 理由 |
|------|------|
| v7/v8/v8b/v9/v10 补跑 EM/Token-F1 | 这些实验在 TF-IDF 框架下，QA 评估无意义；向量检索框架的 QA 指标已由 v11b 覆盖 |
| 等预算 A+VS vs B3+VS 对照实验 | 核心论证以 MRR 为主，上下文差异作为局限性声明已足够支撑答辩 |
| v11 重跑（n=429 A+VS vs B3+VS） | v11b（n=881）已提供更大样本的完整对比，小样本重复无增量价值 |

---

## 六、修改文件记录

| 文件 | 修改内容 |
|------|--------|
| `meeting-report.md` | 修正 v11b EM 绝对值；修正 E3b 绝对值；补充 Token-F1 行；移除未经验证的 v11 n=429 列；新增上下文长度局限性条目 |
| `memory/experiment_results.md` | 新增 v11b 详细数据表（含 EM/Token-F1/CI）；新增 E3a/E3b 表格（正确值）；更新 Key Conclusions |

---

> **审计结论：** MRR 核心数据全部准确，统计显著性成立。EM/Token-F1 相对提升比例正确，但绝对值此前有误（已修正）。上下文长度不均等是真实的方法论局限，建议在论文局限性章节中主动声明。
