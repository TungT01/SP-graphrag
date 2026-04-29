# 文件功能清单

> **更新日期**：2026-04-27
>
> 本清单覆盖项目中的每一个文件，标注其功能、所属版本、当前状态。
> 状态图标含义：🟢 活跃 ｜ 🟡 保留但需注意 ｜ 🔴 过时/冗余 ｜ ⚪ 自动生成/缓存

---

## 一、根目录

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `main.py` | 230 | CLI 入口。`python -m graphrag_improved.main` 的唯一入口点，解析命令行参数后调用 `run.py` | 🟢 活跃 |
| `run.py` | 175 | Pipeline 编排核心。串联 ingestion → extraction → community detection → output 四个阶段 | 🟢 活跃 |
| `pipeline_config.py` | 186 | 配置加载与验证。读取 config.yaml 并转换为 dataclass | 🟡 功能被 config.yaml 覆盖，LLM 理解项目时可跳过 |
| `smoke_v5.py` | 141 | v5 冒烟测试。用构造数据验证三大核心修复（para 锚点、EdgeSchedule、终止条件） | 🟢 活跃 |
| `config.yaml` | 96 | 项目主配置。含 input/extraction/clustering/output 四段，v5 新增 `anchor_granularity` 和 `edge_schedule` 子配置 | 🟢 活跃 |
| `__init__.py` | 0 | 空文件，Python 包标识 | 🟢 |
| `.gitignore` | 57 | Git 忽略规则 | 🟢 |

---

## 二、文档

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `README.md` | 267 | 项目总览。面向人类阅读的入口文档，含状态速览、版本演进、实验数据摘要 | 🟢 活跃 |
| `LLM_REFERENCE.md` | 435 | LLM 权威参考。精确 API 签名、数据结构、实验数字、已知 Bug——所有数字以本文档为准 | 🟢 活跃 |
| `LLM_ONBOARDING_GUIDE.md` | 205 | LLM 引导指南。给项目作者的操作手册：分三轮投喂文件 + 提示词模板 + 验证问题 | 🟢 活跃 |
| `FILE_INVENTORY.md` | — | 文件功能清单（本文档） | 🟢 活跃 |
| `PROJECT_STATUS.md` | 348 | 项目现状盘点。含 v4 完整实验数据表格和 Naive RAG 按题型细分数据 | 🟡 80% 内容已被 LLM_REFERENCE.md 精炼覆盖，仅"按题型细分"和"Dataset A 历史数据"为独占信息 |
| `PROJECT_PLAN.md` | 401 | v5 技术方案与实施计划。含六组消融设计、数据来源交叉引用 | 🟡 规划文档，部分内容与代码现状有偏差（如消融组描述与 run_multihop_eval.py 的实际配置有细微差异） |
| `CHANGELOG.md` | 200 | 版本变更日志 v1→v5 | 🟡 信息已被 LLM_REFERENCE.md 和 README.md 覆盖，保留仅因符合开源惯例 |
| `REFACTOR_PROMPT.md` | 513 | v5 重构提示词。原为指导 LLM 完成 v5 代码改造的结构化指令 | 🔴 代码改造部分已 100% 完成，剩余 7 条实验验证清单有参考价值但主体已过时 |

---

## 三、核心算法 `constrained_leiden/`

本目录是项目的算法核心，实现"带结构熵惩罚的层次化 Leiden 社区检测"。

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `__init__.py` | 52 | 包入口。导出 13 个公开符号（AnnealingConfig、EdgeSchedule、PhysicalNode 等），定义了整个包的 API 边界 | 🟢 活跃 |
| `leiden_constrained.py` | 726 | 核心算法。实现 `J = Q − λ·H` 目标函数下的 local moving → refinement → aggregation 三阶段循环，含增量熵状态 O(1) 更新优化 | 🟢 活跃 |
| `graphrag_workflow.py` | 736 | GraphRAG 接口层。将核心算法封装为 `run_constrained_community_detection()` 入口，处理图构建、PhysicalNode 映射、结果转换为 DataFrame | 🟢 活跃 |
| `edge_scheduler.py` | 430 | v5 分层加边调度器。`EdgeSchedule.build()` 构建三级边（Level 1 同段落 w=0.3 → Level 2 同文档 w=0.2 → Level 3 跨文档 w=0.1），每层聚合后注入 | 🟢 活跃（v5 新增） |
| `physical_anchor.py` | 165 | 物理锚点。`PhysicalNode` dataclass + `compute_structural_entropy()` 函数，定义了结构熵的计算方式 | 🟢 活跃 |
| `annealing.py` | 162 | λ 退火调度。`AnnealingConfig` + 四种曲线（exponential/linear/cosine/step） | 🟢 活跃 |

---

## 四、数据 Pipeline 模块

| 目录/文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `data/ingestion.py` | 465 | 数据摄入。Document → TextUnit(段落) → SentenceUnit(句子) 三级切分，分配 doc_id/para_id/sent_id | 🟢 活跃 |
| `data/__init__.py` | 0 | 空 | 🟢 |
| `extraction/extractor.py` | 581 | 实体抽取。spaCy NER + 句内共现窗口 → Entity(含 sent_id/para_id/doc_id) + Relation | 🟢 活跃 |
| `extraction/__init__.py` | 0 | 空 | 🟢 |
| `proposition/transformer.py` | 443 | 命题转换。共指消解 + 复合句原子化，使命题脱离上下文仍可理解 | 🟡 **已实现但未集成**——代码完整，但未被 run.py、main.py 或任何实验脚本调用，属预留模块 |
| `proposition/__init__.py` | 2 | 包标识 | 🟡 |
| `retrieval/retriever.py` | 647 | U-Retrieval 双轨检索。TopDown（社区层级导航）+ BottomUp（物理锚点检索）融合。⚠️ BottomUp 存在锚点匹配 Bug（sent_id vs para_id 格式不匹配导致 ×1.5 加权不生效） | 🟢 活跃，有已知 Bug |
| `retrieval/__init__.py` | 2 | 包标识 | 🟢 |
| `evaluation/evaluator.py` | 652 | 三维评估器。检索质量(P@K/MRR/NDCG) + 社区质量(模块度/结构熵/纯净率) + 文本匹配(EM/F1/ROUGE-L) | 🟢 活跃 |
| `evaluation/__init__.py` | 2 | 包标识 | 🟢 |
| `output/reporter.py` | 778 | 结果输出。生成 parquet/csv/html/summary.txt，含社区结构可视化 HTML 报告 | 🟢 活跃 |
| `output/__init__.py` | 0 | 空 | 🟢 |

---

## 五、实验 `experiments/`

### 5.1 实验主脚本

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `run_multihop_eval.py` | 790 | v5 六组消融实验主脚本。定义 RunConfig + 六组配置 [0]-[5]，支持 `--groups` 选择性运行 | 🟢 活跃，v5 实验尚未运行 |
| `run_experiment.py` | 559 | v4 四组对照实验脚本。仅对比 Baseline vs Ours 两组 + Path A/B 变体 | 🔴 v4 遗留，已被 run_multihop_eval.py 取代。但仍被 smoke_test.py 和 2 个 probe 脚本 import，删除前需迁移依赖 |
| `data_loader.py` | 253 | MultiHop-RAG 数据集加载器。`load_multihop_dataset()` + `try_download_dataset()` | 🟢 活跃 |
| `download_data.py` | 204 | 数据集下载脚本。从 HuggingFace 下载 MultiHop-RAG | 🟢 活跃 |
| `analyze_results.py` | 452 | 结果分析。读取 JSON 结果生成分析报告和可视化 HTML | 🟢 活跃 |
| `smoke_test.py` | 255 | v4 端到端冒烟测试。用 10 篇模拟文章走完整 Pipeline | 🔴 依赖 v4 的 run_experiment.py 接口，不测试 v5 任何新特性。已被根目录 smoke_v5.py 在算法验证层面取代 |
| `stats_community_distribution.py` | 201 | 统计社区层次分布，估算 LLM 摘要生成的调用次数和费用 | 🟡 工具脚本，偶尔使用 |
| `ablation_study_notes.md` | 325 | v4 消融实验记录文档。含 Path A/Path B 方案详细数据 | 🟡 v4 历史记录，有参考价值 |

### 5.2 探针脚本（一次性调试工具，均已完成使命）

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `probe_constraint_effect.py` | 194 | 验证物理约束能否阻止跨文档合并 | 🔴 v1 阶段产物，smoke_v5.py 已覆盖 |
| `probe_entity_chunks.py` | 99 | 分析实体 chunk_ids 分布，诊断"结构熵恒为0" | 🔴 v4 诊断脚本，问题已在 v5 修复 |
| `probe_incremental_entropy.py` | 205 | 验证增量熵状态优化的正确性 | 🔴 v1.1 阶段产物，已完成验证 |
| `probe_leidenalg.py` | 137 | 调研 leidenalg 库扩展能力 | 🔴 技术选型阶段产物 |
| `probe_perf.py` | 243 | cProfile 性能剖析，定位纯 Python Leiden 瓶颈 | 🔴 一次性分析，已完成 |
| `probe_scale.py` | 71 | 真实数据集规模耗时测试 | 🔴 一次性测试 |
| `probe_scale2.py` | 60 | 模拟数据规模耗时测试（probe_scale 的简化版） | 🔴 一次性测试 |

### 5.3 实验结果与缓存

| 路径 | 类型 | 功能 | 状态 |
|---|---|---|---|
| `results/multihop_results_n200.json` | JSON | v4 四组消融实验完整结果（169 条有效 QA）——**唯一权威结果文件** | 🟢 |
| `results/cache/entities_full.parquet` | parquet | 60,439 实体的抽取缓存（原始版，即历史 "Dataset A"） | 🟡 历史缓存，实际实验未使用此版本 |
| `results/cache/entities_full_b.parquet` | parquet | 47,142 实体的抽取缓存（噪声过滤版）——实际实验使用此版本 | 🟢 |
| `results/cache/relationships_full.parquet` | parquet | 84,704 关系的抽取缓存（原始版） | 🟡 同上 |
| `results/cache/relationships_full_b.parquet` | parquet | 61,812 关系的抽取缓存（噪声过滤版）——实际实验使用此版本 | 🟢 |
| `results/cache/entities_n200.parquet` | parquet | 200 条采样对应的实体缓存（原始版） | 🟡 |
| `results/cache/entities_n200_b.parquet` | parquet | 200 条采样对应的实体缓存（过滤版） | 🟢 |
| `results/cache/relationships_n200.parquet` | parquet | 200 条采样对应的关系缓存（原始版） | 🟡 |
| `results/cache/relationships_n200_b.parquet` | parquet | 200 条采样对应的关系缓存（过滤版） | 🟢 |

---

## 六、基线系统 `baselines/`

### 6.1 顶层

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `__init__.py` | 12 | 包说明，列出 naive_rag / graphrag_official / data_loader / run_evaluation 四个子模块 | 🟢 |
| `data_loader.py` | 347 | experiments/data_loader.py 的封装+降级备份，使 baselines 可独立运行 | 🟢 有意设计的解耦 |
| `run_evaluation.py` | 531 | 统一评估入口。支持在 MultiHop-RAG 上对比 naive_rag / graphrag_local / graphrag_global | 🟢 |
| `naive_rag_index.pkl` | 二进制 | Naive RAG 预建向量索引 | 🟢 |
| `README.md` | — | baselines 使用说明 | 🟢 |

### 6.2 Naive RAG `baselines/naive_rag/`

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `__init__.py` | 6 | 包说明：sentence-transformers all-MiniLM-L6-v2 逐句向量化检索 | 🟢 |
| `indexer.py` | 402 | 文档索引器。将语料向量化并持久化为 .pkl | 🟢 |
| `retriever.py` | 368 | 余弦相似度检索器 | 🟢 |
| `evaluator.py` | 533 | 批量评估脚本 | 🟢 |
| `requirements.txt` | 6 | 依赖清单 | 🟢 |

### 6.3 GraphRAG Official `baselines/graphrag_official/`

| 文件 | 行数 | 功能 | 状态 |
|---|---|---|---|
| `__init__.py` | 6 | 包说明 | 🟢 |
| `indexer.py` | 420 | 调用 `python -m graphrag index` 建索引 | 🟢 |
| `retriever.py` | 646 | local_search + global_search 双模式检索 | 🟢 |
| `evaluator.py` | 706 | 官方 GraphRAG 评估 | 🟢 |
| `local_embedding_server.py` | 112 | 本地 OpenAI 兼容 Embedding API 服务（FastAPI） | 🟢 |
| `requirements.txt` | 2 | 依赖：graphrag>=0.3.0, tiktoken | 🟢 |
| `setup.sh` | 213 | 安装脚本 | 🟢 |
| `run_small_test.sh` | 126 | 20 篇小规模测试启动脚本 | 🟢 |

### 6.4 GraphRAG 工作区 `baselines/graphrag_official/graphrag_workspace/`

| 路径 | 功能 | 状态 |
|---|---|---|
| `.env` | Kimi API Key 配置 | ⚪ 环境配置 |
| `settings.yaml` | GraphRAG v3.x 配置（Kimi LLM + 本地 embedding） | ⚪ |
| `input/` (20 个 .txt) | MultiHop-RAG 前 20 篇文章 | ⚪ 测试数据 |
| `output/` (parquet + lancedb) | GraphRAG 索引输出（communities/entities/relationships 等 6 张表 + stats + context） | ⚪ 自动生成 |
| `cache/` (~800 个文件) | LLM 调用缓存（community_reporting 73 / summarize_descriptions ~500 / text_embedding ~300） | ⚪ 自动生成，体积大 |

### 6.5 评估结果 `baselines/eval_results/`

| 路径 | 功能 | 状态 |
|---|---|---|
| `n200/naive_rag_eval_results.json` | 200 条采样 Naive RAG 结果。MRR=0.6389，P@5=0.2568——**主要对照基线** | 🟢 |
| `naive_rag_indexed/naive_rag_eval_results.json` | 全量索引 Naive RAG 结果（37 条有效查询） | 🟡 小规模参考 |
| `smoke/naive_rag_eval_results.json` | 冒烟测试结果（38 条有效查询） | 🟡 小规模参考 |
| `graphrag_full/graphrag_local_eval_results.json` | GraphRAG Official 全量结果（37 条有效查询），MRR=0.8784 | 🟡 样本量小，与 169 条主实验不完全可比 |
| `graphrag_smoke/graphrag_local_eval_results.json` | GraphRAG 冒烟测试（13 条有效，指标全零） | 🔴 数据无意义 |
| `graphrag_indexed_verify/...json` | 小规模验证（2 条有效） | 🔴 数据无意义 |
| `graphrag_verify/...json` | 小规模验证（3 条有效，指标全零） | 🔴 数据无意义 |

---

## 七、数据与输出

| 路径 | 功能 | 状态 |
|---|---|---|
| `sample_data/sample_paper.txt` | 示例论文文本 | 🟢 开箱即用示例 |
| `sample_data/sample_paper2.json` | 示例论文 JSON | 🟢 |
| `sample_data/sample_paper3.txt` | 示例论文文本 | 🟢 |
| `sample_data/sample_qa.json` | 示例 QA 对 | 🟢 |
| `output_run/communities.csv` | 最近一次 Pipeline 运行输出 | ⚪ 自动生成 |
| `output_run/entities.csv` | 同上 | ⚪ |
| `output_run/relationships.csv` | 同上 | ⚪ |
| `output_run/report.html` | 同上 | ⚪ |
| `output_run/summary.txt` | 同上 | ⚪ |

---

## 八、版本快照 `versions/v5/`

| 路径 | 功能 | 状态 |
|---|---|---|
| `README.md` (368 行) | v5 早期版本的 README | 🔴 比根目录 README (267行) 旧，内容已被取代 |
| `PROJECT_STATUS.md` (359 行) | v5 早期版本的项目状态 | 🔴 缺少根目录版本后续添加的 Dataset A/B 区分说明 |
| `constrained_leiden/*.py` (6 个文件) | v5 核心算法代码的快照 | 🔴 与根目录当前代码逐行一致，纯冗余 |
| `experiments/run_multihop_eval.py` | v5 实验脚本快照 | 🔴 与根目录当前代码一致，冗余 |

**整个 `versions/v5/` 目录都是过时快照**——代码部分与根目录完全重复，文档部分比根目录旧。建议删除或移入 `.gitignore`。

---

## 九、隐藏/系统文件

| 路径 | 功能 | 状态 |
|---|---|---|
| `.pytest_cache/` | Pytest 运行缓存 | ⚪ 自动生成 |
| `.venv-graphrag/` | Python 3.11 虚拟环境 | ⚪ 本地环境 |
| `.DS_Store` (多处) | macOS 系统文件 | ⚪ 应被 .gitignore 排除 |

---

## 十、统计总览

| 分类 | 文件数 | 代码行数 | 状态分布 |
|---|---|---|---|
| 核心算法 (constrained_leiden/) | 6 | 2,271 | 全部 🟢 |
| 数据 Pipeline (data/extraction/retrieval/evaluation/output/) | 5+5空 | 3,123 | 全部 🟢（retriever 有已知 Bug） |
| 预留模块 (proposition/) | 1+1空 | 443 | 🟡 未集成 |
| 根目录入口 | 4 | 732 | 全部 🟢 |
| 文档 | 8 | 2,369 | 3🟢 3🟡 2🔴 |
| 实验脚本 | 14 | 3,723 | 3🟢 1🟡 10🔴 |
| 基线系统 | 13 | 4,089 | 全部 🟢 |
| 版本快照 (versions/) | 9 | 3,061 | 全部 🔴 |
| 实验结果/缓存 | 16 | — | 5🟢 4🟡 3🔴 |

---

## 十一、清理建议

**可安全删除（约节省 3,000+ 行代码 + 大量缓存空间）**：

1. `versions/v5/` 整个目录——根目录已是最新版本
2. `experiments/probe_*.py` 7 个探针脚本——一次性调试工具，已完成使命
3. `baselines/eval_results/` 下的 `graphrag_smoke/`、`graphrag_indexed_verify/`、`graphrag_verify/` 三个目录——样本量极小，数据无统计意义
4. `experiments/results/cache/` 下 `*_full.parquet` 和 `*_n200.parquet`（不带 `_b` 后缀的 4 个文件）——实际实验未使用原始版抽取

**需迁移依赖后再删除**：

5. `experiments/run_experiment.py`——v4 遗留，被 smoke_test.py 和 2 个 probe 脚本 import
6. `experiments/smoke_test.py`——依赖 v4 接口，建议重写为使用 run_multihop_eval.py 接口

**建议保留但标记为过时**：

7. `REFACTOR_PROMPT.md`——代码部分已完成，仅实验验证清单仍有参考价值
8. `experiments/ablation_study_notes.md`——v4 历史实验记录，有溯源价值
