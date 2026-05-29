"""
summarization/summarizer.py
----------------------------
社区摘要生成模块。

为 communities_df 中 level >= min_level 的社区调用 LLM 生成摘要，
结果写回 communities_df["summary"] 列。

设计原则：
  - 只对高层社区（level >= min_level，默认 2）生成摘要，跳过 Level-0/1
    微小社区（平均 3-4 个实体），节省 token 成本
  - 支持 Anthropic 和 OpenAI 两种 provider
  - 批量并发请求（batch_size 控制并发数）
  - 失败时优雅降级：摘要为空，TopDown 自动退化为实体列表
  - 生成结果可缓存到 parquet，避免重复调用

对照实验用法：
  # 无摘要（当前 baseline）
  communities_df["summary"] = ""

  # 有摘要（LLM 接入后）
  communities_df = generate_community_summaries(communities_df, llm_config, text_units_map)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 社区文本构建器（从 entities/text_units 组装 prompt 输入）
# ---------------------------------------------------------------------------

class CommunityTextBuilder:
    """
    将社区的实体列表和原始文本片段组装为 LLM prompt 输入。

    物理纯净社区（低 H）产生的文本输入来自同一段落，语义连贯；
    混杂社区（高 H）产生的文本输入来自多个不相关段落，语义分散。
    这个差异将通过 LLM 摘要质量体现出来。
    """

    def __init__(
        self,
        text_units: List[dict],
        entities_df: Optional[pd.DataFrame] = None,
        max_entities: int = 20,
        max_text_chars: int = 2000,
    ):
        self._chunk_map: Dict[str, str] = {
            u["chunk_id"]: u.get("text", "")
            for u in text_units
            if "chunk_id" in u
        }
        self._entity_titles: Dict[str, str] = {}
        if entities_df is not None and not entities_df.empty and "title" in entities_df.columns:
            id_col = "id" if "id" in entities_df.columns else entities_df.columns[0]
            for _, row in entities_df.iterrows():
                self._entity_titles[str(row[id_col])] = str(row["title"])
        self.max_entities = max_entities
        self.max_text_chars = max_text_chars

    def build_prompt(self, row: pd.Series) -> str:
        """为单个社区构建摘要 prompt。"""
        level = int(row.get("level", 0))
        entropy = float(row.get("structural_entropy", 0.0))
        community_id = row.get("community_id", "?")

        # 实体列表
        entity_ids: List[str] = row.get("entity_ids", []) or []
        entity_names = []
        for eid in entity_ids[:self.max_entities]:
            name = self._entity_titles.get(str(eid), str(eid))
            # node_id 格式 "{sent_id}-{title}"，取最后一段作为名称
            if "-" in name and len(name) > 20:
                name = name.rsplit("-", 1)[-1].replace("_", " ")
            entity_names.append(name)

        # 原文片段（从 text_unit_ids 取段落文本）
        text_unit_ids: List[str] = row.get("text_unit_ids", []) or []
        snippets = []
        total_chars = 0
        seen_chunks = set()
        for tid in text_unit_ids:
            # text_unit_ids 可能是 sent_id，截断为 para_id
            para_id = tid.rsplit("-s", 1)[0] if "-s" in tid else tid
            if para_id in seen_chunks:
                continue
            seen_chunks.add(para_id)
            text = self._chunk_map.get(para_id, "")
            if not text:
                text = self._chunk_map.get(tid, "")
            if text:
                snippets.append(text[:500])
                total_chars += len(text)
                if total_chars >= self.max_text_chars:
                    break

        entities_str = ", ".join(entity_names) if entity_names else "(no entities)"
        texts_str = "\n---\n".join(snippets) if snippets else "(no source text available)"

        prompt = (
            f"You are summarizing a knowledge graph community for a RAG system.\n\n"
            f"Community Level: {level} | Structural Entropy: {entropy:.3f}\n"
            f"Key entities: {entities_str}\n\n"
            f"Source text excerpts:\n{texts_str}\n\n"
            f"Write a concise 2-3 sentence summary of what this community is about. "
            f"Focus on the main topic, key entities, and their relationships. "
            f"Be specific and factual. Do not mention 'community' or 'graph'."
        )
        return prompt


# ---------------------------------------------------------------------------
# LLM 客户端工厂
# ---------------------------------------------------------------------------

def _make_llm_client(llm_config):
    """
    根据 provider 创建 LLM 客户端。

    支持的 provider：
      - "anthropic"          : Anthropic 官方 API
      - "openai"             : OpenAI 官方 API
      - "openai_compatible"  : 任何 OpenAI 兼容接口（Kimi/Moonshot、DeepSeek 等）
                               需同时设置 base_url
    """
    provider = llm_config.provider.lower()
    api_key = llm_config.api_key or ""

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError("需要安装 anthropic：pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key 未设置")
        return anthropic.Anthropic(api_key=key)

    elif provider in ("openai", "openai_compatible"):
        try:
            import openai
        except ImportError:
            raise ImportError("需要安装 openai：pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(f"{provider} API key 未设置")
        kwargs = {"api_key": key}
        # openai_compatible 需要自定义 base_url（如 DeepSeek: https://api.deepseek.com）
        base_url = getattr(llm_config, "base_url", None)
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    else:
        raise ValueError(
            f"不支持的 provider: {provider!r}，"
            f"请使用 'anthropic'、'openai' 或 'openai_compatible'"
        )


def _call_llm(client, provider: str, model: str, prompt: str, max_tokens: int) -> str:
    """调用 LLM 生成单条摘要，失败返回空字符串。"""
    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        elif provider in ("openai", "openai_compatible"):
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"    [摘要] LLM 调用失败（跳过）: {e}")
        return ""

    return ""


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def generate_community_summaries(
    communities_df: pd.DataFrame,
    llm_config,
    text_units: Optional[List[dict]] = None,
    entities_df: Optional[pd.DataFrame] = None,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    为 communities_df 中 level >= llm_config.min_level 的社区生成 LLM 摘要。

    结果写入 communities_df["summary"] 列并返回。
    Level-0/1 的微小社区 summary 保持为空字符串（TopDown 退化为实体列表）。

    Parameters
    ----------
    communities_df : pd.DataFrame
        社区表，来自 run_constrained_community_detection()
    llm_config : LlmConfig
        LLM 配置（provider, model, api_key, max_tokens, batch_size, min_level）
    text_units : List[dict], optional
        段落文本列表，用于构建 prompt（chunk_id → text）
    entities_df : pd.DataFrame, optional
        实体表，用于将 node_id 转换为可读实体名
    cache_path : str, optional
        摘要缓存文件路径（.json）。存在时加载缓存跳过已生成的社区
    verbose : bool
        是否打印进度

    Returns
    -------
    pd.DataFrame
        添加了 "summary" 列的 communities_df（原对象的副本）
    """
    if communities_df.empty:
        return communities_df

    communities_df = communities_df.copy()
    communities_df["summary"] = ""

    # 筛选需要生成摘要的社区（level >= min_level）
    target_mask = communities_df["level"] >= llm_config.min_level
    target_df = communities_df[target_mask]

    if target_df.empty:
        if verbose:
            print(f"  [摘要] 无 level >= {llm_config.min_level} 的社区，跳过摘要生成")
        return communities_df

    if verbose:
        total = len(target_df)
        skipped = len(communities_df) - total
        print(f"  [摘要] 需生成摘要：{total} 个社区（跳过 Level-0/1 共 {skipped} 个）")
        print(f"  [摘要] 模型：{llm_config.provider}/{llm_config.model}")

    # 加载缓存
    cache: Dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if verbose:
            print(f"  [摘要] 已加载缓存：{len(cache)} 条")

    # 构建 prompt builder
    builder = CommunityTextBuilder(
        text_units=text_units or [],
        entities_df=entities_df,
        max_entities=llm_config.max_entities_in_prompt,
    )

    # 创建 LLM 客户端（线程安全：每个线程复用同一个 client）
    client = _make_llm_client(llm_config)
    provider = llm_config.provider.lower()
    max_workers = max(1, llm_config.batch_size)  # batch_size 即并发线程数

    # 分离缓存命中 vs 需要调用的任务
    summaries: Dict[int, str] = {}
    rows = list(target_df.iterrows())
    pending: List[Tuple[int, object, str]] = []  # (df_idx, row, cache_key)

    for df_idx, row in rows:
        cache_key = _community_cache_key(row)
        if cache_key in cache:
            summaries[df_idx] = cache[cache_key]
        else:
            pending.append((df_idx, row, cache_key))

    if verbose:
        print(f"  [摘要] 缓存命中 {len(summaries)} 条，需调用 API {len(pending)} 条")
        print(f"  [摘要] 并发线程数：{max_workers}")

    t0 = time.time()
    done_count = len(summaries)
    total_count = len(rows)
    lock_cache: Dict = {}  # 用于线程安全地累积新结果

    def _call_one(item: Tuple[int, object, str]) -> Tuple[int, str, str]:
        df_idx, row, cache_key = item
        prompt = builder.build_prompt(row)
        summary = _call_llm(client, provider, llm_config.model, prompt, llm_config.max_tokens)
        return df_idx, cache_key, summary

    # 分批并发：每批 max_workers 个，批完后保存缓存
    SAVE_EVERY = max(50, max_workers * 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_one, item): item for item in pending}
        for future in as_completed(futures):
            try:
                df_idx, cache_key, summary = future.result()
            except Exception as e:
                item = futures[future]
                df_idx = item[0]
                cache_key = item[2]
                summary = ""
                print(f"    [摘要] 任务失败（跳过）: {e}")

            summaries[df_idx] = summary
            cache[cache_key] = summary
            done_count += 1

            if verbose and (done_count % SAVE_EVERY == 0 or done_count == total_count):
                elapsed = time.time() - t0
                rate = (done_count - len(summaries) + len(pending)) / elapsed if elapsed > 0 else 0
                api_done = done_count - (total_count - len(pending))
                rate = api_done / elapsed if elapsed > 0 else 0
                print(f"  [摘要] {done_count}/{total_count}  ({rate:.1f} 条/s  已用 {elapsed:.0f}s)")

                # 定期保存缓存，避免中断丢失进度
                if cache_path:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 最终保存缓存
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # 写回 DataFrame
    for df_idx, summary in summaries.items():
        communities_df.at[df_idx, "summary"] = summary

    non_empty = (communities_df["summary"] != "").sum()
    if verbose:
        print(f"  [摘要] 完成：{non_empty} 条有效摘要，耗时 {time.time()-t0:.1f}s")

    return communities_df


def _community_cache_key(row: pd.Series) -> str:
    """基于社区实体列表和层级生成稳定缓存键。"""
    entity_ids = sorted(str(e) for e in (row.get("entity_ids") or []))
    key_str = f"level={row.get('level', 0)}|entities={','.join(entity_ids[:30])}"
    return hashlib.md5(key_str.encode()).hexdigest()
