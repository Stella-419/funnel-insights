# app/core.py
# 纯逻辑层：不依赖 streamlit，可单元测试

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
import random

import pandas as pd


# =========================================================
# Alert / thresholds
# =========================================================
def threshold_pp(window_days: int) -> float:
    if window_days >= 30:
        return 0.15
    if window_days >= 14:
        return 0.20
    return 0.30


def level(delta_pp: float, th: float) -> str:
    if delta_pp <= -th:
        return "🔴 异常下降"
    if delta_pp <= -th / 2:
        return "🟠 轻微下降"
    if delta_pp >= th:
        return "🟢 明显改善"
    return "⚪ 基本稳定"


def emoji_from_level(lv: str) -> str:
    return lv.split()[0] if lv else "⚪"


def build_hint(worst_pp_all: float, th: float) -> str:
    if worst_pp_all <= -th:
        return "优先定位该环节：按渠道/人群/设备/版本拆解；检查近期活动、价格、库存、支付/下单链路是否变更。"
    if worst_pp_all <= -th / 2:
        return "建议做分层对比：拆渠道/新老用户/关键品类/设备，判断是否结构性流量变化或特定人群异常。"
    if worst_pp_all >= th:
        return "建议复盘驱动因素：确认提升是否来自活动/策略/流量结构变化，并沉淀可复用动作。"
    return "建议持续监控：若近期有投放/活动/版本改动，可在后续周期验证影响。"


# =========================================================
# Formatting helpers (UI can reuse)
# =========================================================
def fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def pp(x: float) -> str:
    return f"{x:.2f} pp"


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def safe_rate(num: float, den: float) -> float:
    return (num / den) if den else 0.0


# =========================================================
# Sample data
# =========================================================
def make_sample_data(n_users: int = 400) -> pd.DataFrame:
    base = datetime.now() - timedelta(days=14)
    rows = []
    for i in range(n_users):
        uid = f"user_{i}"
        device = random.choice(["mobile", "desktop", "tablet"])
        country = random.choice(["US", "CN", "JP", "DE"])
        t1 = base + timedelta(minutes=random.randint(0, 60 * 24 * 10))
        rows.append((uid, "page_view", int(t1.timestamp() * 1000), device, country))
        if random.random() < 0.65:
            t2 = t1 + timedelta(minutes=random.randint(1, 120))
            rows.append((uid, "click", int(t2.timestamp() * 1000), device, country))
            if random.random() < 0.45:
                t3 = t2 + timedelta(minutes=random.randint(1, 180))
                rows.append((uid, "purchase", int(t3.timestamp() * 1000), device, country))
    return pd.DataFrame(rows, columns=["user_id", "event", "timestamp", "device", "country"])


# =========================================================
# Breakdown candidate detection
# =========================================================
ENV_KEYWORDS = ["device", "os", "browser", "platform", "app_version", "version", "ua", "user_agent"]
USER_KEYWORDS = [
    "country", "region", "city", "language", "lang", "gender", "age", "member", "vip",
    "segment", "cohort", "new_user", "is_new", "user_type"
]


def classify_dim(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ENV_KEYWORDS):
        return "设备/环境"
    if any(k in c for k in USER_KEYWORDS):
        return "用户属性"
    return "用户属性"


def infer_breakdown_candidates(df: pd.DataFrame, uid_col: str, evt_col: str, ts_col: str) -> List[Tuple[str, str, str]]:
    exclude = {uid_col, evt_col, ts_col}
    candidates: List[Tuple[str, str, str]] = []
    n = len(df)
    if n == 0:
        return candidates

    for col in df.columns:
        if col in exclude:
            continue

        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            continue

        nunique = s.dropna().astype(str).nunique()
        unique_ratio = nunique / max(n, 1)
        if unique_ratio > 0.30:
            continue

        tmp = df[[uid_col, col]].dropna()
        if tmp.empty:
            continue

        per_user_nunique = tmp.groupby(uid_col)[col].nunique()
        multi_rate = (per_user_nunique > 1).mean()
        if multi_rate > 0.15:
            continue

        label = classify_dim(col)
        reason = f"nunique={nunique}, unique_ratio={unique_ratio:.2%}, multi_user_rate={multi_rate:.2%}"
        candidates.append((col, label, reason))

    candidates.sort(key=lambda x: df[x[0]].dropna().astype(str).nunique())
    return candidates


# =========================================================
# SQL builder
# =========================================================
def sql_escape(s: str) -> str:
    return str(s).replace("'", "''")


def funnel_sql_nstep(
    uid: str,
    evt: str,
    ts: str,
    steps: List[str],
    window_days: int,
    strict: bool,
    breakdown_col: Optional[str] = None,
) -> str:
    n = int(window_days)
    steps_esc = [sql_escape(x) for x in steps]
    in_list = ",".join([f"'{x}'" for x in steps_esc])

    bd_select = f", {breakdown_col} AS bd" if breakdown_col else ""
    bd_group = ", bd" if breakdown_col else ""
    bd_cols_select = ", bd" if breakdown_col else ""

    if not strict:
        select_counts = ",\n  ".join([
            f"COUNT(DISTINCT CASE WHEN e='{steps_esc[i]}' THEN u END) AS s{i+1}"
            for i in range(len(steps_esc))
        ])
        return f"""
WITH b AS (SELECT MAX(to_timestamp({ts}/1000)) m FROM events),
c AS (
  SELECT
    {uid} AS u,
    {evt} AS e,
    to_timestamp({ts}/1000) AS t
    {bd_select},
    CASE
      WHEN to_timestamp({ts}/1000) >= (SELECT m FROM b) - INTERVAL {n} DAY THEN 'last'
      WHEN to_timestamp({ts}/1000) >= (SELECT m FROM b) - INTERVAL {2*n} DAY
       AND to_timestamp({ts}/1000) <  (SELECT m FROM b) - INTERVAL {n} DAY THEN 'prev'
    END AS period
  FROM events
  WHERE {evt} IN ({in_list})
)
SELECT
  period
  {bd_cols_select},
  {select_counts}
FROM c
WHERE period IS NOT NULL
GROUP BY period{bd_group}
;
"""
    else:
        t_cols = ",\n    ".join([
            f"MIN(CASE WHEN e='{steps_esc[i]}' THEN t END) AS t{i+1}"
            for i in range(len(steps_esc))
        ])

        filters = []
        for i in range(len(steps_esc)):
            conds = ["t1 IS NOT NULL"]
            for k in range(2, i + 2):
                conds.append(f"t{k} IS NOT NULL")
                conds.append(f"t{k} >= t{k-1}")
            cond = " AND ".join(conds)
            filters.append(f"COUNT(*) FILTER (WHERE {cond}) AS s{i+1}")

        select_counts = ",\n  ".join(filters)

        return f"""
WITH b AS (SELECT MAX(to_timestamp({ts}/1000)) m FROM events),
c AS (
  SELECT
    {uid} AS u,
    {evt} AS e,
    to_timestamp({ts}/1000) AS t
    {bd_select},
    CASE
      WHEN to_timestamp({ts}/1000) >= (SELECT m FROM b) - INTERVAL {n} DAY THEN 'last'
      WHEN to_timestamp({ts}/1000) >= (SELECT m FROM b) - INTERVAL {2*n} DAY
       AND to_timestamp({ts}/1000) <  (SELECT m FROM b) - INTERVAL {n} DAY THEN 'prev'
    END AS period
  FROM events
  WHERE {evt} IN ({in_list})
),
u AS (
  SELECT
    u,
    period
    {bd_cols_select},
    {t_cols}
  FROM c
  WHERE period IS NOT NULL
  GROUP BY u, period{bd_group}
)
SELECT
  period
  {bd_cols_select},
  {select_counts}
FROM u
GROUP BY period{bd_group}
;
"""


# =========================================================
# Metrics
# =========================================================
def compute_rates_and_deltas(prev_row: pd.Series, last_row: pd.Series, step_count: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    rates_last: Dict[str, float] = {}
    deltas: Dict[str, float] = {}

    for i in range(1, step_count):
        a_prev = float(prev_row[f"s{i}"])
        b_prev = float(prev_row[f"s{i+1}"])
        a_last = float(last_row[f"s{i}"])
        b_last = float(last_row[f"s{i+1}"])

        r_prev = safe_rate(b_prev, a_prev)
        r_last = safe_rate(b_last, a_last)

        rates_last[f"r{i}_{i+1}"] = r_last
        deltas[f"d{i}_{i+1}"] = (r_last - r_prev) * 100

    a_prev = float(prev_row["s1"])
    b_prev = float(prev_row[f"s{step_count}"])
    a_last = float(last_row["s1"])
    b_last = float(last_row[f"s{step_count}"])

    r_prev = safe_rate(b_prev, a_prev)
    r_last = safe_rate(b_last, a_last)

    rates_last[f"r1_{step_count}"] = r_last
    deltas[f"d1_{step_count}"] = (r_last - r_prev) * 100

    return rates_last, deltas


def find_worst_delta(deltas_pp: Dict[str, float]) -> Tuple[str, float]:
    worst_k = min(deltas_pp.keys(), key=lambda k: deltas_pp[k])
    return worst_k, float(deltas_pp[worst_k])


def pretty_step_label(dkey: str, steps: List[str]) -> str:
    parts = dkey.replace("d", "").split("_")
    i = int(parts[0])
    j = int(parts[1])
    return f"Step{i}→Step{j}（{steps[i-1]}→{steps[j-1]}）"


# =========================================================
# Export markdown (pure function)
# =========================================================
def build_export_markdown(
    steps: List[str],
    window_days: int,
    strict_mode: bool,
    breakdown_col: Optional[str],
    th: float,
    max_ts: Any,
    worst_readable_all: str,
    worst_pp_all: float,
    risk_level: str,
    hint: str,
    breakdown_summary_text: str,
    report: str,
) -> str:
    return f"""# 事件漏斗洞察日报

- Steps: {steps}
- 周期: last_{window_days}d vs prev_{window_days}d
- 严格漏斗: {strict_mode}
- Breakdown: {breakdown_col or "无"}
- 预警阈值: {th:.2f} pp
- 时间基准: {max_ts}

## 总体预警
- 最大下降步骤：{worst_readable_all}（{worst_pp_all:.2f} pp，{risk_level}）
- 行动提示：{hint}

## 分组洞察（如有）
{breakdown_summary_text or "无"}

## 运营洞察日报
{report or "（尚未生成日报，请先点击“生成/刷新日报”）"}
"""
