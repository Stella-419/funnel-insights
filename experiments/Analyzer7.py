# =========================================================
# Analyzer1.py
# 泛用型 · 事件漏斗自动洞察（含示例数据）- 产品版
# ✅ N-step 漏斗（可增可减）
# ✅ 1 个 Breakdown 维度（自动筛选：用户属性/设备环境）
# ✅ Breakdown：完整分组表 + 自动点名 Top 下滑分组 + 高亮
# ✅ 仪表盘（总体）+ 报告 + 追问 + 导出
# ✅ 修复：示例/上传冲突的 session_state 报错（radio 数据来源）
# =========================================================

import os
import requests
import pandas as pd
import duckdb
import streamlit as st
from datetime import datetime, timedelta
import random
from typing import List, Tuple, Dict, Optional

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="事件漏斗洞察助手", layout="wide")

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
html, body { overflow: auto !important; height: auto !important; }
[data-testid="stAppViewContainer"] { overflow: auto !important; }
[data-testid="stMain"], [data-testid="stSidebar"] { width: 100% !important; box-sizing: border-box !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DeepSeek Config
# =========================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = "deepseek-chat"
REASONER_MODEL = "deepseek-reasoner"

# =========================================================
# Utils
# =========================================================
def deepseek_chat(messages, model, temperature=0.3):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def run_sql(con, q):
    return con.execute(q).df()

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def sql_escape(s: str) -> str:
    return str(s).replace("'", "''")

# =========================================================
# Alert
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

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _pp(x) -> str:
    return f"{x:.2f} pp"

def _pct(x) -> str:
    return f"{x*100:.2f}%"

def safe_rate(num, den):
    return (num / den) if den else 0.0

# =========================================================
# Sample data
# =========================================================
def make_sample_data(n_users=400):
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
# Breakdown candidate detection (user attr + device/env)
# =========================================================
ENV_KEYWORDS = ["device", "os", "browser", "platform", "app_version", "version", "ua", "user_agent"]
USER_KEYWORDS = ["country", "region", "city", "language", "lang", "gender", "age", "member", "vip", "segment", "cohort", "new_user", "is_new", "user_type"]

def classify_dim(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ENV_KEYWORDS):
        return "设备/环境"
    if any(k in c for k in USER_KEYWORDS):
        return "用户属性"
    # 默认也当“用户属性”，但会通过统计过滤把不合适的踢掉
    return "用户属性"

def infer_breakdown_candidates(df: pd.DataFrame, uid_col: str, evt_col: str, ts_col: str) -> List[Tuple[str, str, str]]:
    """
    返回：[(col_name, label, reason)]
    label: "用户属性" or "设备/环境"
    """
    exclude = {uid_col, evt_col, ts_col}
    candidates = []
    n = len(df)
    if n == 0:
        return candidates

    for col in df.columns:
        if col in exclude:
            continue

        s = df[col]
        # 1) 排除数值型（常见的 price/count 等）
        if pd.api.types.is_numeric_dtype(s):
            continue

        # 2) 唯一值比例太高（疑似 item_id / session_id）
        nunique = s.dropna().astype(str).nunique()
        unique_ratio = nunique / max(n, 1)
        if unique_ratio > 0.30:
            continue

        # 3) 单用户多值比例（强力过滤：不是用户/环境属性的列会被踢）
        tmp = df[[uid_col, col]].dropna()
        if tmp.empty:
            continue
        per_user_nunique = tmp.groupby(uid_col)[col].nunique()
        multi_rate = (per_user_nunique > 1).mean()  # 有多个不同值的用户占比
        if multi_rate > 0.15:
            # 多对多明显：先不支持
            continue

        label = classify_dim(col)
        reason = f"nunique={nunique}, unique_ratio={unique_ratio:.2%}, multi_user_rate={multi_rate:.2%}"
        candidates.append((col, label, reason))

    # 让更“像维度”的排在前面：nunique 越小越靠前
    candidates.sort(key=lambda x: df[x[0]].dropna().astype(str).nunique())
    return candidates

# =========================================================
# N-step funnel SQL (loose/strict), optional breakdown
# =========================================================
def funnel_sql_nstep(
    uid: str,
    evt: str,
    ts: str,
    steps: List[str],
    window_days: int,
    strict: bool,
    breakdown_col: Optional[str] = None,
) -> str:
    """
    Output columns:
      period ('prev'/'last'), [breakdown_col], s1..sN
    """
    n = int(window_days)
    steps_esc = [sql_escape(x) for x in steps]
    in_list = ",".join([f"'{x}'" for x in steps_esc])

    bd_select = f", {breakdown_col} AS bd" if breakdown_col else ""
    bd_group = ", bd" if breakdown_col else ""
    bd_cols_select = ", bd" if breakdown_col else ""

    if not strict:
        # Loose: per period(+bd) count distinct users per event
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
        # Strict: per user(+period+bd) earliest time per step, then count sequentially
        t_cols = ",\n    ".join([
            f"MIN(CASE WHEN e='{steps_esc[i]}' THEN t END) AS t{i+1}"
            for i in range(len(steps_esc))
        ])

        # build FILTER conditions for each step count
        # s1: t1 is not null
        # s2: t1 not null AND t2 not null AND t2>=t1
        # s3: ... AND t3>=t2
        filters = []
        for i in range(len(steps_esc)):
            conds = [f"t1 IS NOT NULL"]
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
# Compute rates + pp changes for a single (prev,last) pair
# =========================================================
def compute_rates_and_deltas(prev_row: pd.Series, last_row: pd.Series, step_count: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    returns:
      rates: {"r1_2":..., "r2_3":..., ...}
      deltas_pp: {"d1_2":..., ...} (pp)
    """
    rates_prev = {}
    rates_last = {}
    deltas = {}
    for i in range(1, step_count):
        a_prev = float(prev_row[f"s{i}"])
        b_prev = float(prev_row[f"s{i+1}"])
        a_last = float(last_row[f"s{i}"])
        b_last = float(last_row[f"s{i+1}"])
        r_prev = safe_rate(b_prev, a_prev)
        r_last = safe_rate(b_last, a_last)
        rates_prev[f"r{i}_{i+1}"] = r_prev
        rates_last[f"r{i}_{i+1}"] = r_last
        deltas[f"d{i}_{i+1}"] = (r_last - r_prev) * 100
    # overall first->last
    a_prev = float(prev_row["s1"]); b_prev = float(prev_row[f"s{step_count}"])
    a_last = float(last_row["s1"]); b_last = float(last_row[f"s{step_count}"])
    r_prev = safe_rate(b_prev, a_prev)
    r_last = safe_rate(b_last, a_last)
    rates_prev[f"r1_{step_count}"] = r_prev
    rates_last[f"r1_{step_count}"] = r_last
    deltas[f"d1_{step_count}"] = (r_last - r_prev) * 100

    # merge rates: last rates only (you通常展示本期)
    rates = {k: rates_last[k] for k in rates_last}
    return rates, deltas

def find_worst_delta(deltas_pp: Dict[str, float]) -> Tuple[str, float]:
    # return key like "d2_3" with most negative pp
    worst_k = min(deltas_pp.keys(), key=lambda k: deltas_pp[k])
    return worst_k, float(deltas_pp[worst_k])

def pretty_step_label(dkey: str, steps: List[str]) -> str:
    # dkey like d2_3 -> Step2→Step3 and also event name
    parts = dkey.replace("d", "").split("_")
    i = int(parts[0]); j = int(parts[1])
    return f"Step{i}→Step{j}（{steps[i-1]}→{steps[j-1]}）"

# =========================================================
# Sidebar: data source + funnel config
# =========================================================
st.sidebar.title("数据来源")
data_source = st.sidebar.radio(
    "选择数据来源",
    ["🧪 使用示例数据（无需上传）", "📂 上传 CSV（user / event / timestamp）"],
    index=0,
    key="data_source"
)
uploaded = None
if data_source.startswith("📂"):
    uploaded = st.sidebar.file_uploader("上传 CSV 文件", type=["csv"], key="uploaded_csv")

deep_mode = st.sidebar.toggle("深度分析", value=False)
strict_mode = st.sidebar.toggle("严格漏斗（按顺序）", value=False)
window_days = st.sidebar.radio("对比周期（天）", [7, 14, 30], horizontal=True)

# =========================================================
# Main
# =========================================================
st.title("📊 事件漏斗自动洞察")

# =========================================================
# Load data
# =========================================================
if data_source.startswith("🧪"):
    df = make_sample_data()
    st.info("当前使用：示例数据（page_view → click → purchase），并附带 device/country 作为维度示例。")
else:
    if not uploaded:
        st.info("请在左侧上传 CSV 文件")
        st.stop()
    df = load_csv(uploaded)

with st.expander("数据预览", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# Column mapping
# =========================================================
cols = {c.lower(): c for c in df.columns}
uid_col = cols.get("user_id") or cols.get("visitorid")
evt_col = cols.get("event")
ts_col = cols.get("timestamp")

if not all([uid_col, evt_col, ts_col]):
    st.error("需要包含 user_id（或 visitorid）/ event / timestamp 列")
    st.stop()

# =========================================================
# Funnel steps (N-step)
# =========================================================
st.sidebar.subheader("漏斗步骤（N-step，可增可减）")
events = sorted(df[evt_col].dropna().astype(str).unique().tolist())
if len(events) < 2:
    st.error("事件种类太少（<2），无法进行漏斗分析。")
    st.stop()

# init steps
if "steps" not in st.session_state:
    # 尽量用前三个事件做默认
    default = events[:3] if len(events) >= 3 else events[:2]
    st.session_state["steps"] = default

# step add/remove controls
btn_c1, btn_c2 = st.sidebar.columns(2)
with btn_c1:
    if st.button("➕ 添加一步", use_container_width=True):
        # 新增一步默认选最后一个事件（或第一个）
        st.session_state["steps"].append(events[min(len(events)-1, 0)])
with btn_c2:
    if st.button("➖ 删除最后一步", use_container_width=True):
        if len(st.session_state["steps"]) > 2:
            st.session_state["steps"] = st.session_state["steps"][:-1]

# render step selectors
new_steps = []
for i, cur in enumerate(st.session_state["steps"]):
    idx = events.index(cur) if cur in events else 0
    val = st.sidebar.selectbox(f"Step {i+1}", events, index=idx, key=f"step_{i}")
    new_steps.append(val)

# persist
st.session_state["steps"] = new_steps
steps = [sql_escape(x) for x in st.session_state["steps"]]
step_count = len(steps)

if len(set(steps)) < len(steps):
    st.sidebar.warning("建议每一步选择不同事件，否则漏斗解释会变弱。")

# =========================================================
# Breakdown selection (1 dim, auto candidates)
# =========================================================
st.sidebar.subheader("漏斗维度（Breakdown）")
cands = infer_breakdown_candidates(df, uid_col, evt_col, ts_col)

# Build options: show label + col name
bd_options = ["不分组（总体）"]
bd_meta = {}  # display -> actual col
for col, label, reason in cands:
    display = f"{label}｜{col}"
    bd_options.append(display)
    bd_meta[display] = col

bd_choice = st.sidebar.selectbox("选择分组字段（最多 1 个）", bd_options, index=0)
breakdown_col = None
breakdown_label = None
if bd_choice != "不分组（总体）":
    breakdown_col = bd_meta[bd_choice]
    breakdown_label = classify_dim(breakdown_col)
    with st.sidebar.expander("为什么推荐这个字段？", expanded=False):
        # 找到原因
        reason = next((r for c, l, r in cands if c == breakdown_col), "")
        st.write(f"- 类型：{breakdown_label}")
        st.write(f"- 统计：{reason}")
        st.caption("说明：我们只推荐更像“用户属性/设备环境”的字段，避免 item_id 等多对多维度造成漏斗不可解释。")

# =========================================================
# DuckDB
# =========================================================
con = duckdb.connect(":memory:")
con.register("events", df)

max_ts = con.execute(f"SELECT MAX(to_timestamp({ts_col}/1000)) FROM events").fetchone()[0]
st.caption(f"时间基准：{max_ts}（last_{window_days}d vs prev_{window_days}d）")

# =========================================================
# Overall funnel (always)
# =========================================================
sql_overall = funnel_sql_nstep(uid_col, evt_col, ts_col, steps, window_days, strict_mode, breakdown_col=None)
res_all = run_sql(con, sql_overall)

# normalize period order
res_all["__o"] = res_all["period"].map({"prev": 0, "last": 1})
res_all = res_all.sort_values("__o").drop(columns="__o").reset_index(drop=True)

st.subheader("📈 漏斗对比结果（总体）")
st.dataframe(res_all, use_container_width=True)

if res_all.shape[0] < 2:
    st.warning("总体没有得到 prev/last 两期数据，可能是数据时间跨度不足或事件过少。")
    st.stop()

prev_all, last_all = res_all.iloc[0], res_all.iloc[1]
rates_all, deltas_all = compute_rates_and_deltas(prev_all, last_all, step_count)
worst_k_all, worst_pp_all = find_worst_delta(deltas_all)

th = threshold_pp(int(window_days))
risk_level = level(worst_pp_all, th)

# =========================================================
# Dashboard (overall)
# =========================================================
st.subheader("🚨 自动洞察（总体：本期 vs 上期）")
st.caption(f"周期：last_{window_days}d vs prev_{window_days}d｜预警阈值：{th:.2f} pp")

# KPI users
kcols = st.columns(min(3, step_count))
# 只展示前三个步骤用户数（避免步骤太多挤爆）
for i in range(min(3, step_count)):
    with kcols[i]:
        sname = st.session_state["steps"][i]
        st.metric(
            f"Step{i+1} 用户（{sname}）",
            _fmt_int(last_all[f"s{i+1}"]),
            _fmt_int(int(last_all[f"s{i+1}"]) - int(prev_all[f"s{i+1}"]))
        )

st.divider()

# KPI conversions: 只展示前三个转化：1->2, 2->3, 1->N
conv_cols = st.columns(3)
with conv_cols[0]:
    if step_count >= 2:
        d = deltas_all["d1_2"]
        st.metric(
            f"{st.session_state['steps'][0]} → {st.session_state['steps'][1]} 转化率",
            _pct(rates_all["r1_2"]),
            f"{_pp(d)}  {emoji_from_level(level(d, th))}",
        )
with conv_cols[1]:
    if step_count >= 3:
        d = deltas_all["d2_3"]
        st.metric(
            f"{st.session_state['steps'][1]} → {st.session_state['steps'][2]} 转化率",
            _pct(rates_all["r2_3"]),
            f"{_pp(d)}  {emoji_from_level(level(d, th))}",
        )
with conv_cols[2]:
    d = deltas_all[f"d1_{step_count}"]
    st.metric(
        f"{st.session_state['steps'][0]} → {st.session_state['steps'][-1]} 总转化率",
        _pct(rates_all[f"r1_{step_count}"]),
        f"{_pp(d)}  {emoji_from_level(level(d, th))}",
    )

# Risk summary + action hint
st.divider()
left, right = st.columns([1, 2])

worst_readable_all = pretty_step_label(worst_k_all, st.session_state["steps"])
if worst_pp_all <= -th:
    hint = "优先定位该环节：按渠道/人群/设备/版本拆解；检查近期活动、价格、库存、支付/下单链路是否变更。"
elif worst_pp_all <= -th / 2:
    hint = "建议做分层对比：拆渠道/新老用户/关键品类/设备，判断是否结构性流量变化或特定人群异常。"
elif worst_pp_all >= th:
    hint = "建议复盘驱动因素：确认提升是否来自活动/策略/流量结构变化，并沉淀可复用动作。"
else:
    hint = "建议持续监控：若近期有投放/活动/版本改动，可在后续周期验证影响。"

with left:
    st.markdown("### 🚦 预警总览（总体）")
    st.markdown(f"**最大下降步骤**：**{worst_readable_all}**（{worst_pp_all:.2f} pp）")
    st.markdown(f"**状态**：{risk_level}")
with right:
    st.markdown("### 🧭 行动提示（总体）")
    st.info(hint)

# =========================================================
# Breakdown funnel (optional)
# =========================================================
breakdown_summary_text = ""
top_group_info = None  # (group_value, worst_step_label, worst_pp)

if breakdown_col:
    st.subheader(f"🧩 分组漏斗（Breakdown：{breakdown_label}｜{breakdown_col}）")

    sql_bd = funnel_sql_nstep(uid_col, evt_col, ts_col, steps, window_days, strict_mode, breakdown_col=breakdown_col)
    res_bd = run_sql(con, sql_bd)

    # normalize period order
    res_bd["__o"] = res_bd["period"].map({"prev": 0, "last": 1})
    res_bd = res_bd.sort_values(["bd", "__o"]).drop(columns="__o").reset_index(drop=True)

    # 如果 bd 为空/缺失会出现 NaN，这里统一填充
    res_bd["bd"] = res_bd["bd"].astype(str).fillna("(null)")

    # 生成 “完整分组表”：每个 bd 一行，展示 last 期的 step users + 最差 pp
    groups = []
    for g, gdf in res_bd.groupby("bd"):
        if gdf.shape[0] < 2:
            continue
        p = gdf.iloc[0]
        l = gdf.iloc[1]
        rates_g, deltas_g = compute_rates_and_deltas(p, l, step_count)
        wk, wpp = find_worst_delta(deltas_g)
        groups.append({
            "group": g,
            "worst_step": pretty_step_label(wk, st.session_state["steps"]),
            "worst_pp": round(wpp, 2),
            "status": level(wpp, th),
            **{f"last_s{i+1}": int(l[f"s{i+1}"]) for i in range(step_count)},
            **{f"last_r{i}_{i+1}": round(rates_g[f"r{i}_{i+1}"] * 100, 2) for i in range(1, step_count)},
        })

    if not groups:
        st.info("该维度在当前数据下无法形成完整的 prev/last 两期分组对比。")
    else:
        bd_table = pd.DataFrame(groups)
        bd_table = bd_table.sort_values("worst_pp").reset_index(drop=True)

        # 自动点名最差分组
        top = bd_table.iloc[0]
        top_group_info = (top["group"], top["worst_step"], float(top["worst_pp"]))
        breakdown_summary_text = f"下降最严重的分组是 **{top['group']}**，发生在 **{top['worst_step']}**（{top['worst_pp']:.2f} pp，{top['status']}）。"

        st.markdown(f"🚨 {breakdown_summary_text}")

        # 为展示友好，挑选展示列
        show_cols = ["group", "status", "worst_pp", "worst_step"]
        # 展示本期用户数（最多前 4 步，避免太宽）
        for i in range(min(step_count, 4)):
            show_cols.append(f"last_s{i+1}")
        # 展示关键转化率（最多前 3 个转化）
        for i in range(1, min(step_count, 4)):
            show_cols.append(f"last_r{i}_{i+1}")

        bd_show = bd_table[show_cols].copy()
        # rename
        rename_map = {"group": "分组", "status": "预警", "worst_pp": "最大下滑(pp)", "worst_step": "下滑步骤"}
        for i in range(min(step_count, 4)):
            rename_map[f"last_s{i+1}"] = f"本期 Step{i+1} 用户"
        for i in range(1, min(step_count, 4)):
            rename_map[f"last_r{i}_{i+1}"] = f"本期 Step{i}→{i+1} 转化(%)"
        bd_show = bd_show.rename(columns=rename_map)

        # highlighter
        top_group_value = str(top["group"])
        def highlight_top(row):
            # 高亮 Top 下滑分组
            if str(row["分组"]) == top_group_value:
                return ["background-color: rgba(255, 0, 0, 0.08)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            bd_show.style.apply(highlight_top, axis=1),
            use_container_width=True
        )

        with st.expander("查看分组明细原始表（含更多步骤/指标）", expanded=False):
            st.dataframe(bd_table, use_container_width=True)

# =========================================================
# LLM Report (button + cache)
# =========================================================
model = REASONER_MODEL if deep_mode else CHAT_MODEL
if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}

# key should include steps + breakdown selection
report_key = f"{window_days}|{strict_mode}|{deep_mode}|{','.join(st.session_state['steps'])}|bd={breakdown_col}|all={int(last_all['s1'])}"

st.subheader("🧠 运营洞察日报")
colA, colB = st.columns([1, 3])
with colA:
    gen_report = st.button("生成/刷新日报", type="primary", use_container_width=True)
with colB:
    st.caption("提示：切换周期/漏斗步骤/维度会导致页面重跑；日报建议手动生成，避免频繁调用模型。")

if gen_report:
    # build a compact summary for prompt
    # overall deltas list
    deltas_list = []
    for i in range(1, step_count):
        dk = f"d{i}_{i+1}"
        deltas_list.append(f"- Step{i}→{i+1}：{deltas_all[dk]:.2f}pp")
    deltas_list.append(f"- Step1→{step_count}：{deltas_all[f'd1_{step_count}']:.2f}pp")

    prompt = f"""
你是互联网产品运营分析助手。下面是一个事件漏斗对比（按用户），请输出一份“运营洞察日报”（Markdown），不要反问用户。

【漏斗定义】
Steps = {st.session_state['steps']}
严格漏斗（顺序）={strict_mode}
周期：last_{window_days}d vs prev_{window_days}d
时间基准：{max_ts}

【总体（不分组）统计】
prev/last counts：
{res_all.to_dict(orient="records")}

总体转化变化（pp）：
{chr(10).join(deltas_list)}
最大下降步骤：{worst_readable_all}（{worst_pp_all:.2f}pp，{risk_level}）

【分组洞察（如有）】
Breakdown字段：{breakdown_col or "无"}
{breakdown_summary_text or "无分组或无有效分组对比"}

【输出要求】
必须包含：
## 一句话结论
## 变化最大的步骤与影响（先总体，再分组）
## 可能原因（假设，2-4条）
## 下一步排查与运营动作（按优先级，至少5条，可执行）
"""
    try:
        with st.spinner("生成日报中…（深度分析会更慢）"):
            report = deepseek_chat([{"role": "user", "content": prompt}], model=model)
        st.session_state.report_cache[report_key] = report
    except Exception as e:
        st.error("日报生成失败（可能是网络/限流/Key/超时）。")
        st.code(str(e))

report = st.session_state.report_cache.get(report_key)
if report:
    st.markdown(report)
else:
    st.info("点击上面的「生成/刷新日报」来生成洞察日报。")

# =========================================================
# Chat follow-up
# =========================================================
st.subheader("💬 进一步追问（Chatbot）")
with st.expander("打开追问区", expanded=False):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("例如：哪个分组最值得优先排查？我应该怎么拆？")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        context = f"""
【漏斗定义】
Steps={st.session_state['steps']}, strict={strict_mode}, window_days={window_days}

【总体结果】
{res_all.to_dict(orient="records")}
最大下降步骤：{worst_readable_all}（{worst_pp_all:.2f}pp，{risk_level}）
行动提示：{hint}

【分组结果（如有）】
Breakdown字段：{breakdown_col or "无"}
{breakdown_summary_text or "无分组或无有效分组对比"}

【用户追问】
{q}

请回答：
- 先给结论（1-2句）
- 再给 2-3 个可能原因（必须标注“假设”）
- 给 5 条下一步可执行的拆解/排查建议（能落地）
- 若需要额外字段或 SQL 才能确认，请明确写出“需要哪些字段 + 怎么查”
"""
        try:
            with st.chat_message("assistant"):
                with st.spinner("生成回答中…"):
                    ans = deepseek_chat([{"role": "user", "content": context}], model=CHAT_MODEL, temperature=0.3)
                st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error("追问失败（可能是网络/限流/Key/超时）。")
            st.code(str(e))

# =========================================================
# Export
# =========================================================
st.subheader("📥 导出日报")
md = f"""# 事件漏斗洞察日报

- Steps: {st.session_state['steps']}
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
fname = f"事件漏斗洞察_{window_days}d.md"
st.download_button("⬇️ 下载 Markdown 日报", md.encode("utf-8"), fname)
