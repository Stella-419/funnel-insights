# app/Analyzer.py
# UI 入口层：只负责 Streamlit 交互 + 调用 core 逻辑

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import os
import requests
import pandas as pd
import duckdb
import streamlit as st

from app.core import (
    make_sample_data,
    infer_breakdown_candidates,
    classify_dim,
    funnel_sql_nstep,
    compute_rates_and_deltas,
    find_worst_delta,
    pretty_step_label,
    threshold_pp,
    level,
    emoji_from_level,
    build_hint,
    fmt_int,
    pp,
    pct,
    build_export_markdown,
)

# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(page_title="事件漏斗洞察助手", layout="wide")

st.markdown(
    """
<style>
html, body { overflow: auto !important; height: auto !important; }
[data-testid="stAppViewContainer"] { overflow: auto !important; }
[data-testid="stMain"], [data-testid="stSidebar"] { width: 100% !important; box-sizing: border-box !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# DeepSeek config (UI/infra 层)
# =========================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = "deepseek-chat"
REASONER_MODEL = "deepseek-reasoner"


def deepseek_chat(messages, model, temperature=0.3):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def run_sql(con, q: str) -> pd.DataFrame:
    return con.execute(q).df()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("数据来源")
data_source = st.sidebar.radio(
    "选择数据来源",
    ["🧪 使用示例数据（无需上传）", "📂 上传 CSV（user / event / timestamp）"],
    index=0,
    key="data_source",
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

# Load data
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

# Column mapping
cols = {c.lower(): c for c in df.columns}
uid_col = cols.get("user_id") or cols.get("visitorid")
evt_col = cols.get("event")
ts_col = cols.get("timestamp")
if not all([uid_col, evt_col, ts_col]):
    st.error("需要包含 user_id（或 visitorid）/ event / timestamp 列")
    st.stop()

# Funnel steps
st.sidebar.subheader("漏斗步骤（N-step，可增可减）")
events = sorted(df[evt_col].dropna().astype(str).unique().tolist())
if len(events) < 2:
    st.error("事件种类太少（<2），无法进行漏斗分析。")
    st.stop()

if "steps" not in st.session_state:
    default = events[:3] if len(events) >= 3 else events[:2]
    st.session_state["steps"] = default

btn_c1, btn_c2 = st.sidebar.columns(2)
with btn_c1:
    if st.button("➕ 添加一步", use_container_width=True):
        st.session_state["steps"].append(events[0])
with btn_c2:
    if st.button("➖ 删除最后一步", use_container_width=True):
        if len(st.session_state["steps"]) > 2:
            st.session_state["steps"] = st.session_state["steps"][:-1]

new_steps = []
for i, cur in enumerate(st.session_state["steps"]):
    idx = events.index(cur) if cur in events else 0
    val = st.sidebar.selectbox(f"Step {i+1}", events, index=idx, key=f"step_{i}")
    new_steps.append(val)

st.session_state["steps"] = new_steps
steps = st.session_state["steps"]
step_count = len(steps)

if len(set(steps)) < len(steps):
    st.sidebar.warning("建议每一步选择不同事件，否则漏斗解释会变弱。")

# Breakdown selection
st.sidebar.subheader("漏斗维度（Breakdown）")
cands = infer_breakdown_candidates(df, uid_col, evt_col, ts_col)

bd_options = ["不分组（总体）"]
bd_meta = {}
for col, label, reason in cands:
    display = f"{label}｜{col}"
    bd_options.append(display)
    bd_meta[display] = (col, label, reason)

bd_choice = st.sidebar.selectbox("选择分组字段（最多 1 个）", bd_options, index=0)
breakdown_col = None
breakdown_label = None
breakdown_reason = ""

if bd_choice != "不分组（总体）":
    breakdown_col, breakdown_label, breakdown_reason = bd_meta[bd_choice]
    with st.sidebar.expander("为什么推荐这个字段？", expanded=False):
        st.write(f"- 类型：{classify_dim(breakdown_col)}")
        st.write(f"- 统计：{breakdown_reason}")

# DuckDB
con = duckdb.connect(":memory:")
con.register("events", df)
max_ts = con.execute(f"SELECT MAX(to_timestamp({ts_col}/1000)) FROM events").fetchone()[0]
st.caption(f"时间基准：{max_ts}（last_{window_days}d vs prev_{window_days}d）")

# Overall funnel (always)
sql_overall = funnel_sql_nstep(uid_col, evt_col, ts_col, steps, window_days, strict_mode, breakdown_col=None)
res_all = run_sql(con, sql_overall)
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
worst_readable_all = pretty_step_label(worst_k_all, steps)
hint = build_hint(worst_pp_all, th)

# Dashboard
st.subheader("🚨 自动洞察（总体：本期 vs 上期）")
st.caption(f"周期：last_{window_days}d vs prev_{window_days}d｜预警阈值：{th:.2f} pp")

kcols = st.columns(min(3, step_count))
for i in range(min(3, step_count)):
    with kcols[i]:
        sname = steps[i]
        st.metric(
            f"Step{i+1} 用户（{sname}）",
            fmt_int(last_all[f"s{i+1}"]),
            fmt_int(int(last_all[f"s{i+1}"]) - int(prev_all[f"s{i+1}"])),
        )

st.divider()

conv_cols = st.columns(3)
with conv_cols[0]:
    if step_count >= 2:
        d = deltas_all["d1_2"]
        st.metric(
            f"{steps[0]} → {steps[1]} 转化率",
            pct(rates_all["r1_2"]),
            f"{pp(d)}  {emoji_from_level(level(d, th))}",
        )
with conv_cols[1]:
    if step_count >= 3:
        d = deltas_all["d2_3"]
        st.metric(
            f"{steps[1]} → {steps[2]} 转化率",
            pct(rates_all["r2_3"]),
            f"{pp(d)}  {emoji_from_level(level(d, th))}",
        )
with conv_cols[2]:
    d = deltas_all[f"d1_{step_count}"]
    st.metric(
        f"{steps[0]} → {steps[-1]} 总转化率",
        pct(rates_all[f"r1_{step_count}"]),
        f"{pp(d)}  {emoji_from_level(level(d, th))}",
    )

st.divider()
left, right = st.columns([1, 2])
with left:
    st.markdown("### 🚦 预警总览（总体）")
    st.markdown(f"**最大下降步骤**：**{worst_readable_all}**（{worst_pp_all:.2f} pp）")
    st.markdown(f"**状态**：{risk_level}")
with right:
    st.markdown("### 🧭 行动提示（总体）")
    st.info(hint)

# Breakdown funnel (optional)
breakdown_summary_text = ""
if breakdown_col:
    st.subheader(f"🧩 分组漏斗（Breakdown：{breakdown_label}｜{breakdown_col}）")

    sql_bd = funnel_sql_nstep(uid_col, evt_col, ts_col, steps, window_days, strict_mode, breakdown_col=breakdown_col)
    res_bd = run_sql(con, sql_bd)
    res_bd["__o"] = res_bd["period"].map({"prev": 0, "last": 1})
    res_bd = res_bd.sort_values(["bd", "__o"]).drop(columns="__o").reset_index(drop=True)
    res_bd["bd"] = res_bd["bd"].astype(str).fillna("(null)")

    groups = []
    for g, gdf in res_bd.groupby("bd"):
        if gdf.shape[0] < 2:
            continue
        p = gdf.iloc[0]
        l = gdf.iloc[1]
        rates_g, deltas_g = compute_rates_and_deltas(p, l, step_count)
        wk, wpp = find_worst_delta(deltas_g)
        groups.append(
            {
                "group": g,
                "worst_step": pretty_step_label(wk, steps),
                "worst_pp": round(wpp, 2),
                "status": level(wpp, th),
                **{f"last_s{i+1}": int(l[f"s{i+1}"]) for i in range(step_count)},
                **{f"last_r{i}_{i+1}": round(rates_g[f"r{i}_{i+1}"] * 100, 2) for i in range(1, step_count)},
            }
        )

    if not groups:
        st.info("该维度在当前数据下无法形成完整的 prev/last 两期分组对比。")
    else:
        bd_table = pd.DataFrame(groups).sort_values("worst_pp").reset_index(drop=True)
        top = bd_table.iloc[0]
        breakdown_summary_text = f"下降最严重的分组是 **{top['group']}**，发生在 **{top['worst_step']}**（{top['worst_pp']:.2f} pp，{top['status']}）。"
        st.markdown(f"🚨 {breakdown_summary_text}")

        show_cols = ["group", "status", "worst_pp", "worst_step"]
        for i in range(min(step_count, 4)):
            show_cols.append(f"last_s{i+1}")
        for i in range(1, min(step_count, 4)):
            show_cols.append(f"last_r{i}_{i+1}")

        bd_show = bd_table[show_cols].copy()
        rename_map = {"group": "分组", "status": "预警", "worst_pp": "最大下滑(pp)", "worst_step": "下滑步骤"}
        for i in range(min(step_count, 4)):
            rename_map[f"last_s{i+1}"] = f"本期 Step{i+1} 用户"
        for i in range(1, min(step_count, 4)):
            rename_map[f"last_r{i}_{i+1}"] = f"本期 Step{i}→{i+1} 转化(%)"
        bd_show = bd_show.rename(columns=rename_map)

        top_group_value = str(top["group"])

        def highlight_top(row):
            if str(row["分组"]) == top_group_value:
                return ["background-color: rgba(255, 0, 0, 0.08)"] * len(row)
            return [""] * len(row)

        st.dataframe(bd_show.style.apply(highlight_top, axis=1), use_container_width=True)

        with st.expander("查看分组明细原始表（含更多步骤/指标）", expanded=False):
            st.dataframe(bd_table, use_container_width=True)

# LLM Report
model = REASONER_MODEL if deep_mode else CHAT_MODEL
if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}

report_key = f"{window_days}|{strict_mode}|{deep_mode}|{','.join(steps)}|bd={breakdown_col}|all={int(last_all['s1'])}"

st.subheader("🧠 运营洞察日报")
colA, colB = st.columns([1, 3])
with colA:
    gen_report = st.button("生成/刷新日报", type="primary", use_container_width=True)
with colB:
    st.caption("提示：切换周期/漏斗步骤/维度会导致页面重跑；日报建议手动生成，避免频繁调用模型。")

if gen_report:
    deltas_list = []
    for i in range(1, step_count):
        dk = f"d{i}_{i+1}"
        deltas_list.append(f"- Step{i}→{i+1}：{deltas_all[dk]:.2f}pp")
    deltas_list.append(f"- Step1→{step_count}：{deltas_all[f'd1_{step_count}']:.2f}pp")

    prompt = f"""
你是互联网产品运营分析助手。下面是一个事件漏斗对比（按用户），请输出一份“运营洞察日报”（Markdown），不要反问用户。

【漏斗定义】
Steps = {steps}
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

report = st.session_state.report_cache.get(report_key, "")
if report:
    st.markdown(report)
else:
    st.info("点击上面的「生成/刷新日报」来生成洞察日报。")

# Export
st.subheader("📥 导出日报")
md = build_export_markdown(
    steps=steps,
    window_days=window_days,
    strict_mode=strict_mode,
    breakdown_col=breakdown_col,
    th=th,
    max_ts=max_ts,
    worst_readable_all=worst_readable_all,
    worst_pp_all=worst_pp_all,
    risk_level=risk_level,
    hint=hint,
    breakdown_summary_text=breakdown_summary_text,
    report=report,
)
fname = f"事件漏斗洞察_{window_days}d.md"
st.download_button("⬇️ 下载 Markdown 日报", md.encode("utf-8"), fname)
