# =========================================================
# Analyzer1.py
# 泛用型 · 事件漏斗自动洞察（含示例数据）- 产品版（仪表盘 + 报告 + 追问 + 导出）
# 修复：
# ✅ 解决“选示例后再上传报错”的 Streamlit session_state bug（改为 radio 数据来源）
# ✅ 删除标题下“适合/不适合”说明文字
# =========================================================

import os
import requests
import pandas as pd
import duckdb
import streamlit as st
from datetime import datetime, timedelta
import random

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="事件漏斗自动洞察", layout="wide")

# =========================================================
# CSS 修复滚动条 + 让看板更“仪表盘”一点
# =========================================================
st.markdown("""
<style>
html, body { overflow: auto !important; height: auto !important; }
[data-testid="stAppViewContainer"] { overflow: auto !important; }
[data-testid="stMain"], [data-testid="stSidebar"] { width: 100% !important; box-sizing: border-box !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; }  /* metric 数字更显眼 */
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
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def run_sql(con, q):
    return con.execute(q).df()

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# =========================================================
# 预警（阈值随周期变化）
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

# =========================================================
# 示例数据生成
# =========================================================
def make_sample_data(n_users=200):
    base = datetime.now() - timedelta(days=14)
    rows = []
    for i in range(n_users):
        uid = f"user_{i}"
        t1 = base + timedelta(minutes=random.randint(0, 60*24*10))
        rows.append((uid, "page_view", int(t1.timestamp()*1000)))
        if random.random() < 0.6:
            t2 = t1 + timedelta(minutes=random.randint(1, 120))
            rows.append((uid, "click", int(t2.timestamp()*1000)))
            if random.random() < 0.4:
                t3 = t2 + timedelta(minutes=random.randint(1, 180))
                rows.append((uid, "purchase", int(t3.timestamp()*1000)))
    return pd.DataFrame(rows, columns=["user_id", "event", "timestamp"])

# =========================================================
# Sidebar（✅ 用 radio 解决“示例+上传冲突”）
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
strict_mode = st.sidebar.toggle("严格漏斗", value=False)
window_days = st.sidebar.radio("对比周期（天）", [7, 14, 30], horizontal=True)

# =========================================================
# Main
# =========================================================
st.title("📊 事件漏斗自动洞察")
# ✅ 删除红框文字：不再显示“适合/不适合”那段说明

# =========================================================
# Data load
# =========================================================
if data_source.startswith("🧪"):
    df = make_sample_data()
    st.info("当前使用：示例数据（page_view → click → purchase）")
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
# Event mapping
# =========================================================
st.sidebar.subheader("漏斗事件映射")
events = sorted(df[evt_col].dropna().astype(str).unique().tolist())
if len(events) < 1:
    st.error("event 列为空，无法进行漏斗分析。")
    st.stop()

def _safe_index(default_idx: int) -> int:
    return min(max(default_idx, 0), max(len(events)-1, 0))

s1 = st.sidebar.selectbox("Step 1", events, index=_safe_index(0))
s2 = st.sidebar.selectbox("Step 2", events, index=_safe_index(1))
s3 = st.sidebar.selectbox("Step 3", events, index=_safe_index(2))

if len({s1, s2, s3}) < 3:
    st.sidebar.warning("建议 Step1/2/3 选择不同事件，否则漏斗意义会变弱。")

# SQL 单引号转义
s1, s2, s3 = [x.replace("'", "''") for x in (s1, s2, s3)]

# =========================================================
# DuckDB
# =========================================================
con = duckdb.connect(":memory:")
con.register("events", df)

max_ts = con.execute(f"SELECT MAX(to_timestamp({ts_col}/1000)) FROM events").fetchone()[0]
st.caption(f"时间基准：{max_ts}（last_{window_days}d vs prev_{window_days}d）")

# =========================================================
# Funnel SQL（宽松/严格）
# =========================================================
def funnel_sql(strict=False):
    n = int(window_days)
    if not strict:
        return f"""
WITH b AS (SELECT MAX(to_timestamp({ts_col}/1000)) m FROM events),
c AS (
  SELECT {uid_col} u, {evt_col} e, to_timestamp({ts_col}/1000) t,
  CASE
    WHEN t >= (SELECT m FROM b) - INTERVAL {n} DAY THEN 'last'
    WHEN t >= (SELECT m FROM b) - INTERVAL {2*n} DAY
     AND t <  (SELECT m FROM b) - INTERVAL {n} DAY THEN 'prev'
  END p
  FROM events
  WHERE e IN ('{s1}','{s2}','{s3}')
)
SELECT p,
  COUNT(DISTINCT CASE WHEN e='{s1}' THEN u END) s1,
  COUNT(DISTINCT CASE WHEN e='{s2}' THEN u END) s2,
  COUNT(DISTINCT CASE WHEN e='{s3}' THEN u END) s3
FROM c
WHERE p IS NOT NULL
GROUP BY 1;
"""
    else:
        return f"""
WITH b AS (SELECT MAX(to_timestamp({ts_col}/1000)) m FROM events),
c AS (
  SELECT {uid_col} u, {evt_col} e, to_timestamp({ts_col}/1000) t,
  CASE
    WHEN t >= (SELECT m FROM b) - INTERVAL {n} DAY THEN 'last'
    WHEN t >= (SELECT m FROM b) - INTERVAL {2*n} DAY
     AND t <  (SELECT m FROM b) - INTERVAL {n} DAY THEN 'prev'
  END p
  FROM events
  WHERE e IN ('{s1}','{s2}','{s3}')
),
u AS (
  SELECT u, p,
    MIN(CASE WHEN e='{s1}' THEN t END) t1,
    MIN(CASE WHEN e='{s2}' THEN t END) t2,
    MIN(CASE WHEN e='{s3}' THEN t END) t3
 показатель
  FROM c
  WHERE p IS NOT NULL
  GROUP BY 1,2
)
SELECT p,
  COUNT(*) FILTER (WHERE t1 IS NOT NULL) s1,
  COUNT(*) FILTER (WHERE t1 IS NOT NULL AND t2 >= t1) s2,
  COUNT(*) FILTER (WHERE t1 IS NOT NULL AND t2 >= t1 AND t3 >= t2) s3
FROM u
GROUP BY 1;
"""

res = run_sql(con, funnel_sql(strict_mode))

# 固定顺序：prev 在前，last 在后
res["__o"] = res["p"].map({"prev": 0, "last": 1})
res = res.sort_values("__o").drop(columns="__o").reset_index(drop=True)

st.subheader("📈 漏斗对比结果")
st.dataframe(res, use_container_width=True)

if res.shape[0] < 2:
    st.warning("没有得到 prev/last 两期数据，可能是数据时间跨度不足或事件过少。")
    st.stop()

prev, last = res.iloc[0], res.iloc[1]

def safe_rate(num, den):
    return (num / den) if den else 0.0

prev_r12 = safe_rate(prev.s2, prev.s1)
last_r12 = safe_rate(last.s2, last.s1)
prev_r23 = safe_rate(prev.s3, prev.s2)
last_r23 = safe_rate(last.s3, last.s2)
prev_r13 = safe_rate(prev.s3, prev.s1)
last_r13 = safe_rate(last.s3, last.s1)

d12 = (last_r12 - prev_r12) * 100
d23 = (last_r23 - prev_r23) * 100
d13 = (last_r13 - prev_r13) * 100

worst = min(
    [("Step1→Step2", d12), ("Step2→Step3", d23), ("Step1→Step3", d13)],
    key=lambda x: x[1]
)

# =========================================================
# ✅ 仪表盘布局
# =========================================================
th = threshold_pp(int(window_days))
levels = {
    f"{s1}→{s2}": level(d12, th),
    f"{s2}→{s3}": level(d23, th),
    f"{s1}→{s3}": level(d13, th),
}

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _pp(x) -> str:
    return f"{x:.2f} pp"

def _pct(x) -> str:
    return f"{x*100:.2f}%"

st.subheader("🚨 自动洞察（本期 vs 上期）")
st.caption(f"周期：last_{window_days}d vs prev_{window_days}d｜预警阈值：{th:.2f} pp")

# ① 用户规模 KPI
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(f"Step1 用户（{s1}）", _fmt_int(last.s1), _fmt_int(last.s1 - prev.s1))
with k2:
    st.metric(f"Step2 用户（{s2}）", _fmt_int(last.s2), _fmt_int(last.s2 - prev.s2))
with k3:
    st.metric(f"Step3 用户（{s3}）", _fmt_int(last.s3), _fmt_int(last.s3 - prev.s3))

st.divider()

# ② 转化率 KPI
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label=f"{s1} → {s2} 转化率",
        value=_pct(last_r12),
        delta=f"{_pp(d12)}  {emoji_from_level(levels[f'{s1}→{s2}'])}"
    )
    st.caption(f"上期 {_pct(prev_r12)} → 本期 {_pct(last_r12)}")

with c2:
    st.metric(
        label=f"{s2} → {s3} 转化率",
        value=_pct(last_r23),
        delta=f"{_pp(d23)}  {emoji_from_level(levels[f'{s2}→{s3}'])}"
    )
    st.caption(f"上期 {_pct(prev_r23)} → 本期 {_pct(last_r23)}")

with c3:
    st.metric(
        label=f"{s1} → {s3} 总转化率",
        value=_pct(last_r13),
        delta=f"{_pp(d13)}  {emoji_from_level(levels[f'{s1}→{s3}'])}"
    )
    st.caption(f"上期 {_pct(prev_r13)} → 本期 {_pct(last_r13)}")

st.divider()

# ③ 预警总览 + 行动提示
worst_step, worst_pp = worst[0], worst[1]
step_map = {
    "Step1→Step2": f"{s1}→{s2}",
    "Step2→Step3": f"{s2}→{s3}",
    "Step1→Step3": f"{s1}→{s3}",
}
worst_readable = step_map.get(worst_step, worst_step)

if worst_pp <= -th:
    risk = "🔴 预警：显著下滑"
    hint = "优先定位该环节：按渠道/人群/品类/设备拆解，检查近期活动、价格、库存、支付/下单链路是否变更。"
elif worst_pp <= -th / 2:
    risk = "🟠 提醒：轻微下滑"
    hint = "建议做分层对比：拆渠道/新老用户/关键品类，判断是否结构性流量变化或特定人群异常。"
elif worst_pp >= th:
    risk = "🟢 改善：明显提升"
    hint = "建议复盘驱动因素：确认提升是否来自活动/策略/流量结构变化，并沉淀可复用动作。"
else:
    risk = "⚪ 稳定：波动正常"
    hint = "建议持续监控：若近期有投放/活动/版本改动，可在后续周期验证影响。"

left, right = st.columns([1, 2])
with left:
    st.markdown("### 🚦 预警总览")
    st.markdown(f"**最大下降步骤**：**{worst_readable}**（{worst_pp:.2f} pp）")
    st.markdown(f"**状态**：{risk}")
with right:
    st.markdown("### 🧭 行动提示")
    st.info(hint)

with st.expander("查看明细（本期/上期/变化/预警）", expanded=False):
    detail = pd.DataFrame([
        {"step": f"{s1}→{s2}", "prev_rate(%)": round(prev_r12*100, 2), "last_rate(%)": round(last_r12*100, 2),
         "delta_pp": round(d12, 2), "alert": levels[f"{s1}→{s2}"]},
        {"step": f"{s2}→{s3}", "prev_rate(%)": round(prev_r23*100, 2), "last_rate(%)": round(last_r23*100, 2),
         "delta_pp": round(d23, 2), "alert": levels[f"{s2}→{s3}"]},
        {"step": f"{s1}→{s3}", "prev_rate(%)": round(prev_r13*100, 2), "last_rate(%)": round(last_r13*100, 2),
         "delta_pp": round(d13, 2), "alert": levels[f"{s1}→{s3}"]},
    ])
    st.dataframe(detail, use_container_width=True)

# =========================================================
# LLM Report（按钮触发 + 缓存）
# =========================================================
model = REASONER_MODEL if deep_mode else CHAT_MODEL
if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}

report_key = f"{window_days}|{strict_mode}|{deep_mode}|{s1}|{s2}|{s3}|{int(last.s1)}|{int(last.s2)}|{int(last.s3)}"

st.subheader("🧠 运营洞察日报")

colA, colB = st.columns([1, 3])
with colA:
    gen_report = st.button("生成/刷新日报", type="primary", use_container_width=True)
with colB:
    st.caption("提示：切换周期/事件会导致页面重跑；日报建议手动生成，避免频繁调用模型。")

if gen_report:
    prompt = f"""
这是一个 3 步事件漏斗对比结果：
{res.to_dict(orient="records")}

Step1={s1}, Step2={s2}, Step3={s3}

本期（last）转化：
- {s1}→{s2}: {last_r12*100:.2f}%
- {s2}→{s3}: {last_r23*100:.2f}%
- {s1}→{s3}: {last_r13*100:.2f}%

变化（pp）：
- {s1}→{s2}: {d12:.2f}pp（{levels[f"{s1}→{s2}"]}）
- {s2}→{s3}: {d23:.2f}pp（{levels[f"{s2}→{s3}"]}）
- {s1}→{s3}: {d13:.2f}pp（{levels[f"{s1}→{s3}"]}）

最大下降步骤：{worst_readable}（{worst_pp:.2f}pp）
状态：{risk}

请生成一份运营洞察日报（Markdown），必须包含：
## 一句话结论
## 变化最大的步骤与影响
## 可能原因（假设）
## 下一步排查建议（按优先级）
要求：不要反问用户；原因必须标注“假设”；建议要可执行。
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
# 可追问 chatbot（折叠区）
# =========================================================
st.subheader("💬 进一步追问（Chatbot）")
with st.expander("打开追问区", expanded=False):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("例如：为什么下降的是某一步？建议怎么拆维度？")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        context = f"""
当前漏斗结果（按用户）：
{res.to_dict(orient="records")}

Step1={s1}, Step2={s2}, Step3={s3}
本期转化：
- {s1}→{s2}: {last_r12*100:.2f}%
- {s2}→{s3}: {last_r23*100:.2f}%
- {s1}→{s3}: {last_r13*100:.2f}%

变化（pp）：
- {s1}→{s2}: {d12:.2f}pp（{levels[f"{s1}→{s2}"]}）
- {s2}→{s3}: {d23:.2f}pp（{levels[f"{s2}→{s3}"]}）
- {s1}→{s3}: {d13:.2f}pp（{levels[f"{s1}→{s3}"]}）

最大下降步骤：{worst_readable}（{worst_pp:.2f}pp）
状态：{risk}

行动提示：
{hint}

已生成日报（可能为空）：
{report or "(尚未生成日报)"}

用户追问：
{q}

请回答：
- 先直接给结论
- 再给 2-3 个可能原因（必须标注假设）
- 给 3 条下一步可执行拆解建议
如果需要新的字段或 SQL 才能继续，请明确写出“需要哪些字段 + 下一步怎么查”。
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

- Step1: {s1}
- Step2: {s2}
- Step3: {s3}
- 周期: {window_days}d
- 严格漏斗: {strict_mode}
- 深度分析: {deep_mode}
- 预警阈值: {th:.2f} pp

## 变化摘要与预警（pp）
- {s1}→{s2}: {d12:.2f} pp  {levels[f"{s1}→{s2}"]}
- {s2}→{s3}: {d23:.2f} pp  {levels[f"{s2}→{s3}"]}
- {s1}→{s3}: {d13:.2f} pp  {levels[f"{s1}→{s3}"]}

## 预警总览
- 最大下降步骤：{worst_readable}（{worst_pp:.2f} pp）
- 状态：{risk}
- 行动提示：{hint}

## 运营洞察日报
{report or "（尚未生成日报，请先点击“生成/刷新日报”）"}
"""
fname = f"事件漏斗洞察_{window_days}d.md"
st.download_button("⬇️ 下载 Markdown 日报", md.encode("utf-8"), fname)
