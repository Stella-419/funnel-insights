# =========================================================
# Analyzer1.py
# 泛用型 · 事件漏斗自动洞察（含示例数据）- 稳定版
# 关键修复：
# 1) 日报改成按钮触发 + 缓存（解决“卡住”）
# 2) CSV 读取缓存（大文件不重复读）
# 3) prev/last 排序固定（避免算反）
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
# CSS 修复滚动条
# =========================================================
st.markdown("""
<style>
html, body { overflow: auto !important; height: auto !important; }
[data-testid="stAppViewContainer"] { overflow: auto !important; }
[data-testid="stMain"], [data-testid="stSidebar"] {
  width: 100% !important;
  box-sizing: border-box !important;
}
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
# 示例数据生成（方向 B 的关键）
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
# Sidebar
# =========================================================
st.sidebar.title("数据来源")

use_sample = st.sidebar.checkbox("🧪 使用示例数据（无需上传）", value=False)
uploaded = st.sidebar.file_uploader("📂 或上传 CSV（user / event / timestamp）", type=["csv"])

deep_mode = st.sidebar.toggle("深度分析", value=False)
strict_mode = st.sidebar.toggle("严格漏斗", value=False)
window_days = st.sidebar.radio("对比周期（天）", [7, 14, 30], horizontal=True)

# =========================================================
# Main
# =========================================================
st.title("📊 事件漏斗自动洞察")

st.markdown("""
**适合：** 有事件级数据的产品 / 运营 / 分析人员  
**不适合：** 没有埋点或不清楚事件含义的用户
""")

# =========================================================
# Data load
# =========================================================
if use_sample:
    df = make_sample_data()
    st.info("当前使用：示例数据（page_view → click → purchase）")
elif uploaded:
    df = load_csv(uploaded)
else:
    st.stop()

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
    st.error("需要包含 user_id / event / timestamp 列")
    st.stop()

# =========================================================
# Event mapping
# =========================================================
st.sidebar.subheader("漏斗事件映射")
events = sorted(df[evt_col].astype(str).unique().tolist())

s1 = st.sidebar.selectbox("Step 1", events, index=0)
s2 = st.sidebar.selectbox("Step 2", events, index=min(1, len(events)-1))
s3 = st.sidebar.selectbox("Step 3", events, index=min(2, len(events)-1))

# 防止单引号导致 SQL 报错
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

# 固定顺序：prev 在前，last 在后（避免算反）
res["__o"] = res["p"].map({"prev": 0, "last": 1})
res = res.sort_values("__o").drop(columns="__o").reset_index(drop=True)

st.subheader("📈 漏斗对比结果")
st.dataframe(res, use_container_width=True)

# =========================================================
# Auto Insight
# =========================================================
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

d12 = (last_r12 - prev_r12) * 100
d23 = (last_r23 - prev_r23) * 100

worst = min([("Step1→Step2", d12), ("Step2→Step3", d23)], key=lambda x: x[1])

st.subheader("🚨 自动洞察")
st.markdown(
    f"- Step1（{s1}）用户：**{int(last.s1):,}**（Δ {int(last.s1 - prev.s1):,}）\n"
    f"- Step2（{s2}）用户：**{int(last.s2):,}**（Δ {int(last.s2 - prev.s2):,}）\n"
    f"- Step3（{s3}）用户：**{int(last.s3):,}**（Δ {int(last.s3 - prev.s3):,}）"
)
st.markdown(f"**最大下降步骤**：{worst[0]}（{worst[1]:.2f} pp）")

# =========================================================
# LLM Report (按钮触发 + 缓存) —— 解决“卡住”
# =========================================================
model = REASONER_MODEL if deep_mode else CHAT_MODEL

# 报告缓存（相同参数不重复调模型）
if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}

report_key = f"{window_days}|{strict_mode}|{deep_mode}|{s1}|{s2}|{s3}|{int(last.s1)}|{int(last.s2)}|{int(last.s3)}"

st.subheader("🧠 运营洞察日报")

col1, col2 = st.columns([1, 3])
with col1:
    gen = st.button("生成/刷新日报", type="primary", use_container_width=True)
with col2:
    st.caption("切换周期/事件会导致页面重跑；日报建议手动生成，避免频繁调用模型。")

# 点击按钮才调用 LLM
if gen:
    prompt = f"""
这是一个 3 步事件漏斗对比结果：
{res.to_dict(orient="records")}

Step1={s1}, Step2={s2}, Step3={s3}
本期（last）转化：
- Step1→Step2: {last_r12*100:.2f}%
- Step2→Step3: {last_r23*100:.2f}%

变化（pp）：
- Step1→Step2: {d12:.2f}pp
- Step2→Step3: {d23:.2f}pp
最大下降步骤：{worst[0]}（{worst[1]:.2f}pp）

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

# 展示缓存内容（有就展示，没有就提示）
report = st.session_state.report_cache.get(report_key)
if report:
    st.markdown(report)
else:
    st.info("点击上面的「生成/刷新日报」来生成洞察日报。")

# =========================================================
# Export
# =========================================================
st.subheader("📥 导出")
md = f"# 事件漏斗洞察日报\n\n- Step1: {s1}\n- Step2: {s2}\n- Step3: {s3}\n- 周期: {window_days}d\n- 严格漏斗: {strict_mode}\n- 深度分析: {deep_mode}\n\n---\n\n{report or '（尚未生成日报，请先点击“生成/刷新日报”）'}"
fname = f"事件漏斗洞察_{window_days}d.md"
st.download_button("⬇️ 下载 Markdown 日报", md.encode("utf-8"), fname)
