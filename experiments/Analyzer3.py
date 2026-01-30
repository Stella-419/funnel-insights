# =========================
# Analyzer1.py  (Auto Insights + Multi-window + Export)
# =========================

import os
import re
import json
import requests
import pandas as pd
import duckdb
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="运营漏斗洞察（RetailRocket）",
    layout="wide"
)

# =========================
# 🔧 强力兜底：修复滚动条问题
# =========================
st.markdown("""
<style>
html, body {
  overflow: auto !important;
  height: auto !important;
}
[data-testid="stAppViewContainer"] {
  overflow: auto !important;
}
/* 防止 fixed + 100vw 覆盖滚动条 */
[data-testid="stChatInput"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stMain"],
[data-testid="stSidebar"] {
  width: 100% !important;
  box-sizing: border-box !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DeepSeek Config
# =========================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = "deepseek-chat"
REASONER_MODEL = "deepseek-reasoner"

# =========================
# Utils
# =========================
def deepseek_chat(messages, model, json_mode=False, temperature=0.3):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Please set environment variable DEEPSEEK_API_KEY.")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_sql(con, sql_query, max_rows=200):
    cleaned = sql_query.strip().rstrip(";")
    lower = cleaned.lower()
    if not (lower.startswith("select") or lower.startswith("with")):
        raise ValueError("Only SELECT/WITH allowed")

    limited_sql = f"""
    SELECT *
    FROM (
        {cleaned}
    ) t
    LIMIT {int(max_rows)}
    """
    return con.execute(limited_sql).df()


def get_alert_threshold_pp(window_days: int) -> float:
    """
    周期越长，阈值越严格（更容易捕捉小变化）
    你可以按业务调整。
    """
    if window_days >= 30:
        return 0.15
    if window_days >= 14:
        return 0.20
    return 0.30


def alert_level(delta_pp: float, threshold_pp: float) -> str:
    """
    delta_pp: 变化（pp），负数表示变差
    """
    if delta_pp <= -threshold_pp:
        return "🔴 异常下降"
    if delta_pp <= -threshold_pp / 2:
        return "🟠 轻微下降"
    if delta_pp >= threshold_pp:
        return "🟢 明显改善"
    return "⚪ 基本稳定"


def find_biggest_drop_pp(prev_row, last_row):
    """Return worst step and deltas in percentage points (pp). Negative means worse."""
    deltas = {
        "view→cart": (float(last_row["cr_view_to_cart"]) - float(prev_row["cr_view_to_cart"])) * 100.0,
        "cart→pay":  (float(last_row["cr_cart_to_pay"])  - float(prev_row["cr_cart_to_pay"]))  * 100.0,
        "view→pay":  (float(last_row["cr_view_to_pay"])  - float(prev_row["cr_view_to_pay"]))  * 100.0,
    }
    worst_step = min(deltas, key=deltas.get)  # most negative
    return worst_step, round(deltas[worst_step], 2), deltas


# =========================
# Funnel SQL Templates (data-relative last_Nd vs prev_Nd)
# =========================
def compare_funnel_sql_loose(ts: str, uid: str, evt: str, window_days: int) -> str:
    n = int(window_days)
    return f"""
WITH bounds AS (
  SELECT MAX(to_timestamp({ts}/1000)) AS max_ts
  FROM events
),
base AS (
  SELECT
    {uid} AS user_id,
    {evt} AS event,
    to_timestamp({ts} / 1000) AS ts,
    CASE
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL {n} DAY THEN 'last_{n}d'
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL {2*n} DAY
       AND to_timestamp({ts} / 1000) <  (SELECT max_ts FROM bounds) - INTERVAL {n} DAY THEN 'prev_{n}d'
      ELSE NULL
    END AS period
  FROM events
  WHERE {evt} IN ('view','addtocart','transaction')
),
agg AS (
  SELECT
    period,
    COUNT(DISTINCT CASE WHEN event='view' THEN user_id END)        AS view_users,
    COUNT(DISTINCT CASE WHEN event='addtocart' THEN user_id END)   AS cart_users,
    COUNT(DISTINCT CASE WHEN event='transaction' THEN user_id END) AS pay_users
  FROM base
  WHERE period IS NOT NULL
  GROUP BY 1
)
SELECT
  period,
  view_users, cart_users, pay_users,
  cart_users * 1.0 / NULLIF(view_users,0) AS cr_view_to_cart,
  pay_users  * 1.0 / NULLIF(cart_users,0) AS cr_cart_to_pay,
  pay_users  * 1.0 / NULLIF(view_users,0) AS cr_view_to_pay
FROM agg
ORDER BY CASE period WHEN 'prev_{n}d' THEN 1 ELSE 2 END
"""


def compare_funnel_sql_strict(ts: str, uid: str, evt: str, window_days: int) -> str:
    n = int(window_days)
    return f"""
WITH bounds AS (
  SELECT MAX(to_timestamp({ts}/1000)) AS max_ts
  FROM events
),
filtered AS (
  SELECT
    {uid} AS user_id,
    {evt} AS event,
    to_timestamp({ts} / 1000) AS ts,
    CASE
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL {n} DAY THEN 'last_{n}d'
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL {2*n} DAY
       AND to_timestamp({ts} / 1000) <  (SELECT max_ts FROM bounds) - INTERVAL {n} DAY THEN 'prev_{n}d'
      ELSE NULL
    END AS period
  FROM events
  WHERE {evt} IN ('view','addtocart','transaction')
),
u AS (
  SELECT
    user_id,
    period,
    MIN(CASE WHEN event='view'        THEN ts END) AS t_view,
    MIN(CASE WHEN event='addtocart'   THEN ts END) AS t_cart,
    MIN(CASE WHEN event='transaction' THEN ts END) AS t_pay
  FROM filtered
  WHERE period IS NOT NULL
  GROUP BY 1,2
),
agg AS (
  SELECT
    period,
    COUNT(*) FILTER (WHERE t_view IS NOT NULL) AS view_users,
    COUNT(*) FILTER (WHERE t_view IS NOT NULL AND t_cart IS NOT NULL AND t_cart >= t_view) AS cart_users,
    COUNT(*) FILTER (
      WHERE t_view IS NOT NULL AND t_cart IS NOT NULL AND t_pay IS NOT NULL
        AND t_cart >= t_view AND t_pay >= t_cart
    ) AS pay_users
  FROM u
  GROUP BY 1
)
SELECT
  period,
  view_users, cart_users, pay_users,
  cart_users * 1.0 / NULLIF(view_users,0) AS cr_view_to_cart,
  pay_users  * 1.0 / NULLIF(cart_users,0) AS cr_cart_to_pay,
  pay_users  * 1.0 / NULLIF(view_users,0) AS cr_view_to_pay
FROM agg
ORDER BY CASE period WHEN 'prev_{n}d' THEN 1 ELSE 2 END
"""


# =========================
# UI - Sidebar
# =========================
st.sidebar.title("设置")
deep_mode = st.sidebar.toggle("深度分析（reasoner）", value=False, help="开启后推理更深但更慢")
strict_mode = st.sidebar.toggle("严格漏斗（顺序）", value=False, help="严格：必须 view→addtocart→transaction 顺序满足")
compare_mode = st.sidebar.toggle("对比（本期 vs 上期）", value=True, help="last_Nd vs prev_Nd，基于数据最大时间")
window_days = st.sidebar.radio("对比周期长度", options=[7, 14, 30], index=0, horizontal=True)
max_rows = st.sidebar.slider("SQL 返回最大行数", 50, 500, 200, 50)
uploaded = st.sidebar.file_uploader("上传 events.csv", type=["csv"])

# =========================
# Main
# =========================
st.title("📊 运营漏斗洞察（自动生成）")

if not uploaded:
    st.info("请在左侧上传 RetailRocket 的 events.csv")
    st.stop()

df = pd.read_csv(uploaded)

with st.expander("数据预览", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

# 自动识别列名
cols = {c.lower(): c for c in df.columns}
ts_col = cols.get("timestamp")
uid_col = cols.get("visitorid")
evt_col = cols.get("event")

if not all([ts_col, uid_col, evt_col]):
    st.error("未识别到 timestamp / visitorid / event 列。请确认 CSV 列名。")
    st.stop()

# DuckDB
con = duckdb.connect(database=":memory:")
con.register("events_df", df)
con.execute("CREATE TABLE events AS SELECT * FROM events_df")

# 数据最大时间（告诉用户“最近N天”是相对数据时间）
max_ts = con.execute(f"SELECT MAX(to_timestamp({ts_col}/1000)) FROM events").fetchone()[0]
st.caption(
    f"时间窗口说明：以数据最新时间 **{max_ts}** 为基准，计算 last_{window_days}d（最近{window_days}天）"
    f" 与 prev_{window_days}d（前{window_days}天）。"
)

if not compare_mode:
    st.warning("当前版本为“自动洞察”模式，建议开启对比模式（本期 vs 上期）。")
    st.stop()

# =========================
# Run funnel (Auto)
# =========================
sql_query = (
    compare_funnel_sql_strict(ts_col, uid_col, evt_col, window_days)
    if strict_mode else
    compare_funnel_sql_loose(ts_col, uid_col, evt_col, window_days)
)

try:
    result_df = run_sql(con, sql_query, max_rows=max_rows)
except Exception as e:
    st.error("SQL 执行失败（可能是列名/数据类型问题）。")
    st.code(sql_query, language="sql")
    st.code(str(e))
    st.stop()

st.subheader("📈 对比漏斗结果（按用户）")
st.dataframe(result_df, use_container_width=True)

# =========================
# Auto Insight Cards
# =========================
prev_tag = f"prev_{window_days}d"
last_tag = f"last_{window_days}d"
order = {prev_tag: 0, last_tag: 1}

df2 = result_df.copy()
if "period" not in df2.columns or df2.shape[0] < 2:
    st.warning("未得到本期/上期两期结果，请检查数据或 SQL。")
    st.stop()

df2["__o"] = df2["period"].map(order)
df2 = df2.sort_values("__o").drop(columns="__o")

prev = df2[df2["period"] == prev_tag].iloc[0]
last = df2[df2["period"] == last_tag].iloc[0]

worst_step, worst_pp, deltas = find_biggest_drop_pp(prev, last)

threshold_pp = get_alert_threshold_pp(window_days)
levels = {k: alert_level(v, threshold_pp) for k, v in deltas.items()}

st.subheader("🚨 自动洞察（本期 vs 上期）")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("浏览用户（view）", int(last["view_users"]), int(last["view_users"] - prev["view_users"]))
with c2:
    st.metric("加购用户（addtocart）", int(last["cart_users"]), int(last["cart_users"] - prev["cart_users"]))
with c3:
    st.metric("成交用户（transaction）", int(last["pay_users"]), int(last["pay_users"] - prev["pay_users"]))

st.markdown(f"**最大下降步骤**：**{worst_step}**（{worst_pp} pp）")
st.caption(f"预警阈值：{threshold_pp:.2f} pp（周期越长阈值越严格）")

st.write({
    "view→cart 变化(pp)": f"{deltas['view→cart']:.2f} pp  {levels['view→cart']}",
    "cart→pay 变化(pp)":  f"{deltas['cart→pay']:.2f} pp  {levels['cart→pay']}",
    "view→pay 变化(pp)":  f"{deltas['view→pay']:.2f} pp  {levels['view→pay']}",
})

# =========================
# LLM Interpretation (Auto report)
# =========================
model = REASONER_MODEL if deep_mode else CHAT_MODEL

report_prompt = f"""
你是互联网产品运营数据分析助手。下面是漏斗对比结果（按用户）：
{df2.to_dict(orient="records")}

变化（pp）：
- view→cart: {deltas['view→cart']:.2f} pp（{levels['view→cart']}）
- cart→pay: {deltas['cart→pay']:.2f} pp（{levels['cart→pay']}）
- view→pay: {deltas['view→pay']:.2f} pp（{levels['view→pay']}）
最大下降步骤：{worst_step}（{worst_pp} pp）
时间窗口：last_{window_days}d vs prev_{window_days}d（基于数据最大时间 {max_ts}）
漏斗模式：{"严格（顺序）" if strict_mode else "宽松（发生过即算）"}

请输出一份“运营洞察日报”，必须包含以下小标题（markdown）：
## 一句话结论
## 变化最大的步骤与影响
## 可能原因（假设）
## 下一步排查/拆解建议（按优先级）
要求：
- 不要反问用户，不要要求补充数据
- 原因必须标明是假设，不要装作已验证
- 建议要可执行（例如：按 itemid/时间段/高频用户/新老用户拆解、对比异常时段等）
"""

with st.spinner("生成运营洞察日报中…"):
    explanation = deepseek_chat(
        messages=[{"role": "user", "content": report_prompt}],
        model=model,
        temperature=0.3
    )

st.subheader("🧠 运营洞察日报（自动生成）")
st.markdown(explanation)

# =========================
# 📥 Export Report (Markdown)
# =========================
st.subheader("📥 一键导出日报（Markdown）")

report_md = f"""# 运营洞察日报（漏斗对比）

**数据窗口（以数据最新时间为基准）**：{max_ts}  
**对比周期**：last_{window_days}d vs prev_{window_days}d  
**漏斗模式**：{"严格（顺序）" if strict_mode else "宽松（发生过即算）"}  
**深度分析**：{"是" if deep_mode else "否"}

## 核心指标（{last_tag}）
- 浏览用户（view）：{int(last["view_users"]):,}
- 加购用户（addtocart）：{int(last["cart_users"]):,}
- 成交用户（transaction）：{int(last["pay_users"]):,}
- view→cart：{float(last["cr_view_to_cart"])*100:.2f}%
- cart→pay：{float(last["cr_cart_to_pay"])*100:.2f}%
- view→pay：{float(last["cr_view_to_pay"])*100:.2f}%

## 变化摘要（{last_tag} - {prev_tag}，pp；阈值 {threshold_pp:.2f} pp）
- view→cart：{deltas["view→cart"]:.2f} pp（{levels["view→cart"]}）
- cart→pay：{deltas["cart→pay"]:.2f} pp（{levels["cart→pay"]}）
- view→pay：{deltas["view→pay"]:.2f} pp（{levels["view→pay"]}）
- **最大下降步骤**：**{worst_step}**（{worst_pp} pp）

---

{explanation}
"""

report_date = max_ts.strftime("%Y-%m-%d") if max_ts else "unknown_date"
funnel_tag = "严格" if strict_mode else "宽松"
deep_tag = "_深度" if deep_mode else ""
file_name = f"运营洞察日报_{report_date}_{window_days}d_{funnel_tag}{deep_tag}.md"

st.caption(f"导出文件名：{file_name}")

st.download_button(
    label="⬇️ 下载 Markdown 日报（.md）",
    data=report_md.encode("utf-8"),
    file_name=file_name,
    mime="text/markdown"
)

# =========================
# Optional follow-up chat (collapsed)
# =========================
with st.expander("💬 可选：继续追问（基于当前洞察）", expanded=False):
    q = st.chat_input("例如：为什么 view→cart 会低？能按 itemid/类目拆一下吗？")
    if q:
        follow_prompt = f"""
这是当前的对比漏斗结果（按用户）：
{df2.to_dict(orient="records")}

用户追问：{q}

请基于已有结果回答；如果需要新的维度/SQL，请明确说明需要哪些字段以及下一步怎么查。
"""
        with st.spinner("生成回答中…"):
            ans = deepseek_chat(
                messages=[{"role": "user", "content": follow_prompt}],
                model=model,
                temperature=0.3
            )
        st.markdown(ans)
