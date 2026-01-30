# =========================
# Analyzer1.py
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
    page_title="运营漏斗分析助手（RetailRocket）",
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
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"
REASONER_MODEL = "deepseek-reasoner"

# =========================
# Utils
# =========================
def deepseek_chat(messages, model, json_mode=False, temperature=0.3):
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


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


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


# =========================
# Funnel SQL Templates
# =========================

def compare_funnel_sql_loose(ts: str, uid: str, evt: str) -> str:
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
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL 7 DAY THEN 'last_7d'
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL 14 DAY
       AND to_timestamp({ts} / 1000) <  (SELECT max_ts FROM bounds) - INTERVAL 7 DAY THEN 'prev_7d'
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
ORDER BY CASE period WHEN 'prev_7d' THEN 1 ELSE 2 END
"""



def compare_funnel_sql_strict(ts: str, uid: str, evt: str) -> str:
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
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL 7 DAY THEN 'last_7d'
      WHEN to_timestamp({ts} / 1000) >= (SELECT max_ts FROM bounds) - INTERVAL 14 DAY
       AND to_timestamp({ts} / 1000) <  (SELECT max_ts FROM bounds) - INTERVAL 7 DAY THEN 'prev_7d'
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
ORDER BY CASE period WHEN 'prev_7d' THEN 1 ELSE 2 END
"""



# =========================
# UI - Sidebar
# =========================
st.sidebar.title("设置")

deep_mode = st.sidebar.toggle("深度分析（reasoner）", value=False)
strict_mode = st.sidebar.toggle("严格漏斗（顺序）", value=False)
compare_mode = st.sidebar.toggle("对比：最近7天 vs 前7天", value=True)

uploaded = st.sidebar.file_uploader("上传 events.csv", type=["csv"])

# =========================
# Main
# =========================
st.title("📊 互联网产品运营漏斗分析助手")

if not uploaded:
    st.info("请在左侧上传 RetailRocket 的 events.csv")
    st.stop()

df = pd.read_csv(uploaded)
st.write("数据预览", df.head())

# 自动识别列名
cols = {c.lower(): c for c in df.columns}
ts_col = cols.get("timestamp")
uid_col = cols.get("visitorid")
evt_col = cols.get("event")

if not all([ts_col, uid_col, evt_col]):
    st.error("未识别到 timestamp / visitorid / event 列")
    st.stop()

# DuckDB
con = duckdb.connect(database=":memory:")
con.register("events_df", df)
con.execute("CREATE TABLE events AS SELECT * FROM events_df")

# =========================
# Run funnel
# =========================
if compare_mode:
    sql_query = (
        compare_funnel_sql_strict(ts_col, uid_col, evt_col)
        if strict_mode else
        compare_funnel_sql_loose(ts_col, uid_col, evt_col)
    )

    result_df = run_sql(con, sql_query)
    st.subheader("📈 对比漏斗结果（按用户）")
    st.dataframe(result_df, use_container_width=True)

    # Summary
    if result_df.shape[0] == 2:
        prev, last = result_df.iloc[0], result_df.iloc[1]

        def pp(a, b): return round((b - a) * 100, 2)

        st.subheader("📌 变化摘要（百分点 pp）")
        st.write({
            "浏览用户变化": int(last.view_users - prev.view_users),
            "加购用户变化": int(last.cart_users - prev.cart_users),
            "成交用户变化": int(last.pay_users - prev.pay_users),
            "view→cart 变化(pp)": pp(prev.cr_view_to_cart, last.cr_view_to_cart),
            "cart→pay 变化(pp)": pp(prev.cr_cart_to_pay, last.cr_cart_to_pay),
            "view→pay 变化(pp)": pp(prev.cr_view_to_pay, last.cr_view_to_pay),
        })

else:
    st.warning("当前版本请开启对比模式")

# =========================
# LLM Interpretation
# =========================
model = REASONER_MODEL if deep_mode else CHAT_MODEL

prompt = f"""
这是一个电商漏斗分析结果（按用户）：
{result_df.to_dict(orient="records")}

请你：
1) 指出最近7天相比前7天，变化最大的漏斗步骤
2) 给出 2-3 个可能原因（假设）
3) 给出 3 条可执行的运营拆解建议
"""

with st.spinner("生成运营解读中…"):
    explanation = deepseek_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )

st.subheader("🧠 运营解读与建议")
st.markdown(explanation)
