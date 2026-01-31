# Funnel Insights

一个面向 产品 / 运营 / 数据分析 场景的轻量级事件漏斗分析工具，支持自动对比转化变化、定位异常步骤，并结合大模型生成可读的运营洞察与追问分析。

本项目以 “稳定可解释的 demo” 为目标，核心逻辑简单清晰，适合作为：

- 产品经理的数据分析工具原型
- 漏斗分析方法论示例
- LLM + 数据分析结合的参考实现

## ✨ Features

1️⃣ 事件漏斗分析（N-Step）
支持任意 N 步事件漏斗（如 page_view → click → purchase）

- 用户级别漏斗（按 user 去重）
- 支持 严格 / 非严格 两种口径
  - 非严格：只要发生过事件即可计入
  - 严格：必须按顺序发生

2️⃣ 周期对比（Prev / Last）

- 自动以 **数据中的最大时间戳** 作为基准
- 对比：
  - last_{N}d（最近 N 天）
  -  prev_{N}d（再往前 N 天）

- 计算每一步：
  - 用户数 
  - 转化率 
  - 转化率变化（pp）

3️⃣ 自动异常识别

- 自动找出 转化下降最大的步骤

- 根据周期长度动态设置阈值（7 / 14 / 30 天）

- 输出直观状态：

    - 🔴 异常下降
    - 🟠 轻微下降 
    - ⚪ 基本稳定 
    - 🟢 明显改善

- 给出可执行的运营提示（rule-based）

4️⃣ 智能维度推荐（Breakdown）
- 自动扫描数据列，推荐适合拆解的维度
    - 如：device、country
- 排除： 
  - user_id / event / timestamp 
  - 高基数、用户内多值字段
- 支持按单一维度拆解漏斗，对比各分组的异常程度

5️⃣ LLM 运营洞察日报

- 一键生成结构化「运营洞察日报」（Markdown）
- 内容包括： 
  - 一句话结论 
  - 变化最大的步骤及影响 
  - 可能原因（假设） 
  - 下一步可执行动作 
- 支持普通 / 深度分析模式

6️⃣ Chatbot 追问分析
- 基于当前漏斗结果的上下文追问 
- 示例问题： 
  - 哪个分组最值得优先排查？ 
  - 如果要进一步确认原因，需要补哪些字段？ 
- 回答要求强约束： 
  - 先结论
  - 再假设 
  - 再可执行动作 
  - 明确是否需要额外数据 / SQL

## 🧠 Demo 的取数与计算口径说明

**漏斗取数逻辑（SQL）**
- 数据表：events 
- 核心字段： 
  - user_id：用户标识 
  - event：事件名 
  - timestamp：毫秒级时间戳
  
**非严格漏斗**
  - 每一步独立统计： 
  - 在周期内发生过该事件的 去重用户数 
  - 不要求事件顺序

**严格漏斗**
- 先在用户粒度聚合： 
  - 每个用户在周期内各步骤的**最早发生时间**
- 仅当：
  - Step1 ≤ Step2 ≤ Step3 …
- 才计入后续步骤

**周期划分**

- 以 MAX(timestamp) 作为时间基准
- 时间区间： 
  - last：[max_ts - N days, max_ts)
  - prev：[max_ts - 2N days, max_ts - N days)

**转化率计算**
- r(i→j) = s_j / s_i
- 转化率变化（pp）：
  - (last_rate - prev_rate) × 100

## 🚀 Quick Start

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/funnel-insights.git
cd funnel-insights
```
### 2. 创建并激活虚拟环境（可选但推荐）
```bash
python -m venv .venv
source .venv/bin/activate   # Windows 使用 .venv\Scripts\activate
```
### 3. 安装依赖
```bash
pip install -r requirements.txt
```
### 4. 启动 Streamlit 应用
```bash
streamlit run app/Analyzer.py

启动后在浏览器中访问：
http://localhost:8501
```
### 5.配置 DeepSeek Key（可选）
```bash
export DEEPSEEK_API_KEY=your_api_key

未配置 Key 也可以正常使用 所有漏斗分析功能仅无法生成日报与 Chatbot。_
```

## 📁 Project Structure
```bash
funnel-insights/
├── app/
│   ├── Analyzer.py        # Streamlit UI 入口
│   └── core.py            # 核心漏斗分析逻辑
├── tests/
│   └── test_core.py       # core 模块的 pytest 测试
├── .streamlit/
│   └── secrets.toml       # 本地 / 云端配置（不入库）
├── requirements.txt
├── pytest.ini
└── README.md

Analyzer.py：只负责交互、展示、调用 core

core.py：不依赖 Streamlit，可独立测试、复用

demo 数据完全自生成，保证开箱即用
```

## 🧪 Testing

**项目核心逻辑已使用 pytest 覆盖。
在项目根目录运行：**


```bash
pytest -q
```

## 🛠 Tech Stack
- Python
- Streamlit
- DuckDB
- Pandas
- Pytest


**测试覆盖：**
- 阈值与预警逻辑 
- 漏斗 SQL 是否可执行 
- 严格漏斗的单调性 
- 维度推荐规则 
- 转化率与 pp 计算正确性

## 📌 可演进方向
- Session 级漏斗 / 时间间隔约束 
- 多维度交叉拆解 
- 日粒度 / 周粒度趋势 
- 接入真实埋点仓库（ClickHouse / BigQuery） 
- 自动生成 SQL 排查模板 
- 将 Chatbot 升级为「分析 Copilot」
