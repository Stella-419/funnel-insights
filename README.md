# Funnel Insights

Funnel Insights 是一个面向产品与数据分析场景的轻量级漏斗分析工具，用于快速发现转化率异常并定位潜在问题环节。

它可以基于用户事件数据：

- 构建任意 N 步转化漏斗
- 对比不同时间窗口的转化变化
- 自动定位异常下降步骤
- 推荐可解释的拆解维度，辅助分析原因

项目采用 **UI 与核心逻辑解耦** 的工程结构设计：

- Streamlit 仅负责交互与结果展示
- 核心分析逻辑可独立测试、复用与扩展
- 
## ✨ Features

- 🔢 支持任意 N 步漏斗分析
- 🔁 严格 / 非严格漏斗模式
- 📉 自动计算转化率与环比变化（pp）
- 🚨 智能判断异常下降等级
- 🧠 自动推荐可解释的拆解维度（如 device、country）
- ⚡ 基于 DuckDB 的 SQL 分析，速度快、零外部依赖
- 🧪 核心逻辑已覆盖 pytest 自动测试

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

核心分析逻辑全部集中在 `app/core.py` 中，避免与 UI 耦合，便于测试、复用和后续扩展。
```

## 🧪 Testing

项目核心逻辑已使用 pytest 覆盖。

在项目根目录运行：

```bash
pytest -q
```

## 🛠 Tech Stack
- Python
- Streamlit
- DuckDB
- Pandas
- Pytest


测试将验证：
  - 漏斗 SQL 生成逻辑
  - 转化率与环比变化计算
  - 异常步骤识别与判定逻辑

## 📌 Notes
- 当前示例数据为模拟生成，后续可接入真实埋点数据
- 项目结构支持进一步扩展为 API 或定时分析任务