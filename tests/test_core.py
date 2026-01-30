import duckdb
import pandas as pd

from app import core


def run_query(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.register("events", df)
        return con.execute(sql).df()
    finally:
        con.close()


def test_threshold_pp():
    assert core.threshold_pp(7) == 0.30
    assert core.threshold_pp(14) == 0.20
    assert core.threshold_pp(30) == 0.15


def test_level():
    th = 0.30
    assert "异常下降" in core.level(-0.30, th)
    assert "轻微下降" in core.level(-0.16, th)
    assert "明显改善" in core.level(0.30, th)
    assert "基本稳定" in core.level(0.01, th)


def test_breakdown_candidates_should_include_device_country():
    df = core.make_sample_data(200)
    cands = core.infer_breakdown_candidates(df, uid_col="user_id", evt_col="event", ts_col="timestamp")
    cols = [c[0] for c in cands]
    assert "device" in cols
    assert "country" in cols
    assert "event" not in cols
    assert "timestamp" not in cols
    assert "user_id" not in cols


def test_funnel_sql_runs_loose_overall():
    df = core.make_sample_data(300)
    steps = ["page_view", "click", "purchase"]
    sql = core.funnel_sql_nstep("user_id", "event", "timestamp", steps, window_days=7, strict=False, breakdown_col=None)
    out = run_query(df, sql)

    assert {"period", "s1", "s2", "s3"}.issubset(set(out.columns))
    assert out["period"].nunique() >= 1
    assert len(out) >= 1


def test_funnel_sql_runs_strict_overall_and_monotonic():
    df = core.make_sample_data(400)
    steps = ["page_view", "click", "purchase"]
    sql = core.funnel_sql_nstep("user_id", "event", "timestamp", steps, window_days=7, strict=True, breakdown_col=None)
    out = run_query(df, sql)

    assert {"period", "s1", "s2", "s3"}.issubset(set(out.columns))
    for _, row in out.iterrows():
        assert int(row["s1"]) >= int(row["s2"]) >= int(row["s3"])


def test_funnel_sql_runs_with_breakdown():
    df = core.make_sample_data(400)
    steps = ["page_view", "click", "purchase"]
    sql = core.funnel_sql_nstep("user_id", "event", "timestamp", steps, window_days=7, strict=False, breakdown_col="device")
    out = run_query(df, sql)

    assert "bd" in out.columns
    assert out["bd"].nunique() >= 1
    assert len(out) >= 1


def test_compute_rates_and_find_worst_delta():
    prev = pd.Series({"s1": 100, "s2": 50, "s3": 10})
    last = pd.Series({"s1": 100, "s2": 40, "s3": 20})
    rates, deltas = core.compute_rates_and_deltas(prev, last, step_count=3)

    assert abs(deltas["d1_2"] - (-10.0)) < 1e-6
    assert abs(deltas["d2_3"] - (30.0)) < 1e-6

    worst_k, worst_pp = core.find_worst_delta(deltas)
    assert worst_k == "d1_2"
    assert abs(worst_pp - (-10.0)) < 1e-6
