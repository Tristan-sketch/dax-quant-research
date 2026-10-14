import numpy as np
import pandas as pd
import pytest

from daxquant import backtest as bt
from daxquant import config as C
from daxquant import metrics, quality, signals, stats


def _toy_returns(n=400, k=12, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0.0004, 0.015, (n, k)), index=days,
                       columns=[f"S{i}" for i in range(k)])
    dax = pd.Series(rng.normal(0.0003, 0.01, n), index=days)
    return ret, dax


def test_no_lookahead_in_signals():
    """Scrambling everything after the formation date must not change the signals."""
    ret, dax = _toy_returns()
    t = ret.index[300]
    base = signals.compute_signals(ret, dax, pd.DatetimeIndex([t]))[t]
    ret2, dax2 = ret.copy(), dax.copy()
    ret2.iloc[301:] = ret2.iloc[301:].to_numpy()[::-1] * 5
    dax2.iloc[301:] = -dax2.iloc[301:]
    other = signals.compute_signals(ret2, dax2, pd.DatetimeIndex([t]))[t]
    pd.testing.assert_frame_equal(base, other)


def test_momentum_skips_last_month():
    ret, dax = _toy_returns()
    t_pos = 300
    t = ret.index[t_pos]
    ret2 = ret.copy()
    ret2.iloc[t_pos - 5, 0] = 0.5  # inside the skipped month
    a = signals.compute_signals(ret, dax, pd.DatetimeIndex([t]))[t]
    b = signals.compute_signals(ret2, dax, pd.DatetimeIndex([t]))[t]
    assert a.at["S0", "MOM"] == pytest.approx(b.at["S0", "MOM"])
    assert a.at["S0", "STR"] != pytest.approx(b.at["S0", "STR"])


def test_buy_and_hold_matches_average_of_holding_returns():
    ret, _ = _toy_returns(n=60, k=4)
    names = ["S0", "S1", "S2", "S3"]
    f = ret.index[0]
    ex = pd.Series([f], index=[f])
    res = bt.simulate(ret, {f: bt.equal_weights(names)}, ex, cost_bps=0)
    port = (1 + res.daily["gross"].iloc[1:]).prod() - 1
    single = ((1 + ret.iloc[1:]).prod() - 1).mean()
    assert port == pytest.approx(single, rel=1e-10)


def test_turnover_and_costs():
    ret, _ = _toy_returns(n=40, k=4)
    f0, f1 = ret.index[0], ret.index[20]
    ex = pd.Series([f0, f1], index=[f0, f1])
    targets = {f0: bt.equal_weights(["S0", "S1"]), f1: bt.equal_weights(["S2", "S3"])}
    res = bt.simulate(ret, targets, ex, cost_bps=10)
    assert res.daily.at[f0, "turnover"] == pytest.approx(1.0)
    assert res.daily.at[f1, "turnover"] == pytest.approx(2.0)  # sell everything, buy everything
    assert res.daily.at[f1, "cost"] == pytest.approx(2.0 * 0.001)


def test_pick_buckets_sizes():
    s = pd.Series(np.arange(31.0), index=[f"S{i}" for i in range(31)])
    top, bottom = bt.pick_buckets(s, 3)
    assert len(top) == len(bottom) == 10
    assert "S30" in top and "S0" in bottom


def test_split_and_spinoff_corrections():
    days = pd.bdate_range("2007-06-05", periods=6)
    prices = pd.DataFrame({"DB1.DE": [100, 101, 102, 51, 51.5, 52.0]}, index=days)
    bench = pd.DataFrame({"DAX": [1000, 1001, 1002, 1003, 1004, 1005.0]}, index=days)
    actions = pd.DataFrame({"date": [days[3]], "ticker": ["DB1.DE"], "action": ["split"],
                            "value": [2], "note": ["test"]})
    ret, log = quality.clean_returns(prices, bench, actions)
    assert ret.at[days[3], "DB1.DE"] == pytest.approx(0.0, abs=1e-12)  # 51/102 x 2 - 1
    assert len(log) == 1


def test_metrics_basics():
    idx = pd.period_range("2020-01", periods=4, freq="M")
    r = pd.Series([0.1, -0.5, 0.2, 0.0], index=idx)
    assert metrics.drawdown(r).min() == pytest.approx(-0.5)
    assert metrics.max_dd_duration(r) == 3
    assert metrics.cagr(pd.Series([0.01] * 12, index=pd.period_range("2020-01", periods=12, freq="M"))) == \
        pytest.approx(1.01**12 - 1)


def test_holm_and_deflated_sharpe():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = stats.holm(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.01, 0.04, 240))
    trials = rng.normal(0, 0.05, 36)
    assert stats.deflated_sharpe(x, trials) < stats.probabilistic_sharpe(x)


def test_frozen_spec_unchanged():
    """The live record is only meaningful if the base specification never moves."""
    assert C.FREEZE_DATE == "2026-10-14"
    assert C.BASE == C.Spec(lookback=252, skip=21, reversal_window=21, min_obs=200, beta_shrink=0.6,
                            winsor_z=3.0, buckets=3, lag=1, cost_bps=10.0, borrow_bps_pa=25.0,
                            composite=("MOM", "STR", "LVOL", "LBETA", "HIGH52"))
