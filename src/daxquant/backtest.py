"""Daily portfolio simulation with monthly rebalancing.

Weights are set at the close of the execution day (formation day + lag) and
then drift with prices until the next rebalance. Turnover is measured against
the drifted weights, so costs reflect the trades that would actually be needed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


def formation_dates(days: pd.DatetimeIndex, start: str = C.FIRST_FORMATION) -> pd.DatetimeIndex:
    """Last trading day of each month, from `start` on."""
    s = pd.Series(days, index=days)
    month_ends = s.groupby(days.to_period("M")).max()
    d = pd.DatetimeIndex(month_ends.values)
    d = d[d >= pd.Timestamp(start)]
    # The latest month is only a formation date if it is complete.
    last = days[-1]
    if len(d) and d[-1] == last and last + pd.offsets.BMonthEnd(0) != last:
        d = d[:-1]
    return d


def execution_dates(days: pd.DatetimeIndex, formations: pd.DatetimeIndex, lag: int) -> pd.Series:
    pos = days.get_indexer(formations) + lag
    keep = pos < len(days)
    return pd.Series(days[pos[keep]], index=formations[keep])


def pick_buckets(score: pd.Series, n_buckets: int) -> tuple[list[str], list[str]]:
    s = score.dropna()
    size = len(s) // n_buckets
    if size == 0:
        return [], []
    order = s.sort_index().sort_values(ascending=False, kind="mergesort")
    return list(order.index[:size]), list(order.index[-size:])


def equal_weights(names: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


@dataclass
class Result:
    daily: pd.DataFrame  # gross, cost, net, turnover
    weights: dict  # execution date -> target weights


def simulate(ret: pd.DataFrame, targets: dict, exec_dates: pd.Series, cost_bps: float) -> Result:
    """targets: formation date -> weight Series (summing to 1)."""
    cols = {c: i for i, c in enumerate(ret.columns)}
    R = ret.fillna(0.0).to_numpy()
    days = ret.index
    by_pos = {}
    for f, w in targets.items():
        if f in exec_dates.index and len(w):
            v = np.zeros(len(cols))
            v[[cols[c] for c in w.index]] = w.to_numpy()
            by_pos[days.get_loc(exec_dates[f])] = v
    n = len(days)
    gross = np.zeros(n)
    cost = np.zeros(n)
    turn = np.zeros(n)
    w = np.zeros(len(cols))
    c = cost_bps / 1e4
    first = min(by_pos) if by_pos else n
    for i in range(first, n):
        if w.any():
            p = w @ R[i]
            gross[i] = p
            w = w * (1 + R[i]) / (1 + p)
        if i in by_pos:
            new = by_pos[i]
            turn[i] = np.abs(new - w).sum()
            cost[i] = turn[i] * c
            w = new
    daily = pd.DataFrame({"gross": gross, "cost": cost, "turnover": turn}, index=days)
    daily["net"] = daily["gross"] - daily["cost"]
    daily = daily.iloc[first:]
    return Result(daily, {exec_dates[f]: w for f, w in targets.items() if f in exec_dates.index})


def long_short(long: Result, short: Result, borrow_bps_pa: float) -> pd.DataFrame:
    idx = long.daily.index.union(short.daily.index)
    L = long.daily.reindex(idx).fillna(0.0)
    S = short.daily.reindex(idx).fillna(0.0)
    borrow = borrow_bps_pa / 1e4 / 252
    out = pd.DataFrame(index=idx)
    out["gross"] = L["gross"] - S["gross"]
    out["cost"] = L["cost"] + S["cost"] + borrow
    out["turnover"] = L["turnover"] + S["turnover"]
    out["net"] = out["gross"] - out["cost"]
    return out


def to_monthly(daily: pd.Series) -> pd.Series:
    m = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    return m


def forward_returns(ret: pd.DataFrame, exec_dates: pd.Series) -> dict:
    """Holding-period return of every stock from one execution date to the next."""
    growth = (1 + ret.fillna(0.0)).cumprod()
    ex = list(exec_dates.items())
    out = {}
    for (f, d0), (_, d1) in zip(ex[:-1], ex[1:]):
        out[f] = growth.loc[d1] / growth.loc[d0] - 1
    if ex:
        f, d0 = ex[-1]
        if d0 < ret.index[-1]:
            out[f] = growth.iloc[-1] / growth.loc[d0] - 1
    return out
