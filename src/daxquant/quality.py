"""Turn raw adjusted closes into clean daily total returns.

Free data has known defects. Each fix is applied explicitly and written to a log
that ends up in the report, so nothing is silently altered:

1. Splits that Yahoo did not adjust (from corporate_actions.csv).
2. Spin-off ex-dates. Yahoo's spin-off adjustment for German stocks is
   inconsistent, so the parent's return on the ex-date is replaced by the DAX
   return of that day. This removes the error at the cost of one day of
   stock-specific information.
3. Bad ticks: a jump of more than 20% that is reversed the next day.
4. Series that fail the bad-tick test too often are dropped completely.
"""

import numpy as np
import pandas as pd

from . import config as C

BAD_TICK_JUMP = 0.20
BAD_TICK_NET = 0.05
MAX_BAD_TICK_SHARE = 0.002


def load_corporate_actions() -> pd.DataFrame:
    return pd.read_csv(C.REFERENCE_DIR / "corporate_actions.csv", parse_dates=["date"])


def calendar(bench: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading days are the days on which the DAX was computed."""
    return bench["DAX"].dropna().index


def clean_returns(
    prices: pd.DataFrame, bench: pd.DataFrame, actions: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = calendar(bench)
    px = prices.reindex(days)
    # Returns only between consecutive valid prices; gaps stay NaN.
    ret = px / px.ffill().shift(1) - 1
    ret = ret.where(px.notna())
    dax_ret = bench["DAX"].reindex(days).pct_change()
    log = []

    actions = load_corporate_actions() if actions is None else actions
    for a in actions.itertuples():
        if a.ticker not in ret:
            continue
        d = ret.index[ret.index.searchsorted(a.date)]
        before = ret.at[d, a.ticker]
        if a.action == "split":
            ret.at[d, a.ticker] = (1 + before) * float(a.value) - 1
        elif a.action == "spinoff":
            ret.at[d, a.ticker] = dax_ret.at[d]
        else:
            raise ValueError(f"unknown action {a.action}")
        log.append((d, a.ticker, a.action, before, ret.at[d, a.ticker], a.note))

    dropped = []
    for t in ret.columns:
        r = ret[t]
        nxt = r.shift(-1)
        bad = (
            (r.abs() > BAD_TICK_JUMP)
            & (nxt.abs() > BAD_TICK_JUMP)
            & (np.sign(r) != np.sign(nxt))
            & (((1 + r) * (1 + nxt) - 1).abs() < BAD_TICK_NET)
        )
        n_valid = r.notna().sum()
        if n_valid and bad.sum() / n_valid > MAX_BAD_TICK_SHARE:
            dropped.append(t)
            log.append((r.first_valid_index(), t, "series dropped", np.nan, np.nan,
                        f"{int(bad.sum())} reversed jumps in {n_valid} days"))
            continue
        for d in r.index[bad]:
            i = r.index.get_loc(d)
            for dd in r.index[i:i + 2]:
                log.append((dd, t, "bad tick", ret.at[dd, t], 0.0, "jump reversed next day"))
                ret.at[dd, t] = 0.0
    ret = ret.drop(columns=dropped)
    log = pd.DataFrame(log, columns=["date", "ticker", "action", "raw", "clean", "note"])
    return ret, log.sort_values("date").reset_index(drop=True)


def large_moves(ret: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Single-day moves above the threshold that survived cleaning (kept, but listed)."""
    s = ret.stack()
    s = s[s.abs() > threshold]
    out = s.rename("return").reset_index()
    out.columns = ["date", "ticker", "return"]
    return out.sort_values("date").reset_index(drop=True)


def price_index(ret: pd.DataFrame) -> pd.DataFrame:
    """Cumulative total-return index from clean returns, NaN where no data."""
    idx = (1 + ret.fillna(0)).cumprod()
    first = ret.notna().cummax()
    return idx.where(first & ret.notna())
