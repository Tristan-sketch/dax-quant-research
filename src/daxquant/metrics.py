"""Performance metrics on monthly returns."""

import numpy as np
import pandas as pd
from scipy import stats as st

from . import stats


def wealth(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = wealth(r)
    return w / w.cummax() - 1


def max_dd_duration(r: pd.Series) -> int:
    """Longest stretch, in periods, spent below a previous peak."""
    under = drawdown(r) < 0
    runs = under.groupby((~under).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def cagr(r: pd.Series, periods: int = 12) -> float:
    r = r.dropna()
    if not len(r):
        return np.nan
    return float(wealth(r).iloc[-1] ** (periods / len(r)) - 1)


def capture(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, bench], axis=1).dropna()
    up = df[df.iloc[:, 1] > 0]
    dn = df[df.iloc[:, 1] < 0]
    upc = (1 + up.iloc[:, 0]).prod() ** (1 / len(up)) - 1
    upb = (1 + up.iloc[:, 1]).prod() ** (1 / len(up)) - 1
    dnc = (1 + dn.iloc[:, 0]).prod() ** (1 / len(dn)) - 1
    dnb = (1 + dn.iloc[:, 1]).prod() ** (1 / len(dn)) - 1
    return float(upc / upb), float(dnc / dnb)


def summary(
    r: pd.Series,
    rf: pd.Series,
    bench: pd.Series | None = None,
    self_financing: bool = False,
    turnover: pd.Series | None = None,
    gross: pd.Series | None = None,
) -> dict:
    """All headline statistics for one monthly return series.

    self_financing: long-short spreads already are excess returns.
    turnover: monthly traded value (sum of |weight changes|, both sides).
    gross: monthly gross returns, for the break-even cost.
    """
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0)
    ex = r if self_financing else r - rf
    q05 = r.quantile(0.05)
    dd = drawdown(r)
    out = {
        "months": len(r),
        "cagr": cagr(r),
        "mean_excess": ex.mean() * 12,
        "vol": r.std() * np.sqrt(12),
        "sharpe": stats.sharpe_per_period(ex) * np.sqrt(12),
        "sharpe_se": stats.sharpe_se(ex) * np.sqrt(12),
        "sortino": ex.mean() * 12 / (np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12)),
        "max_dd": dd.min(),
        "max_dd_months": max_dd_duration(r),
        "calmar": cagr(r) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "skew": float(st.skew(r)),
        "kurt": float(st.kurtosis(r)),
        "var95": -q05,
        "cvar95": -r[r <= q05].mean(),
        "hit_rate": (r > 0).mean(),
        "best": r.max(),
        "worst": r.min(),
        "psr": stats.probabilistic_sharpe(ex),
    }
    t = stats.nw_mean_test(ex)
    out.update({"mean_t": t["t"], "mean_p": t["p"]})
    if bench is not None:
        b = bench.reindex(r.index)
        bex = b - rf
        reg = stats.capm(ex, bex)
        active = r - b
        upc, dnc = capture(r, b)
        out.update(
            {
                "beta": reg["beta"],
                "alpha": reg["alpha"] * 12,
                "alpha_t": reg["alpha_t"],
                "alpha_p": reg["alpha_p"],
                "corr": r.corr(b),
                "te": active.std() * np.sqrt(12),
                "ir": active.mean() / active.std() * np.sqrt(12),
                "up_capture": upc,
                "down_capture": dnc,
            }
        )
    if turnover is not None:
        tv = turnover.reindex(r.index).fillna(0)
        out["turnover"] = tv.mean() * 12 / 2  # one-way, per year
        if gross is not None:
            g = gross.reindex(r.index)
            traded = tv.mean() * 12
            gross_mean = (g if self_financing else g - rf).mean() * 12
            out["breakeven_bps"] = gross_mean / traded * 1e4 if traded > 0 else np.nan
    return out


def rolling_sharpe(r: pd.Series, window: int = 36) -> pd.Series:
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(12)


def calendar_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    years = r.index.year
    return (1 + r).groupby(years).prod() - 1
