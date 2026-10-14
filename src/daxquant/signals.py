"""Cross-sectional signals. Higher value = expected to outperform.

Every signal for formation date t uses only returns up to and including t.
tests/test_no_lookahead.py checks this by scrambling all later data.
"""

import numpy as np
import pandas as pd

from . import config as C


def _window_stats(ret: pd.DataFrame, dax_ret: pd.Series, t_pos: int, spec: C.Spec) -> pd.DataFrame:
    win = ret.iloc[t_pos - spec.lookback + 1 : t_pos + 1]
    valid = win.notna().sum()
    ok = valid >= spec.min_obs

    growth = (1 + win.fillna(0)).cumprod()
    mom_part = (1 + win.iloc[: spec.lookback - spec.skip].fillna(0)).prod() - 1
    rev_part = (1 + win.iloc[-spec.reversal_window :].fillna(0)).prod() - 1
    vol = win.std() * np.sqrt(252)

    m = dax_ret.iloc[t_pos - spec.lookback + 1 : t_pos + 1]
    x = win.sub(win.mean())
    y = m - m.mean()
    cov = x.mul(y, axis=0).where(win.notna()).sum() / (valid - 1)
    beta_raw = cov / m.var()
    beta = spec.beta_shrink * beta_raw + (1 - spec.beta_shrink) * 1.0

    high = growth.max()
    last = growth.iloc[-1]
    live = win.iloc[-spec.reversal_window :].notna().sum() > 0  # still trading recently

    out = pd.DataFrame(
        {
            "MOM": mom_part,
            "STR": -rev_part,
            "LVOL": -vol,
            "LBETA": -beta,
            "HIGH52": last / high,
            "vol": vol,
            "beta": beta,
        }
    )
    return out[ok & live]


def compute_signals(
    ret: pd.DataFrame, dax_ret: pd.Series, dates: pd.DatetimeIndex, spec: C.Spec = C.BASE
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Raw signals for every formation date; stocks without enough history are absent."""
    out = {}
    for t in dates:
        pos = ret.index.get_loc(t)
        if pos + 1 < spec.lookback:
            continue
        out[t] = _window_stats(ret, dax_ret, pos, spec)
    return out


def zscore(s: pd.Series, clip: float) -> pd.Series:
    z = (s - s.mean()) / s.std()
    return z.clip(-clip, clip)


def composite(sig: pd.DataFrame, spec: C.Spec = C.BASE) -> pd.Series:
    z = pd.concat([zscore(sig[f], spec.winsor_z) for f in spec.composite], axis=1)
    return z.mean(axis=1)
