"""Inference: HAC t-statistics, Sharpe ratio uncertainty, multiple testing."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def nw_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def nw_mean_test(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 12:
        return {"mean": np.nan, "t": np.nan, "p": np.nan, "n": n}
    res = sm.OLS(x.to_numpy(), np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(n)})
    return {"mean": float(res.params[0]), "t": float(res.tvalues[0]), "p": float(res.pvalues[0]), "n": n}


def capm(y: pd.Series, x: pd.Series) -> dict:
    """y and x are excess returns. Alpha per period with HAC standard errors."""
    df = pd.concat([y, x], axis=1).dropna()
    n = len(df)
    if n < 12:
        return {"alpha": np.nan, "alpha_t": np.nan, "alpha_p": np.nan, "beta": np.nan, "r2": np.nan}
    X = sm.add_constant(df.iloc[:, 1].to_numpy())
    res = sm.OLS(df.iloc[:, 0].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(n)})
    return {
        "alpha": float(res.params[0]),
        "alpha_t": float(res.tvalues[0]),
        "alpha_p": float(res.pvalues[0]),
        "beta": float(res.params[1]),
        "r2": float(res.rsquared),
    }


def sharpe_per_period(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std() > 0 else np.nan


def sharpe_se(x: pd.Series) -> float:
    """Standard error of the per-period Sharpe ratio, non-normal returns (Mertens 2002)."""
    x = x.dropna()
    sr = sharpe_per_period(x)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    return float(np.sqrt((1 + 0.5 * sr**2 - g3 * sr + (g4 - 3) / 4 * sr**2) / len(x)))


def probabilistic_sharpe(x: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), Bailey & Lopez de Prado (2012). Per-period units."""
    x = x.dropna()
    n = len(x)
    sr = sharpe_per_period(x)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_srs: np.ndarray) -> float:
    """Expected maximum Sharpe of N unskilled trials (Bailey & Lopez de Prado 2014)."""
    srs = np.asarray(trial_srs, dtype=float)
    srs = srs[np.isfinite(srs)]
    n = len(srs)
    if n < 2:
        return 0.0
    sd = srs.std(ddof=1)
    return float(
        sd
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
        )
    )


def deflated_sharpe(x: pd.Series, trial_srs: np.ndarray) -> float:
    return probabilistic_sharpe(x, expected_max_sharpe(trial_srs))


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    p = pvalues.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=p.index).reindex(pvalues.index)


def rank_ic(score: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([score, fwd], axis=1).dropna()
    if len(df) < 8:
        return np.nan
    r = df.rank().to_numpy()
    return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])
