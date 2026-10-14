"""The full study: universe, strategies, benchmarks, statistics."""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import backtest as bt
from . import config as C
from . import data, metrics, quality, signals, stats, universe


@dataclass
class Results:
    meta: dict
    securities: pd.DataFrame
    intervals: pd.DataFrame
    clean_log: pd.DataFrame
    big_moves: pd.DataFrame
    coverage: pd.DataFrame  # daily: members, covered
    missing: pd.DataFrame  # members without data
    rf_m: pd.Series
    daily: pd.DataFrame  # daily net returns of everything
    monthly: pd.DataFrame  # monthly returns of everything (complete months)
    turnover_m: pd.DataFrame
    gross_m: pd.DataFrame
    ic: pd.DataFrame  # formation date x strategy
    buckets: dict  # strategy -> DataFrame of bucket period returns
    summary: pd.DataFrame  # row per series
    factor_tests: pd.DataFrame
    robustness: pd.DataFrame
    cost_curve: pd.DataFrame
    subperiods: pd.DataFrame
    live: pd.DataFrame
    stocks: pd.DataFrame
    index_daily: pd.DataFrame
    survivorship: dict
    holdings: dict = field(default_factory=dict)


def _riskfree(rf_pct: pd.Series, days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    months = pd.period_range(days[0].to_period("M"), days[-1].to_period("M"), freq="M")
    s = rf_pct.copy()
    s.index = s.index.to_period("M")
    s = s.groupby(level=0).last().reindex(months).ffill().bfill()
    rf_m = (1 + s / 100) ** (1 / 12) - 1
    rf_d = pd.Series(((1 + s / 100) ** (1 / 252) - 1).reindex(days.to_period("M")).to_numpy(), index=days)
    return rf_m, rf_d


def _complete_months(m: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    if last_day + pd.offsets.BMonthEnd(0) != last_day:
        m = m[m.index < last_day.to_period("M")]
    return m


def _scores(sig: pd.DataFrame, eligible: list[str], spec: C.Spec) -> pd.DataFrame:
    s = sig.loc[sig.index.intersection(eligible)]
    out = s[list(C.FACTORS)].copy()
    out[C.COMPOSITE] = signals.composite(s, spec)
    return out


def build_scores(ret, dax_ret, member_f, formations, spec):
    sigs = signals.compute_signals(ret, dax_ret, formations, spec)
    scores = {}
    for t, sig in sigs.items():
        members = member_f.columns[member_f.loc[t]]
        traded = ret.columns[ret.loc[t].notna()]
        eligible = sorted(set(members) & set(traded) & set(sig.index))
        if len(eligible) >= 15:
            scores[t] = _scores(sig, eligible, spec)
    return scores, sigs


def legs(scores, key, n_buckets):
    longs, shorts = {}, {}
    for t, sc in scores.items():
        top, bottom = bt.pick_buckets(sc[key], n_buckets)
        longs[t] = bt.equal_weights(top)
        shorts[t] = bt.equal_weights(bottom)
    return longs, shorts


def run_strategy(ret, picks, exec_dates, spec):
    longs, shorts = picks
    L = bt.simulate(ret, longs, exec_dates, spec.cost_bps)
    S = bt.simulate(ret, shorts, exec_dates, spec.cost_bps)
    ls = bt.long_short(L, S, spec.borrow_bps_pa)
    return L, S, ls


def run_universe_ew(ret, member_f, formations, exec_dates, spec, names=None):
    targets = {}
    for t in formations:
        pool = names if names is not None else member_f.columns[member_f.loc[t]]
        traded = [n for n in pool if n in ret.columns and pd.notna(ret.at[t, n])]
        targets[t] = bt.equal_weights(sorted(traded))
    return bt.simulate(ret, targets, exec_dates, spec.cost_bps)


def bucket_returns(scores, fwd, key, n_buckets):
    rows = {}
    for t, sc in scores.items():
        if t not in fwd:
            continue
        s = sc[key].dropna().sort_index().sort_values(ascending=False, kind="mergesort")
        groups = np.array_split(s.index.to_numpy(), n_buckets)
        rows[t] = [fwd[t].reindex(g).mean() for g in groups]
    cols = [f"B{i + 1}" for i in range(n_buckets)]  # B1 = highest score
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def run(meta_override: dict | None = None) -> Results:
    prices, bench, rf_pct, meta = data.load_cached()
    if meta_override:
        meta.update(meta_override)
    sec = data.load_securities()
    ret, clean_log = quality.clean_returns(prices, bench)
    days = ret.index
    last_day = days[-1]
    dax_ret = bench["DAX"].reindex(days).pct_change()
    spx_eur = (bench["SPX_USD"] / bench["EURUSD"]).ffill().reindex(days).ffill()
    spx_ret = spx_eur.pct_change()
    rf_m, rf_d = _riskfree(rf_pct, days)

    intervals = universe.membership_intervals()
    member_d = universe.membership_matrix(days, intervals)
    for t in member_d.columns:
        if t not in ret.columns:
            ret[t] = np.nan
    ret = ret[sorted(ret.columns)]

    # Coverage of the point-in-time index by usable price data
    has_px = ret.reindex(columns=member_d.columns).notna().rolling(5, min_periods=1).max().astype(bool)
    coverage = pd.DataFrame(
        {"members": member_d.sum(axis=1), "covered": (member_d & has_px).sum(axis=1)}, index=days
    )
    coverage = coverage[coverage.index >= universe.HISTORY_START]
    missing = []
    for row in intervals.itertuples():
        end = row.end if pd.notna(row.end) else last_day
        cov = ret.loc[row.start : end, row.ticker].notna().mean() if row.ticker in ret else 0.0
        if cov < 0.5:
            missing.append({"ticker": row.ticker, "name": sec.at[row.ticker, "name"],
                            "start": row.start, "end": row.end, "coverage": cov})
    missing = pd.DataFrame(missing)

    formations = bt.formation_dates(days)
    spec = C.BASE
    exec_dates = bt.execution_dates(days, formations, spec.lag)
    member_f = member_d.loc[formations]
    scores, sigs = build_scores(ret, dax_ret, member_f, formations, spec)
    exec_dates = exec_dates[exec_dates.index.isin(list(scores))]
    fwd = bt.forward_returns(ret, exec_dates)

    daily = {}
    turnover = {}
    gross = {}
    holdings = {}
    picks = {(k, nb): legs(scores, k, nb) for k in C.STRATEGY_NAMES for nb in C.ROBUST_BUCKETS}
    for key in C.STRATEGY_NAMES:
        L, S, ls = run_strategy(ret, picks[(key, spec.buckets)], exec_dates, spec)
        daily[f"{key}_ls"] = ls["net"]
        turnover[f"{key}_ls"] = ls["turnover"]
        gross[f"{key}_ls"] = ls["gross"]
        daily[f"{key}_lo"] = L.daily["net"]
        turnover[f"{key}_lo"] = L.daily["turnover"]
        gross[f"{key}_lo"] = L.daily["gross"]
        holdings[key] = {"long": L.weights, "short": S.weights}

    ew = run_universe_ew(ret, member_f, formations, bt.execution_dates(days, formations, spec.lag), spec)
    daily["EW"] = ew.daily["net"]
    turnover["EW"] = ew.daily["turnover"]
    gross["EW"] = ew.daily["gross"]
    survivor_names = universe.load_current()
    ew_surv = run_universe_ew(ret, member_f, formations, bt.execution_dates(days, formations, spec.lag),
                              spec, names=survivor_names)
    daily["EW_SURV"] = ew_surv.daily["net"]
    daily["DAX"] = dax_ret
    daily["SPX"] = spx_ret

    daily = pd.DataFrame(daily)
    start = exec_dates.iloc[0]
    daily = daily[daily.index > start]
    monthly = _complete_months(daily.apply(bt.to_monthly), last_day)
    turnover_m = _complete_months(
        pd.DataFrame(turnover).loc[lambda d: d.index >= start]
        .groupby(lambda d: d.to_period("M")).sum(), last_day)
    gross_m = _complete_months(
        pd.DataFrame(gross).loc[lambda d: d.index > start].apply(bt.to_monthly), last_day)

    # ---- summary statistics -------------------------------------------------
    rows = {}
    for col in monthly.columns:
        sf = col.endswith("_ls")
        rows[col] = metrics.summary(
            monthly[col], rf_m, bench=None if col == "DAX" else monthly["DAX"], self_financing=sf,
            turnover=turnover_m.get(col), gross=gross_m.get(col),
        )
    summary = pd.DataFrame(rows).T

    # ---- information coefficients and buckets --------------------------------
    ic = pd.DataFrame(
        {k: {t: stats.rank_ic(sc[k], fwd.get(t, pd.Series(dtype=float))) for t, sc in scores.items()}
         for k in C.STRATEGY_NAMES}
    )
    ic = ic[ic.index.isin(list(fwd))]
    if last_day + pd.offsets.BMonthEnd(0) != last_day and len(ic):
        ic = ic.iloc[:-1]  # last holding period still open
    buckets = {k: bucket_returns(scores, fwd, k, spec.buckets) for k in C.STRATEGY_NAMES}
    buckets = {k: v.loc[v.index.isin(ic.index)] for k, v in buckets.items()}

    # ---- robustness grid -------------------------------------------------------
    rob = []
    for lag in C.ROBUST_LAGS:
        ex_l = bt.execution_dates(days, pd.DatetimeIndex(list(scores)), lag)
        for nb in C.ROBUST_BUCKETS:
            sp = replace(spec, lag=lag, buckets=nb)
            for key in C.STRATEGY_NAMES:
                _, _, ls = run_strategy(ret, picks[(key, nb)], ex_l, sp)
                m = _complete_months(bt.to_monthly(ls["net"][ls.index > start]).to_frame("r"), last_day)["r"]
                g = _complete_months(bt.to_monthly(ls["gross"][ls.index > start]).to_frame("r"), last_day)["r"]
                rob.append({"strategy": key, "lag": lag, "buckets": nb,
                            "sharpe_net": stats.sharpe_per_period(m) * np.sqrt(12),
                            "sharpe_gross": stats.sharpe_per_period(g) * np.sqrt(12),
                            "mean_net": m.mean() * 12, "sr_pp": stats.sharpe_per_period(m)})
    robustness = pd.DataFrame(rob)

    # ---- factor tests (base spec) ---------------------------------------------
    trial_srs = robustness["sr_pp"].to_numpy()
    sr_star = stats.expected_max_sharpe(trial_srs)
    ft = []
    for key in C.STRATEGY_NAMES:
        x = monthly[f"{key}_ls"]
        t = stats.nw_mean_test(x)
        ict = stats.nw_mean_test(ic[key])
        b = buckets[key].mean() * 12
        ft.append({
            "strategy": key,
            "ic_mean": ic[key].mean(), "ic_t": ict["t"], "ic_hit": (ic[key] > 0).mean(),
            "icir": ic[key].mean() / ic[key].std() * np.sqrt(12),
            "ls_mean": t["mean"] * 12, "ls_t": t["t"], "ls_p": t["p"],
            "sharpe": summary.at[f"{key}_ls", "sharpe"],
            "psr": stats.probabilistic_sharpe(x),
            "dsr": stats.probabilistic_sharpe(x, sr_star),
            "spread_gross": b.iloc[0] - b.iloc[-1],
            "monotonic": bool(np.all(np.diff(b.to_numpy()) <= 0)),
            "alpha_lo": summary.at[f"{key}_lo", "alpha"], "alpha_lo_t": summary.at[f"{key}_lo", "alpha_t"],
        })
    factor_tests = pd.DataFrame(ft).set_index("strategy")
    factor_tests["ls_p_holm"] = stats.holm(factor_tests["ls_p"])
    factor_tests.attrs["sr_star_annual"] = sr_star * np.sqrt(12)
    factor_tests.attrs["n_trials"] = len(trial_srs)

    # ---- cost sensitivity -------------------------------------------------------
    cc = []
    borrow_m = spec.borrow_bps_pa / 1e4 / 12
    for key in C.STRATEGY_NAMES:
        for kind in ("ls", "lo"):
            col = f"{key}_{kind}"
            g = gross_m[col]
            tv = turnover_m[col].reindex(g.index).fillna(0)
            for c in C.COST_GRID_BPS:
                net = g - tv * c / 1e4 - (borrow_m if kind == "ls" else 0)
                ex = net if kind == "ls" else net - rf_m.reindex(net.index)
                cc.append({"series": col, "cost_bps": c, "sharpe": stats.sharpe_per_period(ex) * np.sqrt(12)})
    cost_curve = pd.DataFrame(cc)

    # ---- subperiods and live record ------------------------------------------------
    sub = []
    for a, b, label in C.SUBPERIODS:
        sl = monthly.loc[pd.Period(a, "M") : (pd.Period(b, "M") if b else None)]
        for col in ["DAX", "SPX", "EW"] + [f"{k}_ls" for k in C.STRATEGY_NAMES] + [f"{C.COMPOSITE}_lo"]:
            s = metrics.summary(sl[col], rf_m, self_financing=col.endswith("_ls"))
            sub.append({"period": label, "series": col, "cagr": s["cagr"], "sharpe": s["sharpe"],
                        "max_dd": s["max_dd"], "months": s["months"]})
    subperiods = pd.DataFrame(sub)
    freeze = pd.Timestamp(C.FREEZE_DATE)
    live = daily[daily.index > freeze]

    # ---- single stocks -------------------------------------------------------------
    stocks = _stock_table(ret, dax_ret, rf_d, intervals, sec, member_d)

    # ---- index-level daily data -----------------------------------------------------
    index_daily = pd.DataFrame({"DAX": dax_ret, "SPX": spx_ret, "rf": rf_d}).loc[start:]

    surv = {
        "cagr_pit": summary.at["EW", "cagr"],
        "cagr_survivor": summary.at["EW_SURV", "cagr"],
        "sharpe_pit": summary.at["EW", "sharpe"],
        "sharpe_survivor": summary.at["EW_SURV", "sharpe"],
    }
    meta.update({
        "first_month": str(monthly.index[0]),
        "last_month": str(monthly.index[-1]),
        "n_months": len(monthly),
        "n_formations": len(scores),
        "avg_universe": float(np.mean([len(s) for s in scores.values()])),
        "freeze_date": C.FREEZE_DATE,
        "n_tickers_used": int(ret.notna().any().sum()),
        "coverage_mean": float((coverage.covered / coverage.members).mean()),
    })
    return Results(
        meta=meta, securities=sec, intervals=intervals, clean_log=clean_log,
        big_moves=quality.large_moves(ret.loc[start:]), coverage=coverage, missing=missing,
        rf_m=rf_m, daily=daily, monthly=monthly, turnover_m=turnover_m, gross_m=gross_m, ic=ic,
        buckets=buckets, summary=summary, factor_tests=factor_tests, robustness=robustness,
        cost_curve=cost_curve, subperiods=subperiods, live=live, stocks=stocks,
        index_daily=index_daily, survivorship=surv, holdings=holdings,
    )


def _stock_table(ret, dax_ret, rf_d, intervals, sec, member_d) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(C.FIRST_FORMATION)
    for t in member_d.columns:
        if t not in ret or ret[t].notna().sum() < 60:
            continue
        mask = member_d[t] & ret[t].notna() & (ret.index > start)
        r = ret[t][mask]
        if len(r) < 60:
            continue
        spells = intervals[intervals.ticker == t]
        ex = r - rf_d[mask]
        m = dax_ret[mask]
        growth = (1 + r).cumprod()
        rows.append({
            "ticker": t, "name": sec.at[t, "name"], "sector": sec.at[t, "sector"],
            "spells": len(spells),
            "first_in": max(spells.start.min(), start).date(),
            "last_out": spells.end.max().date() if spells.end.notna().all() else None,
            "years": len(r) / 252,
            "ann_return": growth.iloc[-1] ** (252 / len(r)) - 1,
            "vol": r.std() * np.sqrt(252),
            "sharpe": ex.mean() / ex.std() * np.sqrt(252),
            "beta": r.cov(m) / m.var(),
            "max_dd": (growth / growth.cummax() - 1).min(),
            "worst_day": r.min(),
            "current": bool(member_d[t].iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("ann_return", ascending=False)
