"""Write every number the paper uses into report/generated/.

The prose in report/main.tex never contains a hard-coded result. Numbers come in
as macros, tables as \\input files, and sentences whose meaning depends on the
result (significant or not, positive or negative) are generated here, so the
paper stays correct after every data update.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import universe

NAMES = {
    "DAX": "DAX", "SPX": r"S\&P 500 (EUR)", "EW": "Equal-weight PIT", "EW_SURV": "Equal-weight survivors",
    **{f"{k}_ls": v for k, v in C.STRATEGY_NAMES.items()},
    **{f"{k}_lo": v for k, v in C.STRATEGY_NAMES.items()},
}


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def num(x, d=2, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def pct(x, d=1, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return num(100 * x, d, sign) + r"\%"


def stars(p) -> str:
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""


def _table(cols: str, header: list[str], rows: list[list[str]], midrules: tuple = ()) -> str:
    out = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append(r"\midrule")
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_benchmarks(R) -> str:
    s = R.summary
    cols = ["DAX", "SPX", "EW", "EW_SURV"]
    spec = [
        ("CAGR", "cagr", pct), ("Volatility p.a.", "vol", pct),
        ("Sharpe ratio", "sharpe", num), ("\\quad standard error", "sharpe_se", num),
        ("Sortino ratio", "sortino", num), ("Maximum drawdown", "max_dd", pct),
        ("Longest drawdown (months)", "max_dd_months", lambda x: f"{x:.0f}"),
        ("Calmar ratio", "calmar", num), ("Skewness (monthly)", "skew", num),
        ("Excess kurtosis (monthly)", "kurt", num), ("VaR 95\\% (monthly)", "var95", pct),
        ("Expected shortfall 95\\%", "cvar95", pct), ("Positive months", "hit_rate", pct),
        ("Best month", "best", pct), ("Worst month", "worst", pct),
        ("Beta to DAX", "beta", num), ("Correlation with DAX", "corr", num),
    ]
    rows = [[lab] + [f(s.at[c, k]) if k in s and pd.notna(s.at[c, k]) else "--" for c in cols]
            for lab, k, f in spec]
    return _table("lrrrr", [""] + [NAMES[c] for c in cols], rows, midrules=(5, 8, 12, 15))


def tab_factor_tests(R) -> str:
    ft = R.factor_tests
    rows = []
    for k, r in ft.iterrows():
        rows.append([
            C.STRATEGY_NAMES[k], num(r.ic_mean, 3), num(r.ic_t), pct(r.ic_hit, 0), num(r.icir),
            pct(r.spread_gross), pct(r.ls_mean) + stars(r.ls_p), num(r.ls_t), num(r.ls_p_holm),
            num(r.psr), num(r.dsr), "yes" if r.monotonic else "no",
        ])
    head = ["Signal", "IC", "$t$(IC)", "IC$>$0", "ICIR", "Spread", "L/S net", "$t$", "Holm $p$",
            "PSR", "DSR", "Monot."]
    return _table("l" + "r" * 11, head, rows, midrules=(5,))


def _perf_rows(R, keys, extra):
    s = R.summary
    rows = []
    for k in keys:
        r = s.loc[k]
        base = [NAMES[k], pct(r.cagr), pct(r.vol), num(r.sharpe), pct(r.max_dd), num(r.beta),
                pct(r.alpha) + stars(r.alpha_p), num(r.alpha_t)]
        rows.append(base + [f(r) for f in extra])
    return rows


def tab_long_short(R) -> str:
    keys = [f"{k}_ls" for k in C.STRATEGY_NAMES]
    extra = [lambda r: num(r.turnover, 1), lambda r: num(r.breakeven_bps, 0)]
    head = ["Signal", "CAGR", "Vol.", "Sharpe", "Max DD", "Beta", "Alpha", "$t$", "Turnover", "Break-even"]
    return _table("l" + "r" * 9, head, _perf_rows(R, keys, extra), midrules=(5,))


def tab_long_only(R) -> str:
    keys = [f"{k}_lo" for k in C.STRATEGY_NAMES] + ["EW"]
    extra = [lambda r: pct(r.te), lambda r: num(r.ir), lambda r: pct(r.up_capture, 0),
             lambda r: pct(r.down_capture, 0), lambda r: num(r.turnover, 1)]
    head = ["Portfolio", "CAGR", "Vol.", "Sharpe", "Max DD", "Beta", "Alpha", "$t$", "TE", "IR",
            "Up", "Down", "Turnover"]
    return _table("l" + "r" * 12, head, _perf_rows(R, keys, extra), midrules=(5, 6))


def tab_subperiods(R) -> str:
    sp = R.subperiods
    order = ["DAX", "SPX", "EW"] + [f"{k}_ls" for k in C.STRATEGY_NAMES] + [f"{C.COMPOSITE}_lo"]
    periods = [p[2] for p in C.SUBPERIODS]
    piv = sp.pivot(index="series", columns="period", values="sharpe").reindex(order)[periods]
    cg = sp.pivot(index="series", columns="period", values="cagr").reindex(order)[periods]
    rows = []
    for k in order:
        name = NAMES[k] + (" (L/S)" if k.endswith("_ls") else " (long only)" if k.endswith("_lo") else "")
        rows.append([name] + [f"{num(piv.at[k, p])} / {pct(cg.at[k, p])}" for p in periods])
    head = ["Series"] + [esc(p) for p in periods]
    return _table("l" + "r" * len(periods), head, rows, midrules=(3, 9))


def tab_calendar(R) -> str:
    from . import metrics

    keys = ["DAX", "SPX", "EW", f"{C.COMPOSITE}_lo", "MOM_ls"]
    cal = pd.DataFrame({k: metrics.calendar_returns(R.monthly[k]) for k in keys})
    head = ["Year", "DAX", r"S\&P 500 (EUR)", "Equal-weight", "Composite (long)", "Momentum (L/S)"]
    rows = [[str(y)] + [pct(v) for v in r] for y, r in cal.iterrows()]
    return _table("lrrrrr", head, rows)


def tab_quality(R) -> str:
    log = R.clean_log
    rows = []
    for r in log.itertuples():
        rows.append([str(pd.Timestamp(r.date).date()), esc(R.securities.at[r.ticker, "name"]),
                     esc(r.action), pct(r.raw), pct(r.clean), esc(r.note)])
    return _table("lllrrp{5.2cm}", ["Date", "Stock", "Fix", "Raw", "Clean", "Reason"], rows)


def tab_missing(R) -> str:
    rows = []
    for r in R.missing.itertuples():
        end = "today" if pd.isna(r.end) else str(r.end.date())
        rows.append([esc(r.name), str(r.start.date()) if r.start.year > 2005 else "before 2005", end])
    return _table("lll", ["Company", "In DAX from", "Until"], rows)


def tab_stocks(R) -> str:
    s = R.stocks
    out = [r"\begin{longtable}{llrrrrrrr}", r"\toprule",
           r"Company & Sector & Years & Return & Vol. & Sharpe & Beta & Max DD & Worst day \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for t, r in s.iterrows():
        mark = "" if r.current else r"$^{\dagger}$"
        out.append(" & ".join([esc(r["name"]) + mark, esc(r.sector), num(r.years, 1), pct(r.ann_return),
                               pct(r.vol), num(r.sharpe), num(r.beta), pct(r.max_dd), pct(r.worst_day)])
                   + r" \\")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def tab_live(R) -> str:
    live = R.live
    if len(live) < 5:
        return (r"\emph{No out-of-sample data yet. The specification was frozen on "
                + C.FREEZE_DATE + r"; this table fills in automatically with every update.}" + "\n")
    keys = ["DAX", "SPX", "EW", f"{C.COMPOSITE}_lo"] + [f"{k}_ls" for k in C.STRATEGY_NAMES]
    rows = []
    for k in keys:
        r = live[k].dropna()
        cum = (1 + r).prod() - 1
        vol = r.std() * np.sqrt(252)
        rows.append([NAMES[k] + (" (L/S)" if k.endswith("_ls") else ""), pct(cum), pct(vol), str(len(r))])
    return _table("lrrr", ["Series", "Cumulative return", "Vol. p.a.", "Days"], rows, midrules=(4,))


# ---------------------------------------------------------------------------
def _sig_word(t: float) -> str:
    a = abs(t)
    if a >= 3:
        return "highly significant ($|t|\\geq 3$, which also clears the Harvey--Liu--Zhu hurdle)"
    if a >= 1.96:
        return "significant at the 5\\% level but below the $t=3$ hurdle for multiple testing"
    if a >= 1.65:
        return "only marginally significant (10\\% level)"
    return "not statistically different from zero"


def join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def findings(R) -> dict:
    ft = R.factor_tests
    s = R.summary
    best = ft["sharpe"].idxmax()
    worst = ft["sharpe"].idxmin()
    n_sig = int((ft["ls_p_holm"] < 0.05).sum())
    sig_alpha = [k for k in C.STRATEGY_NAMES if s.at[f"{k}_ls", "alpha_t"] >= 1.96]
    comp_lo = s.loc[f"{C.COMPOSITE}_lo"]
    surv = R.survivorship
    dsr_ok = [k for k in ft.index if ft.at[k, "dsr"] > 0.95]

    f = {}
    f["FindBest"] = (
        f"The strongest signal is \\emph{{{C.STRATEGY_NAMES[best]}}}, whose long-short portfolio earns "
        f"{pct(ft.at[best, 'ls_mean'])} per year after costs (Sharpe {num(ft.at[best, 'sharpe'])}); "
        f"its mean return is {_sig_word(ft.at[best, 'ls_t'])}."
    )
    f["FindWorst"] = (
        f"The weakest is \\emph{{{C.STRATEGY_NAMES[worst]}}} with a net Sharpe ratio of "
        f"{num(ft.at[worst, 'sharpe'])}"
        + (f"; it loses money mostly through trading costs, since its gross spread is only "
           f"{pct(ft.at[worst, 'spread_gross'])} per year at a turnover of "
           f"{num(s.at[f'{worst}_ls', 'turnover'], 1)} times the portfolio per year."
           if s.at[f"{worst}_ls", "turnover"] > 6 else ".")
    )
    f["FindHolm"] = (
        "After a Holm correction for testing six strategies, "
        + ("no long-short premium remains significant at the 5\\% level."
           if n_sig == 0 else
           f"{n_sig} of six long-short premia remain significant at the 5\\% level.")
    )
    f["FindDSR"] = (
        f"Across all {R.factor_tests.attrs['n_trials']} specifications that were run, an unskilled "
        f"researcher would expect a best Sharpe ratio of {num(R.factor_tests.attrs['sr_star_annual'])} by luck "
        "alone. "
        + ("No strategy has a deflated Sharpe ratio above 0.95, so none of them can be called a "
           "discovery on this data."
           if not dsr_ok else
           "Only " + join(C.STRATEGY_NAMES[k] for k in dsr_ok)
           + " clear a deflated Sharpe ratio of 0.95.")
    )
    betas = {k: s.at[f"{k}_ls", "beta"] for k in C.STRATEGY_NAMES}
    neg = [k for k, b in betas.items() if b < 0]
    f["FindBeta"] = (
        f"{['None', 'One', 'Two', 'Three', 'Four', 'Five', 'All six'][len(neg)]}"
        f"{'' if len(neg) == 6 else ' of six'} long-short portfolios carry a negative market beta (lowest: "
        f"{C.STRATEGY_NAMES[min(betas, key=betas.get)]} at {num(min(betas.values()))}), because the "
        "bottom tercile of these signals is dominated by volatile, high-beta stocks. In a rising market "
        "that drag hides part of the premium: "
        + (("after controlling for the DAX, the CAPM alpha is significant at 5\\% for "
            + join(C.STRATEGY_NAMES[k] for k in sig_alpha) + ".")
           if sig_alpha else "even after controlling for the DAX, no CAPM alpha is significant at 5\\%.")
    )
    f["FindComposite"] = (
        f"The long-only composite portfolio returns {pct(comp_lo.cagr)} per year against "
        f"{pct(s.at['DAX', 'cagr'])} for the DAX, with a beta of {num(comp_lo.beta)}, a tracking error of "
        f"{pct(comp_lo.te)} and an information ratio of {num(comp_lo.ir)}. Its CAPM alpha of "
        f"{pct(comp_lo.alpha)} per year has a $t$-statistic of {num(comp_lo.alpha_t)}."
    )
    f["FindSurvivorship"] = (
        f"Using today's members for the whole history raises the equal-weight return from "
        f"{pct(surv['cagr_pit'])} to {pct(surv['cagr_survivor'])} per year, a bias of "
        f"{pct(surv['cagr_survivor'] - surv['cagr_pit'])} per year that would be mistaken for skill."
    )
    f["FindSPX"] = (
        f"Measured in euros, the S\\&P 500 compounded at {pct(s.at['SPX', 'cagr'])} per year versus "
        f"{pct(s.at['DAX', 'cagr'])} for the DAX, with lower volatility ({pct(s.at['SPX', 'vol'])} vs "
        f"{pct(s.at['DAX', 'vol'])})"
        if s.at["SPX", "vol"] < s.at["DAX", "vol"] else
        f"Measured in euros, the S\\&P 500 compounded at {pct(s.at['SPX', 'cagr'])} per year versus "
        f"{pct(s.at['DAX', 'cagr'])} for the DAX, with volatility of {pct(s.at['SPX', 'vol'])} vs "
        f"{pct(s.at['DAX', 'vol'])}"
    )
    f["FindSPX"] += (
        f". The correlation of monthly returns is {num(s.at['SPX', 'corr'])} and the S\\&P 500's beta to "
        f"the DAX is {num(s.at['SPX', 'beta'])}, so US equities diversified a euro investor's DAX risk "
        "rather than replicating it."
    )
    d = [k for k in ("LVOL", "LBETA")]
    lo_vol = np.mean([s.at[f"{k}_lo", "vol"] for k in d])
    lo_sr = np.mean([s.at[f"{k}_lo", "sharpe"] for k in d])
    ls_sr = np.mean([s.at[f"{k}_ls", "sharpe"] for k in d])
    f["FindDefensive"] = (
        f"The defensive signals (low volatility and low beta) give long-only portfolios with an average "
        f"volatility of {pct(lo_vol)} against {pct(s.at['DAX', 'vol'])} for the DAX and an average Sharpe "
        f"ratio of {num(lo_sr)}, while their long-short spreads average a Sharpe ratio of {num(ls_sr)}"
        + (": in raw returns their benefit is lower risk rather than a dollar-neutral premium."
           if ls_sr < 0.5 * lo_sr else ".")
    )
    return f


def macros(R) -> str:
    m = R.meta
    s = R.summary
    vals = {
        "DataDate": m["last_price_date"],
        "FreezeDate": C.FREEZE_DATE,
        "FirstMonth": m["first_month"],
        "LastMonth": m["last_month"],
        "NMonths": str(m["n_months"]),
        "NFormations": str(m["n_formations"]),
        "AvgUniverse": num(m["avg_universe"], 1),
        "NTickers": str(m["n_tickers_used"]),
        "NMembersEver": str(R.intervals.ticker.nunique()),
        "NMissing": str(len(R.missing)),
        "CoverageMean": pct(m["coverage_mean"]),
        "NChanges": str(len(universe.load_changes())),
        "CostBps": num(C.BASE.cost_bps, 0),
        "BorrowBps": num(C.BASE.borrow_bps_pa, 0),
        "NTrials": str(R.factor_tests.attrs["n_trials"]),
        "SRStar": num(R.factor_tests.attrs["sr_star_annual"]),
        "DaxCagr": pct(s.at["DAX", "cagr"]),
        "DaxSharpe": num(s.at["DAX", "sharpe"]),
        "DaxMaxDD": pct(s.at["DAX", "max_dd"]),
        "SpxCagr": pct(s.at["SPX", "cagr"]),
        "SpxSharpe": num(s.at["SPX", "sharpe"]),
        "CompCagr": pct(s.at[f"{C.COMPOSITE}_lo", "cagr"]),
        "CompSharpe": num(s.at[f"{C.COMPOSITE}_lo", "sharpe"]),
        "SurvBias": pct(R.survivorship["cagr_survivor"] - R.survivorship["cagr_pit"]),
        "NBigMoves": str(len(R.big_moves)),
        "NFixes": str(len(R.clean_log)),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in findings(R).items()]
    return "\n".join(lines) + "\n"


def write_all(R):
    C.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "macros.tex": macros(R),
        "tab_benchmarks.tex": tab_benchmarks(R),
        "tab_factor_tests.tex": tab_factor_tests(R),
        "tab_long_short.tex": tab_long_short(R),
        "tab_long_only.tex": tab_long_only(R),
        "tab_subperiods.tex": tab_subperiods(R),
        "tab_calendar.tex": tab_calendar(R),
        "tab_quality.tex": tab_quality(R),
        "tab_missing.tex": tab_missing(R),
        "tab_stocks.tex": tab_stocks(R),
        "tab_live.tex": tab_live(R),
    }
    for name, body in files.items():
        (C.GENERATED_DIR / name).write_text(body, encoding="utf-8")
    return list(files)
