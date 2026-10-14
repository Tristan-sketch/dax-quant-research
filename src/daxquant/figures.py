"""Report figures. One consistent, quiet style; the data is the only loud element.

Every figure is saved as vector PDF (for LaTeX) and PNG (for the website).
"""

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats as st  # noqa: E402

from . import config as C  # noqa: E402
from . import metrics  # noqa: E402

# Categorical slots in fixed order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
# Colour follows the entity in every chart.
ENTITY = {"DAX": SERIES[0], "SPX": SERIES[1], "EW": SERIES[2], "COMP_lo": "#4a3aa7", "EW_SURV": INK2}
NEUTRAL = "#f0efec"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#b8312f", "#e34948", "#f3b3b2", NEUTRAL, "#9ec5f4", "#2a78d6", "#104281"]
)

WIDTH = 6.3  # inches, LaTeX text width
SHORT = {"MOM": "Momentum\n12-1", "STR": "Short-term\nreversal", "LVOL": "Low\nvolatility",
         "LBETA": "Low\nbeta", "HIGH52": "52-week\nhigh", "COMP": "Composite"}
LABELS = {
    "DAX": "DAX (performance index)",
    "SPX": "S&P 500 TR in EUR",
    "EW": "Equal-weight DAX, point in time",
    "EW_SURV": "Equal-weight, today's members back-filled",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })


def _save(fig, name: str):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ts(m: pd.DataFrame | pd.Series):
    out = m.copy()
    out.index = out.index.to_timestamp(how="end")
    return out


def _wealth(r: pd.Series) -> pd.Series:
    w = metrics.wealth(r) * 100
    first = w.index[0] - (w.index[1] - w.index[0])
    return pd.concat([pd.Series([100.0], index=[first]), w])


def _pct(ax, axis="y", decimals=0):
    fmt = mt.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _log_wealth_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mt.LogLocator(base=2))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(mt.NullLocator())


def _end_dot(ax, s: pd.Series, color):
    ax.plot([s.index[-1]], [s.iloc[-1]], "o", ms=4.5, color=color, mec=SURFACE, mew=1.5, zorder=5)


def fig_index_growth(R):
    m = _ts(R.monthly[["DAX", "SPX", "EW"]])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 4.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    for col in ["DAX", "SPX", "EW"]:
        w = _wealth(R.monthly[col])
        w.index = [m.index[0] - pd.offsets.MonthEnd(1)] + list(m.index)
        a1.plot(w.index, w, color=ENTITY[col], label=f"{LABELS[col]}: {w.iloc[-1]:,.0f}")
        _end_dot(a1, w, ENTITY[col])
        a2.plot(m.index, metrics.drawdown(R.monthly[col]), color=ENTITY[col], lw=1.2)
    _log_wealth_axis(a1)
    a1.set_title("Growth of 100 EUR, total return, log scale")
    a1.legend(loc="upper left")
    a2.set_title("Drawdown from previous peak")
    _pct(a2)
    a1.margins(x=0.01)
    _save(fig, "index_growth")


def fig_rolling_risk(R):
    d = R.index_daily
    vol = d[["DAX", "SPX"]].rolling(252).std() * np.sqrt(252)
    m = R.monthly[["DAX", "SPX"]]
    corr = _ts(m["DAX"].rolling(36).corr(m["SPX"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"wspace": 0.25})
    a1.plot(vol.index, vol["DAX"], color=ENTITY["DAX"], lw=1.2, label="DAX")
    a1.plot(vol.index, vol["SPX"], color=ENTITY["SPX"], lw=1.2, label="S&P 500 (EUR)")
    a1.set_title("Rolling 1-year volatility")
    a1.legend(loc="upper right")
    _pct(a1)
    a2.plot(corr.index, corr, color=INK2, lw=1.4)
    a2.set_ylim(0, 1)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    a2.set_title("Rolling 3-year correlation, DAX vs S&P 500")
    _save(fig, "rolling_risk")


def fig_calendar(R):
    y = pd.DataFrame({c: metrics.calendar_returns(R.monthly[c]) for c in ["DAX", "SPX"]})
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    w = 0.38
    for i, col in enumerate(["DAX", "SPX"]):
        ax.bar(x + (i - 0.5) * w, y[col], width=w * 0.92, color=ENTITY[col],
               label=LABELS[col] if col == "SPX" else "DAX")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x, [str(v) if v % 2 == 0 else "" for v in y.index])
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title("Calendar-year total returns (latest year to date)")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, "calendar_returns")


def fig_monthly_heatmap(R):
    r = R.monthly["DAX"]
    tab = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.to_numpy()}).pivot(
        index="y", columns="m", values="r")
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    lim = 0.15
    ax.imshow(tab.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(v) > 0.09 else INK)
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                              "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(tab)), tab.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("DAX monthly returns in %")
    _save(fig, "dax_monthly_heatmap")


def fig_distribution(R):
    d = R.index_daily["DAX"].dropna()
    q = d.quantile(0.01)
    es = d[d <= q].mean()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    bins = np.linspace(-0.08, 0.08, 121)
    ax.hist(d.clip(-0.08, 0.08), bins=bins, density=True, color=SERIES[0], alpha=0.85,
            rwidth=0.85, label="DAX daily returns (clipped at ±8%)")
    x = np.linspace(-0.08, 0.08, 400)
    ax.plot(x, st.norm.pdf(x, d.mean(), d.std()), color=INK2, lw=1.2, label="Normal, same mean and vol")
    top = ax.get_ylim()[1]
    for v, lab, y, ha, dx in [(es, "ES 99%", 0.85, "right", -4), (q, "VaR 99%", 0.6, "left", 4)]:
        ax.axvline(v, color=SERIES[1], lw=1.2)
        ax.annotate(f"{lab}\n{v:.1%}", (v, top * y), xytext=(dx, 0),
                    textcoords="offset points", ha=ha, fontsize=7.5, color=INK)
    _pct(ax, "x")
    ax.set_yticks([])
    ax.set_title(f"Daily DAX returns: fat tails (excess kurtosis {st.kurtosis(d):.1f})")
    ax.legend(loc="upper right")
    _save(fig, "dax_distribution")


def fig_coverage(R):
    cv = R.coverage.resample("ME").last()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ax.plot(cv.index, cv["members"], color=MUTED, lw=1.4, label="DAX members")
    ax.plot(cv.index, cv["covered"], color=SERIES[0], lw=1.6, label="Members with usable price data")
    ax.fill_between(cv.index, cv["covered"], cv["members"], color=SERIES[1], alpha=0.18, lw=0,
                    label="Missing (delisted, no free data)")
    ax.set_ylim(0, 42)
    ax.set_title("Point-in-time universe and data coverage")
    ax.legend(loc="lower left", ncol=1)
    _save(fig, "coverage")


def fig_survivorship(R):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    idx = _ts(R.monthly).index
    for col in ["EW", "EW_SURV"]:
        w = _wealth(R.monthly[col])
        w.index = [idx[0] - pd.offsets.MonthEnd(1)] + list(idx)
        ax.plot(w.index, w, color=ENTITY[col], label=f"{LABELS[col]}: {w.iloc[-1]:,.0f}")
        _end_dot(ax, w, ENTITY[col])
    _log_wealth_axis(ax)
    s = R.survivorship
    ax.set_title(f"Survivorship bias: {s['cagr_survivor'] - s['cagr_pit']:+.1%} per year from using today's members")
    ax.legend(loc="upper left")
    _save(fig, "survivorship")


def fig_factor_grid(R):
    keys = list(C.STRATEGY_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    idx = _ts(R.monthly).index
    for ax, k in zip(axes.flat, keys):
        g = _wealth(R.gross_m[f"{k}_ls"].reindex(R.monthly.index))
        n = _wealth(R.monthly[f"{k}_ls"])
        for s in (g, n):
            s.index = [idx[0] - pd.offsets.MonthEnd(1)] + list(idx)
        ax.plot(g.index, g, color=AXIS, lw=1.1, label="before costs")
        ax.plot(n.index, n, color=SERIES[0], lw=1.5, label="after costs")
        ax.axhline(100, color=AXIS, lw=0.8)
        sr = R.summary.at[f"{k}_ls", "sharpe"]
        ax.set_title(f"{C.STRATEGY_NAMES[k]}\nnet Sharpe {sr:.2f}", fontsize=8.5)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    _log_wealth_axis(axes[0, 0])
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Long-short factor portfolios, growth of 100 (top minus bottom tercile)",
                 x=0.125, ha="left", fontsize=9, fontweight="bold", y=1.03)
    _save(fig, "factor_long_short")


def fig_buckets(R):
    keys = list(C.STRATEGY_NAMES)
    b = pd.DataFrame({k: R.buckets[k].mean() * 12 for k in keys}).T
    x = np.arange(len(keys))
    nb = b.shape[1]
    w = 0.8 / nb
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    names = ["Top tercile", "Middle", "Bottom tercile"] if nb == 3 else list(b.columns)
    ramp = [BLUE_RAMP[2], BLUE_RAMP[1], BLUE_RAMP[0]]
    for i in range(nb):
        ax.bar(x + (i - (nb - 1) / 2) * w, b.iloc[:, i], width=w * 0.9, color=ramp[i], label=names[i])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x, [SHORT[k] for k in keys])
    ax.grid(axis="x", visible=False)
    _pct(ax)
    ax.set_title("Average annual return by signal tercile, before costs")
    ax.legend(loc="upper right", ncol=3)
    _save(fig, "tercile_returns")


def fig_rolling_ic(R):
    keys = list(C.STRATEGY_NAMES)
    ic = R.ic.rolling(24).mean()
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    for ax, k in zip(axes.flat, keys):
        s = ic[k]
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.fill_between(s.index, 0, s, where=s >= 0, color=SERIES[0], alpha=0.12, lw=0)
        ax.fill_between(s.index, 0, s, where=s < 0, color=SERIES[1], alpha=0.12, lw=0)
        ax.plot(s.index, s, color=SERIES[0], lw=1.3)
        ax.set_title(f"{C.STRATEGY_NAMES[k]}\nmean IC {R.ic[k].mean():+.3f}", fontsize=8.5)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.suptitle("Rank information coefficient, 24-month rolling mean", x=0.125, ha="left",
                 fontsize=9, fontweight="bold", y=1.03)
    _save(fig, "rolling_ic")


def fig_correlations(R):
    cols = [f"{k}_ls" for k in C.STRATEGY_NAMES] + ["DAX"]
    c = R.monthly[cols].corr()
    labels = [C.STRATEGY_NAMES.get(x.replace("_ls", ""), x) for x in cols]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow(c.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(c.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(cols)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Correlation of monthly long-short returns")
    _save(fig, "factor_correlations")


def fig_composite(R):
    keys = [f"{C.COMPOSITE}_lo", "EW", "DAX"]
    labels = ["Composite, top tercile (long only, after costs)", LABELS["EW"], LABELS["DAX"]]
    idx = _ts(R.monthly).index
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 4.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    for k, lab in zip(keys, labels):
        w = _wealth(R.monthly[k])
        w.index = [idx[0] - pd.offsets.MonthEnd(1)] + list(idx)
        a1.plot(w.index, w, color=ENTITY[k], label=f"{lab}: {w.iloc[-1]:,.0f}")
        _end_dot(a1, w, ENTITY[k])
    _log_wealth_axis(a1)
    a1.legend(loc="upper left")
    a1.set_title("Long-only composite portfolio vs benchmarks, growth of 100")
    rel = metrics.wealth(R.monthly[keys[0]]) / metrics.wealth(R.monthly["DAX"]) - 1
    rel.index = idx
    a2.axhline(0, color=AXIS, lw=0.8)
    a2.plot(rel.index, rel, color=ENTITY[keys[0]], lw=1.4)
    a2.set_title("Cumulative outperformance of the composite vs DAX")
    _pct(a2)
    _save(fig, "composite_long_only")


def fig_costs(R):
    cc = R.cost_curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, kind, title in [(a1, "ls", "Long-short"), (a2, "lo", "Long only (excess of cash)")]:
        for i, k in enumerate(C.STRATEGY_NAMES):
            s = cc[cc.series == f"{k}_{kind}"]
            ax.plot(s.cost_bps, s.sharpe, color=SERIES[i], lw=1.4, marker="o", ms=3,
                    label=C.STRATEGY_NAMES[k])
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(C.BASE.cost_bps, color=MUTED, lw=0.8)
        ax.set_xlabel("One-way trading cost (bps)")
        ax.set_title(title)
    a1.set_ylabel("Sharpe ratio")
    handles, labs = a1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.2, columnspacing=1.0)
    fig.suptitle("Sensitivity of the Sharpe ratio to trading costs (grey line: base case)",
                 x=0.125, ha="left", fontsize=9, fontweight="bold", y=1.02)
    _save(fig, "cost_sensitivity")


def fig_robustness(R):
    rb = R.robustness.copy()
    order = [(lag, nb) for lag in C.ROBUST_LAGS for nb in C.ROBUST_BUCKETS]
    names = {o: f"lag {o[0]}d\n{'terciles' if o[1] == 3 else 'quintiles'}"
             + ("\n(base)" if o == (C.BASE.lag, C.BASE.buckets) else "") for o in order}
    rb["variant"] = [names[(r.lag, r.buckets)] for r in rb.itertuples()]
    p = rb.pivot(index="strategy", columns="variant", values="sharpe_net")
    p = p.reindex(index=list(C.STRATEGY_NAMES), columns=[names[o] for o in order])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ax.imshow(p.to_numpy(), cmap=DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto")
    for (i, j), v in np.ndenumerate(p.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 0.5 else INK)
    ax.set_xticks(range(p.shape[1]), p.columns, fontsize=7.5)
    ax.set_yticks(range(p.shape[0]), [C.STRATEGY_NAMES[k] for k in p.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Net Sharpe ratio of the long-short portfolio across specifications")
    _save(fig, "robustness")


def fig_stocks(R):
    s = R.stocks[R.stocks.years >= 1]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    for i, (flag, lab) in enumerate([(True, "Current member"), (False, "Former member")]):
        x = s[s.current == flag]
        ax.scatter(x.vol, x.ann_return, s=26, color=SERIES[i], edgecolor=SURFACE, linewidth=1.2,
                   label=lab, zorder=3)
    dax = R.summary.loc["DAX"]
    ax.scatter([dax.vol], [dax.cagr], marker="D", s=40, color=INK, edgecolor=SURFACE, zorder=4)
    ax.annotate("DAX", (dax.vol, dax.cagr), xytext=(6, -2), textcoords="offset points", fontsize=7.5)
    pick = pd.concat([s.nlargest(3, "ann_return"), s.nsmallest(3, "ann_return"), s.nlargest(1, "vol")])
    pick = pick[~pick.index.duplicated()]
    for t, r in pick.iterrows():
        ax.annotate(r["name"], (r.vol, r.ann_return), xytext=(5, 2), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    _pct(ax)
    _pct(ax, "x")
    ax.set_xlabel("Annualised volatility while in the DAX")
    ax.set_ylabel("Annualised return while in the DAX")
    ax.set_title("Constituents with at least one year in the DAX: risk and return while a member")
    ax.legend(loc="upper right")
    _save(fig, "stocks_risk_return")


ALL = [
    fig_index_growth, fig_rolling_risk, fig_calendar, fig_monthly_heatmap, fig_distribution,
    fig_coverage, fig_survivorship, fig_factor_grid, fig_buckets, fig_rolling_ic,
    fig_correlations, fig_composite, fig_costs, fig_robustness, fig_stocks,
]


def make_all(R):
    _style()
    for f in ALL:
        f(R)
    return [f.__name__.replace("fig_", "") for f in ALL]
