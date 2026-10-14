"""Research specification.

Everything in this file was fixed on FREEZE_DATE, before any strategy result was
looked at. Factor definitions follow the cited papers; no parameter was tuned on
the data. Changing anything here after the freeze invalidates the live
out-of-sample record, so treat it as read-only.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "data" / "reference"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
FIGURES_DIR = REPORT_DIR / "figures"
SITE_TEMPLATE_DIR = ROOT / "site"
PUBLIC_DIR = ROOT / "public"

FREEZE_DATE = "2026-10-14"
PRICE_START = "2004-06-01"
FIRST_FORMATION = "2006-01-31"

DAX_SYMBOL = "^GDAXI"  # DAX performance (total return) index
SPX_TR_SYMBOL = "^SP500TR"  # S&P 500 total return index, USD
EURUSD_SYMBOL = "EURUSD=X"  # USD per EUR
RISK_FREE_FRED = "IR3TIB01DEM156N"  # Germany 3-month interbank rate, % p.a.

FACTORS = {
    "MOM": "Momentum 12-1",
    "STR": "Short-term reversal",
    "LVOL": "Low volatility",
    "LBETA": "Low beta",
    "HIGH52": "52-week high",
}
COMPOSITE = "COMP"
STRATEGY_NAMES = {**FACTORS, COMPOSITE: "Composite"}


@dataclass(frozen=True)
class Spec:
    lookback: int = 252  # trading days for momentum, volatility, beta, 52w high
    skip: int = 21  # momentum skips the most recent month (Jegadeesh & Titman 1993)
    reversal_window: int = 21
    min_obs: int = 200  # valid daily returns required inside the lookback window
    beta_shrink: float = 0.6  # Vasicek-style shrinkage towards 1 (Frazzini & Pedersen 2014)
    winsor_z: float = 3.0
    buckets: int = 3  # terciles: about 10 stocks per leg with 30 members, 13 with 40
    lag: int = 1  # trade one day after the signal date
    cost_bps: float = 10.0  # one-way cost per unit of turnover
    borrow_bps_pa: float = 25.0  # securities lending fee on the short leg, DAX large caps
    composite: tuple = ("MOM", "STR", "LVOL", "LBETA", "HIGH52")


BASE = Spec()

# Robustness grid. Every cell counts as a trial for the deflated Sharpe ratio.
ROBUST_LAGS = (0, 1, 5)
ROBUST_BUCKETS = (3, 5)
COST_GRID_BPS = (0, 5, 10, 15, 20, 30, 40, 50)

SUBPERIODS = [
    ("2006-02", "2012-12", "Crisis years 2006-2012"),
    ("2013-01", "2019-12", "Low-rate era 2013-2019"),
    ("2020-01", None, "2020 to date"),
]


PRICES_FILE = CACHE_DIR / "prices.parquet"
BENCHMARKS_FILE = CACHE_DIR / "benchmarks.parquet"
RISKFREE_FILE = CACHE_DIR / "riskfree.parquet"
META_FILE = CACHE_DIR / "meta.json"
