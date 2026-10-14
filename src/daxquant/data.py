"""Download and cache prices, benchmarks and the risk-free rate."""

import io
import json
import time
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config as C


def load_securities() -> pd.DataFrame:
    sec = pd.read_csv(C.REFERENCE_DIR / "securities.csv", keep_default_na=False)
    sec["yahoo"] = sec["yahoo"].replace("", None)
    return sec.set_index("ticker")


def _yahoo_close(symbols: list[str], start: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                symbols, start=start, auto_adjust=True, progress=False, threads=False
            )
        if raw is not None and not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(symbols[0])
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            missing = [s for s in symbols if s not in close or close[s].notna().sum() == 0]
            if not missing or attempt == retries - 1:
                return close
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Yahoo download failed for {symbols}")


def download_prices() -> pd.DataFrame:
    sec = load_securities()
    sources = sec["yahoo"].dropna()
    close = _yahoo_close(sorted(set(sources)), C.PRICE_START)
    out = pd.DataFrame(
        {t: close[y] for t, y in sources.items() if y in close and close[y].notna().any()}
    )
    return out.sort_index()


def download_benchmarks() -> pd.DataFrame:
    symbols = [C.DAX_SYMBOL, C.SPX_TR_SYMBOL, C.EURUSD_SYMBOL]
    close = _yahoo_close(symbols, C.PRICE_START)
    out = pd.DataFrame(
        {
            "DAX": close[C.DAX_SYMBOL],
            "SPX_USD": close[C.SPX_TR_SYMBOL],
            "EURUSD": close[C.EURUSD_SYMBOL],
        }
    )
    return out.sort_index()


def download_riskfree() -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={C.RISK_FREE_FRED}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
    s = pd.to_numeric(df[C.RISK_FREE_FRED], errors="coerce")
    s.index = df["observation_date"]
    return s.dropna().rename("rf_pct")


def refresh() -> dict:
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices = download_prices()
    bench = download_benchmarks()
    rf = download_riskfree()
    prices.to_parquet(C.PRICES_FILE)
    bench.to_parquet(C.BENCHMARKS_FILE)
    rf.to_frame().to_parquet(C.RISKFREE_FILE)
    meta = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_price_date": str(bench["DAX"].dropna().index.max().date()),
        "n_price_series": int(prices.shape[1]),
        "last_riskfree_month": str(rf.index.max().date()),
    }
    C.META_FILE.write_text(json.dumps(meta, indent=2))
    return meta


def load_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    if not C.PRICES_FILE.exists():
        raise FileNotFoundError("No cached data. Run `daxquant data` first.")
    prices = pd.read_parquet(C.PRICES_FILE)
    bench = pd.read_parquet(C.BENCHMARKS_FILE)
    rf = pd.read_parquet(C.RISKFREE_FILE)["rf_pct"]
    meta = json.loads(C.META_FILE.read_text())
    return prices, bench, rf, meta
