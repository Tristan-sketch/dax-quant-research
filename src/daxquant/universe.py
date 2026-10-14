"""Point-in-time DAX membership.

The current constituent list is rolled backwards through the documented index
changes. A change is effective at the open of `effective_date`, so a stock that
is added on day d is a member on d and a stock removed on d is not.
"""

import pandas as pd

from . import config as C

HISTORY_START = pd.Timestamp("2005-01-01")


def load_changes() -> pd.DataFrame:
    ch = pd.read_csv(C.REFERENCE_DIR / "dax_changes.csv", parse_dates=["effective_date"])
    return ch.sort_values("effective_date").reset_index(drop=True)


def load_current() -> list[str]:
    return pd.read_csv(C.REFERENCE_DIR / "dax_members_current.csv")["ticker"].tolist()


def membership_intervals() -> pd.DataFrame:
    """One row per (ticker, spell) with start inclusive and end exclusive (NaT = still in)."""
    changes = load_changes()
    members = set(load_current())
    open_spell = {t: pd.NaT for t in members}  # ticker -> end of the spell being built
    spells = []
    for row in changes.iloc[::-1].itertuples():
        d = row.effective_date
        added = row.added if isinstance(row.added, str) else None
        removed = row.removed if isinstance(row.removed, str) else None
        if added:
            if added not in members:
                raise ValueError(f"{added} added on {d.date()} but not a member afterwards")
            spells.append((added, d, open_spell.pop(added)))
            members.remove(added)
        if removed:
            if removed in members:
                raise ValueError(f"{removed} removed on {d.date()} but still a member before")
            members.add(removed)
            open_spell[removed] = d
    for t, end in open_spell.items():
        spells.append((t, HISTORY_START, end))
    out = pd.DataFrame(spells, columns=["ticker", "start", "end"])
    return out.sort_values(["ticker", "start"]).reset_index(drop=True)


def membership_matrix(dates: pd.DatetimeIndex, intervals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Boolean frame dates x tickers: True when the stock is in the DAX on that date."""
    iv = membership_intervals() if intervals is None else intervals
    tickers = sorted(iv.ticker.unique())
    m = pd.DataFrame(False, index=dates, columns=tickers)
    for row in iv.itertuples():
        end = row.end if pd.notna(row.end) else dates.max() + pd.Timedelta(days=1)
        m.loc[(dates >= row.start) & (dates < end), row.ticker] = True
    return m
