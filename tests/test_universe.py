import pandas as pd

from daxquant import universe


def test_index_size_is_30_then_40():
    days = pd.bdate_range("2005-01-03", "2026-10-14")
    counts = universe.membership_matrix(days).sum(axis=1)
    assert (counts[days < "2021-09-20"] == 30).all()
    assert (counts[days >= "2021-09-20"] == 40).all()


def test_2005_composition_matches_history():
    days = pd.DatetimeIndex(["2005-01-03"])
    members = set(universe.membership_matrix(days).columns[universe.membership_matrix(days).iloc[0]])
    for t in ["SCH.DE", "ALT.DE", "HVM.DE", "TUI1.DE", "MAN.DE", "LIN.DE", "CEC.DE", "VOW.DE"]:
        assert t in members
    for t in ["MRK.DE", "VNA.DE", "ZAL.DE", "RHM.DE"]:
        assert t not in members


def test_change_is_effective_on_its_date():
    days = pd.DatetimeIndex(["2020-08-21", "2020-08-24"])
    m = universe.membership_matrix(days)
    assert m.at[days[0], "WDI.DE"] and not m.at[days[1], "WDI.DE"]
    assert not m.at[days[0], "DHER.DE"] and m.at[days[1], "DHER.DE"]


def test_spells_for_reentering_stock():
    iv = universe.membership_intervals()
    bei = iv[iv.ticker == "BEI.DE"].sort_values("start")
    assert len(bei) == 3  # 2008-2021, 2021-2022, 2022-today
    assert pd.isna(bei.end.iloc[-1])
