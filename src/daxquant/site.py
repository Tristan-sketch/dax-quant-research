"""Build the static website in public/ from the same generated LaTeX the paper uses."""

import re
import shutil

import jinja2
from markupsafe import Markup

from . import config as C
from . import latex


def tex_inline_to_html(s: str) -> str:
    s = s.replace(r"\textbackslash{}", "\\")
    s = re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\$\^\{(\*+|\\dagger)\}\$", lambda m: "<sup>" + ("†" if "dagger" in m.group(1) else m.group(1)) + "</sup>", s)
    s = s.replace("$-$", "−").replace(r"$|t|\geq 3$", "|t| ≥ 3").replace("$t$", "t")
    s = s.replace("$t=3$", "t = 3").replace("IC$>$0", "IC>0").replace("$p$", "p")
    s = s.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_").replace(r"\#", "#")
    s = s.replace("--", "–").replace("$", "")
    return s


def tabular_to_html(tex: str) -> str:
    """Convert the simple booktabs tabulars written by latex.py into an HTML table."""
    if r"\begin{tabular}" not in tex and r"\begin{longtable}" not in tex:
        return f'<p class="note">{tex_inline_to_html(tex.strip())}</p>'
    body = re.sub(r"\\begin\{(tabular|longtable)\}\{[^}]*\}", "", tex)
    body = re.sub(r"\\end\{(tabular|longtable)\}", "", body)
    for cmd in (r"\endhead", r"\endfoot", r"\bottomrule"):
        body = body.replace(cmd, "")
    header_done = False
    head, rows, sep_next = [], [], False
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line == r"\toprule":
            continue
        if line == r"\midrule":
            if header_done:
                sep_next = True
            header_done = True
            continue
        cells = [tex_inline_to_html(c.strip()) for c in line.rstrip("\\").rstrip().rstrip("\\").split(" & ")]
        if not header_done:
            head = cells
        else:
            rows.append((cells, sep_next))
            sep_next = False
    out = ['<div class="table-wrap"><table>', "<thead><tr>"]
    out += [f"<th>{h}</th>" for h in head]
    out += ["</tr></thead><tbody>"]
    for cells, sep in rows:
        cls = ' class="sep"' if sep else ""
        tds = "".join(
            f'<td class="wrap-cell">{c}</td>' if len(c) > 45 else f"<td>{c}</td>" for c in cells
        )
        out.append(f"<tr{cls}>{tds}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def build(R) -> list[str]:
    C.PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = C.PUBLIC_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    for png in C.FIGURES_DIR.glob("*.png"):
        shutil.copy2(png, fig_dir / png.name)
    shutil.copy2(C.SITE_TEMPLATE_DIR / "style.css", C.PUBLIC_DIR / "style.css")
    pdf = C.REPORT_DIR / "main.pdf"
    has_pdf = pdf.exists()
    if has_pdf:
        shutil.copy2(pdf, C.PUBLIC_DIR / "report.pdf")

    tables = {}
    for f in C.GENERATED_DIR.glob("tab_*.tex"):
        tables[f.stem] = tabular_to_html(f.read_text(encoding="utf-8"))

    s = R.summary
    ft = R.factor_tests
    best = ft["sharpe"].idxmax()
    tiles = [
        {"label": "DAX, CAGR", "value": f"{s.at['DAX', 'cagr']:.1%}",
         "note": f"Sharpe {s.at['DAX', 'sharpe']:.2f}, max drawdown {s.at['DAX', 'max_dd']:.0%}"},
        {"label": "S&P 500 in EUR, CAGR", "value": f"{s.at['SPX', 'cagr']:.1%}",
         "note": f"Sharpe {s.at['SPX', 'sharpe']:.2f}, correlation {s.at['SPX', 'corr']:.2f}"},
        {"label": f"Best factor: {C.STRATEGY_NAMES[best]}", "value": f"{ft.at[best, 'sharpe']:.2f}",
         "note": f"net Sharpe, long-short, DSR {ft.at[best, 'dsr']:.2f}"},
        {"label": "Long-only composite, CAGR", "value": f"{s.at[f'{C.COMPOSITE}_lo', 'cagr']:.1%}",
         "note": f"beta {s.at[f'{C.COMPOSITE}_lo', 'beta']:.2f}, IR {s.at[f'{C.COMPOSITE}_lo', 'ir']:.2f}"},
        {"label": "Survivorship bias", "value": f"{R.survivorship['cagr_survivor'] - R.survivorship['cagr_pit']:+.1%}",
         "note": "per year, from back-filling today's members"},
    ]
    fnd = latex.findings(R)
    findings = [tex_inline_to_html(fnd[k]) for k in
                ("FindBest", "FindWorst", "FindHolm", "FindDSR", "FindBeta", "FindDefensive",
                 "FindComposite", "FindSurvivorship", "FindSPX")]

    def fig(name, caption, cls=""):
        c = f' class="{cls}"' if cls else ""
        return Markup(f'<figure{c}><img src="figures/{name}.png" alt="{caption}" loading="lazy">'
                      f"<figcaption>{caption}</figcaption></figure>")

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(C.SITE_TEMPLATE_DIR), autoescape=True)
    html = env.get_template("index.html.j2").render(
        meta=R.meta, tiles=tiles, findings=findings, tables=tables, fig=fig, has_pdf=has_pdf,
        cost_bps=f"{C.BASE.cost_bps:.0f}", borrow_bps=f"{C.BASE.borrow_bps_pa:.0f}",
        n_trials=ft.attrs["n_trials"], sr_star=f"{ft.attrs['sr_star_annual']:.2f}",
        freeze_date=C.FREEZE_DATE, rf_series=C.RISK_FREE_FRED,
    )
    (C.PUBLIC_DIR / "index.html").write_text(html, encoding="utf-8")
    (C.PUBLIC_DIR / ".nojekyll").write_text("")
    return ["index.html", "style.css"] + (["report.pdf"] if has_pdf else [])
