"""Command line: `daxquant all` refreshes data and rebuilds results, paper and website."""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time

from . import config as C

PICKLE = C.RESULTS_DIR / "results.pkl"


def _step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_data(_):
    from . import data

    _step("downloading prices, benchmarks and risk-free rate")
    print(json.dumps(data.refresh(), indent=2))


def cmd_run(_):
    from . import analysis

    _step("running analysis")
    R = analysis.run()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(PICKLE, "wb") as fh:
        pickle.dump(R, fh)
    R.summary.to_csv(C.RESULTS_DIR / "summary.csv", float_format="%.6f")
    R.factor_tests.to_csv(C.RESULTS_DIR / "factor_tests.csv", float_format="%.6f")
    R.monthly.to_csv(C.RESULTS_DIR / "monthly_returns.csv", float_format="%.8f")
    R.robustness.to_csv(C.RESULTS_DIR / "robustness.csv", index=False, float_format="%.6f")
    R.stocks.to_csv(C.RESULTS_DIR / "stocks.csv", float_format="%.6f")
    R.clean_log.to_csv(C.RESULTS_DIR / "data_quality_log.csv", index=False)
    (C.RESULTS_DIR / "meta.json").write_text(json.dumps(R.meta, indent=2))
    return R


def _load():
    if not PICKLE.exists():
        return cmd_run(None)
    with open(PICKLE, "rb") as fh:
        return pickle.load(fh)


def compile_pdf() -> bool:
    for tool, args in [("latexmk", ["-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
                       ("tectonic", ["main.tex"])]:
        if shutil.which(tool):
            _step(f"compiling the paper with {tool}")
            res = subprocess.run([tool, *args], cwd=C.REPORT_DIR)
            return res.returncode == 0
    _step("no LaTeX engine found (latexmk or tectonic); skipping PDF. CI builds it.")
    return False


def cmd_report(args):
    from . import figures, latex

    R = _load()
    _step("drawing figures")
    figures.make_all(R)
    _step("writing LaTeX tables and macros")
    latex.write_all(R)
    if not getattr(args, "no_pdf", False):
        compile_pdf()
    return R


def cmd_site(_):
    from . import site

    R = _load()
    _step("building website")
    site.build(R)
    print(f"open {C.PUBLIC_DIR / 'index.html'}")


def cmd_all(args):
    cmd_data(args)
    cmd_run(args)
    cmd_report(args)
    cmd_site(args)


def cmd_serve(args):
    import functools
    import http.server
    import webbrowser

    if not (C.PUBLIC_DIR / "index.html").exists():
        cmd_site(args)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # keep the console window readable
            pass

    handler = functools.partial(Quiet, directory=str(C.PUBLIC_DIR))
    for port in range(args.port, args.port + 20):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    else:
        sys.exit(f"no free port between {args.port} and {args.port + 19}")
    url = f"http://localhost:{port}/"
    _step(f"website running at {url}  (close this window or press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="daxquant", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download fresh data")
    sub.add_parser("run", help="run the analysis on cached data")
    r = sub.add_parser("report", help="figures, LaTeX inputs and PDF")
    r.add_argument("--no-pdf", action="store_true")
    sub.add_parser("site", help="build the website in public/")
    a = sub.add_parser("all", help="data + run + report + site")
    a.add_argument("--no-pdf", action="store_true")
    s = sub.add_parser("serve", help="open the website in the browser")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    commands = {"data": cmd_data, "run": cmd_run, "report": cmd_report, "site": cmd_site,
                "all": cmd_all, "serve": cmd_serve}
    commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
