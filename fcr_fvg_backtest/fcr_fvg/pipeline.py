"""End-to-end study: load -> QC -> backtests (both stop variants) -> IS/OOS ->
bootstrap -> sensitivity -> charts -> machine-generated report.

The report written here contains ONLY computed numbers plus a mechanical checklist.
The verdict (category 1-5) is a human judgement made from those numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots, stats
from .data import Session, build_sessions, load_ohlcv
from .engine import SignalCache, run_backtest
from .instruments import INSTRUMENTS
from .params import StrategyParams

VARIANTS = ("A", "B")
SENS_SLIPPAGE = (1, 2, 3, 5)
SENS_TTL = (5, 10, 15, 20, 30)
SENS_CONFIRM = (0, 1, 2, 3, 5)


# ----------------------------------------------------------------------------
def _slice(res: dict, first: object | None, last_excl: object | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, t = res["daily"], res["trades"]
    dm = pd.Series(True, index=d.index)
    if first is not None:
        dm &= d.index >= first
    if last_excl is not None:
        dm &= d.index < last_excl
    d2 = d[dm]
    t2 = t[t["date"].isin(d2.index)] if len(t) else t
    return t2, d2


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and np.isinf(x):
        return "inf"
    return f"{x:,.{nd}f}" if isinstance(x, float) else str(x)


def md_table(df: pd.DataFrame, floatfmt: int = 2) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.reset_index() if (isinstance(df.index, pd.MultiIndex) or df.index.name) else df.reset_index(names="")
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            return _fmt(fv, 0) if (fv.is_integer() and floatfmt == 0) else _fmt(fv, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def audit_text(s: Session, tr: pd.Series, context_before: int = 3) -> str:
    """Human-checkable dump of the bars behind one trade (for manual verification)."""
    lines = [f"{tr['symbol']} {tr['date']} stop-variant {tr['variant']} {tr['direction']}",
             f"OR high/low: {tr['or_high']} / {tr['or_low']}  FVG zone: [{tr['zone_lo']}, {tr['zone_hi']}]",
             f"FVG complete {tr['fvg_time']} | retest bar {tr['retest_time']} | signal (engulf closes) {tr['signal_time']}",
             f"entry {tr['entry_price']} @ {tr['entry_time']} | stop {tr['stop']} | target {tr['target']} | "
             f"exit {tr['exit_price']} @ {tr['exit_time']} ({tr['exit_reason']})",
             f"units {tr['units']}  commissions {tr['commissions']:.2f}  net P&L {tr['net_pnl']:.2f}  R {tr['r_multiple']:.3f}", "",
             "idx  time   open     high     low      close"]
    lo = 0
    hi = int(tr["exit_idx"]) + 1
    for i in range(lo, hi):
        if i >= 8 and i < int(tr["entry_idx"]) - 12:
            continue
        if s.present(i):
            lines.append(f"{i:3d}  {s.bar_open_time(i).strftime('%H:%M')}  {s.o[i]:<8.2f} {s.h[i]:<8.2f} {s.l[i]:<8.2f} {s.c[i]:<8.2f}")
        else:
            lines.append(f"{i:3d}  {s.bar_open_time(i).strftime('%H:%M')}  (missing bar)")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
def run_study(files: dict[str, str], out_dir: str, input_tz: str | None = None, ts_convention: str = "start",
              base: StrategyParams | None = None, n_boot: int = 5000, sensitivity: bool = True,
              audit_n: int = 10, oos_frac: float = 0.7, use_calendar: bool = True) -> dict:
    base = base or StrategyParams()
    out = Path(out_dir)
    for sub in ("trades", "breakdowns", "audit", "qc", "charts"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. data -------------------------------------------------------------
    sessions: dict[str, list[Session]] = {}
    qcs = {}
    for sym, path in files.items():
        if sym not in INSTRUMENTS:
            raise KeyError(f"Unknown instrument {sym}; known: {list(INSTRUMENTS)}")
        df = load_ohlcv(path, input_tz=input_tz, ts_convention=ts_convention)
        sess, qc = build_sessions(df, INSTRUMENTS[sym], base, use_calendar=use_calendar)
        sessions[sym], qcs[sym] = sess, qc
        (out / "qc" / f"{sym}.json").write_text(json.dumps(qc.to_dict(), indent=2, default=str))
    all_dates = [s.date for ss in sessions.values() for s in ss]
    if not all_dates:
        raise RuntimeError("No usable session in the provided data - see qc/*.json")
    oos_start = stats.split_dates(all_dates, oos_frac)

    # ---- 2. main runs ----------------------------------------------------------
    caches = {sym: SignalCache() for sym in sessions}
    results: dict[tuple, dict] = {}
    for sym, ss in sessions.items():
        for v in VARIANTS:
            res = run_backtest(ss, INSTRUMENTS[sym], base.with_(stop_variant=v), caches[sym])
            results[(sym, v)] = res
            res["trades"].to_csv(out / "trades" / f"trades_{sym}_{v}.csv", index=False)
            if len(res["rejections"]):
                res["rejections"].to_csv(out / "trades" / f"rejections_{sym}_{v}.csv", index=False)
            for _, row in res["trades"].head(audit_n).iterrows():
                s = next(x for x in ss if x.date == row["date"])
                (out / "audit" / f"{sym}_{v}_{row['date']}.txt").write_text(audit_text(s, row))

    # ---- 3. tables ----------------------------------------------------------------
    rows, bootstrap = [], {}
    for (sym, v), res in results.items():
        for seg, (a, b) in {"FULL": (None, None), "IS": (None, oos_start), "OOS": (oos_start, None)}.items():
            t, d = _slice(res, a, b)
            m = stats.summarize(t, d, base.initial_capital)
            m.update(instrument=sym, stop=v, segment=seg)
            rows.append(m)
            if seg in ("FULL", "OOS"):
                bootstrap[(sym, v, seg)] = stats.bootstrap_sessions(t, d, n_boot)
    summary = pd.DataFrame(rows).set_index(["instrument", "stop", "segment"])
    summary.to_csv(out / "summary.csv")

    rej_rows = []
    for (sym, v), res in results.items():
        r = res["rejections"]
        counts = r["reason"].value_counts().to_dict() if len(r) else {}
        rej_rows.append({"instrument": sym, "stop": v, "sessions": len(res["daily"]),
                         "sessions_with_signal": int((res["daily"]["n_signals"] > 0).sum()),
                         "trades": len(res["trades"]), **{f"rej_{k}": c for k, c in counts.items()}})
    rejections = pd.DataFrame(rej_rows).set_index(["instrument", "stop"]).fillna(0)
    rejections.to_csv(out / "rejections_summary.csv")

    breakdown_md = []
    for (sym, v), res in results.items():
        t = res["trades"]
        parts = {
            "by_direction": stats.by_direction(t), "by_entry_slot_30min": stats.by_entry_slot(t),
            "by_exit_reason": stats.by_exit_reason(t), "by_month": stats.by_month(t),
            "by_half_year": stats.by_period(t, "H"), "by_year": stats.by_period(t, "Y"),
            "by_or_width_tercile": stats.by_tercile(t, res["features"], "or_width_pct"),
            "by_prev20_range_tercile": stats.by_tercile(t, res["features"], "prev20_range_pct"),
        }
        for name, df in parts.items():
            if len(df):
                df.to_csv(out / "breakdowns" / f"{name}_{sym}_{v}.csv")
        breakdown_md.append((sym, v, parts, stats.concentration(t)))

    # ---- 4. sensitivity (one-at-a-time; reported, never used to pick anything) ---
    sens_rows = []
    if sensitivity:
        grid = [("slippage_ticks", x) for x in SENS_SLIPPAGE] + [("fvg_ttl_bars", x) for x in SENS_TTL] \
            + [("confirm_extra_bars", x) for x in SENS_CONFIRM] + [("check_exit_on_entry_bar", False)]
        for sym, ss in sessions.items():
            inst = INSTRUMENTS[sym]
            g = list(grid) + ([("max_leverage_etf", 1.0)] if inst.kind == "etf" else [])
            for v in VARIANTS:
                for field, val in g:
                    p = base.with_(stop_variant=v, **{field: val})
                    res = run_backtest(ss, inst, p, caches[sym])
                    for seg, (a, b) in {"FULL": (None, None), "OOS": (oos_start, None)}.items():
                        t, d = _slice(res, a, b)
                        m = stats.summarize(t, d, base.initial_capital)
                        sens_rows.append({"instrument": sym, "stop": v, "param": field, "value": val, "segment": seg,
                                          "is_main": getattr(base, field) == val,
                                          "trades": m["trades"], "win_rate": m["win_rate"], "expectancy_r": m["expectancy_r"],
                                          "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
                                          "max_dd_pct": m["max_dd_pct"]})
        sens = pd.DataFrame(sens_rows)
        sens.to_csv(out / "sensitivity.csv", index=False)
    else:
        sens = pd.DataFrame()

    # ---- 5. charts -----------------------------------------------------------------
    plots.equity_curves(results, str(out / "charts" / "equity_curves.png"), base.initial_capital)
    plots.drawdowns(results, str(out / "charts" / "drawdowns.png"), base.initial_capital)
    plots.monthly_bars(results, str(out / "charts" / "monthly_pnl.png"))
    plots.r_histograms(results, str(out / "charts" / "r_distribution.png"))
    for (sym, v), res in results.items():
        res["daily"].to_csv(out / "trades" / f"daily_{sym}_{v}.csv")

    # ---- 6. machine-generated report ----------------------------------------------
    report = _write_report(out, base, files, qcs, summary, rejections, bootstrap, breakdown_md, sens, oos_start)
    return {"summary": summary, "bootstrap": bootstrap, "results": results, "report": report, "qc": qcs,
            "sensitivity": sens, "oos_start": oos_start}


# ----------------------------------------------------------------------------
def _write_report(out, base, files, qcs, summary, rejections, bootstrap, breakdown_md, sens, oos_start) -> str:
    L = []
    if any("synth" in str(f).lower() for f in files.values()):
        L.append("# WARNING: AN INPUT FILE NAME CONTAINS 'synth'. IF THIS DATA IS SIMULATED, NOTHING BELOW IS A MARKET RESULT.\n")
    L.append("# First Candle Rule + FVG — machine-generated results\n")
    L.append("> Every figure below was computed by `fcr_fvg` from the data files listed in section 1. "
             "This file contains **no verdict**; the classification is a human judgement (see README).\n")
    L.append("## 1. Data actually used\n")
    for sym, qc in qcs.items():
        d = qc.to_dict()
        L.append(f"### {sym}  (`{files[sym]}`)\n")
        keep = {k: d[k] for k in ("source_rows", "data_start", "data_end", "sessions_in_data", "sessions_kept",
                                  "sessions_excluded", "calendar_days_without_data", "early_close_sessions",
                                  "rows_outside_rth_dropped", "rows_after_official_close_dropped", "rows_duplicate_dropped", "rows_bad_ohlc_dropped",
                                  "missing_minutes_in_kept_sessions", "expected_minutes_in_kept_sessions", "contract_changes")}
        L.append("```\n" + json.dumps(keep, indent=1, default=str) + "\n```\n")
        if d["notes"]:
            L.append("Notes: " + "; ".join(d["notes"]) + "\n")
    L.append(f"IS/OOS split: first {int(100 * 0.7)}% of sorted session dates = in-sample; "
             f"**first OOS session = {oos_start}**. No parameter was fitted on either part.\n")
    L.append("## 2. Main configuration\n")
    L.append("```\n" + json.dumps({k: getattr(base, k) for k in base.__dataclass_fields__}, indent=1) + "\n```\n")

    cols = ["trades", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "net_return_pct", "expectancy_r",
            "avg_win_r", "avg_loss_r", "payoff_ratio"]
    L.append("## 3. Performance (net of commissions and slippage)\n")
    for seg in ("FULL", "IS", "OOS"):
        L.append(f"### {seg}\n")
        L.append(md_table(summary.xs(seg, level="segment")[cols]))
    risk_cols = ["max_dd_usd", "max_dd_pct", "longest_loss_streak", "longest_win_streak", "avg_risk_pct_equity",
                 "max_risk_pct_equity", "max_exposure_pct_equity", "max_margin_pct_equity", "trades_per_week",
                 "avg_duration_min", "commissions", "slippage_cost", "size_capped_trades"]
    L.append("## 4. Risk and behaviour (FULL period)\n")
    L.append(md_table(summary.xs("FULL", level="segment")[risk_cols]))
    L.append("## 5. Signals that did not become trades\n")
    L.append(md_table(rejections, 0))

    L.append("## 6. Statistical uncertainty — session bootstrap (days resampled, 95% interval)\n")
    brow = []
    for (sym, v, seg), b in bootstrap.items():
        if b.get("n_boot", 0) == 0:
            brow.append({"instrument": sym, "stop": v, "segment": seg, "note": b.get("note")})
            continue
        brow.append({"instrument": sym, "stop": v, "segment": seg,
                     "expectancy_R 2.5%": b["expectancy_r_ci95"][0], "median": b["expectancy_r_ci95"][1],
                     "97.5%": b["expectancy_r_ci95"][2], "P(exp<=0)": b["p_expectancy_le_0"],
                     "PF 2.5%": b["profit_factor_ci95"][0], "PF 97.5%": b["profit_factor_ci95"][2],
                     "daily-R t-stat": b["daily_r_t_stat"]})
    L.append(md_table(pd.DataFrame(brow).set_index(["instrument", "stop", "segment"]) if brow else pd.DataFrame(), 3))

    L.append("## 7. Breakdowns and concentration\n")
    for sym, v, parts, conc in breakdown_md:
        L.append(f"### {sym} — stop {v}\n")
        if conc:
            L.append("Concentration: " + ", ".join(f"{k}={_fmt(x, 2) if isinstance(x, float) else x}" for k, x in conc.items()) + "\n")
        for name in ("by_direction", "by_entry_slot_30min", "by_exit_reason", "by_year", "by_or_width_tercile",
                     "by_prev20_range_tercile"):
            df = parts.get(name)
            if df is not None and len(df):
                L.append(f"**{name}**" + (f" (tercile cut-offs {tuple(round(x, 4) for x in df.attrs['thresholds'])})"
                                          if "thresholds" in df.attrs else "") + "\n")
                L.append(md_table(df))
        L.append("Monthly results: see `breakdowns/by_month_*.csv`.\n")

    if len(sens):
        L.append("## 8. Sensitivity (one parameter changed at a time; `is_main` marks the main configuration)\n")
        for seg in ("FULL", "OOS"):
            L.append(f"### {seg}\n")
            s = sens[sens.segment == seg].drop(columns="segment")
            L.append(md_table(s.set_index(["instrument", "stop", "param", "value"]), 3))

    L.append("## 9. Mechanical checklist (NOT a verdict)\n")
    chk = []
    for (sym, v) in sorted({(i, s) for i, s, _ in summary.index}):
        f, o = summary.loc[(sym, v, "FULL")], summary.loc[(sym, v, "OOS")]
        bf = bootstrap.get((sym, v, "FULL"), {})
        lo = bf.get("expectancy_r_ci95", (np.nan,))[0] if bf.get("n_boot") else np.nan
        chk.append({"instrument": sym, "stop": v, "net_pnl>0 (full)": bool(f["net_pnl"] > 0),
                    "net_pnl>0 (OOS)": bool(o["net_pnl"] > 0), "expectancy_R>0 (OOS)": bool(o["expectancy_r"] > 0),
                    "bootstrap 2.5% of E[R] > 0 (full)": bool(lo > 0) if lo == lo else "n/a",
                    "trades (full)": int(f["trades"]), "trades (OOS)": int(o["trades"])})
    L.append(md_table(pd.DataFrame(chk).set_index(["instrument", "stop"]) if chk else pd.DataFrame(), 0))
    text = "\n".join(L)
    (out / "report_auto.md").write_text(text)
    return text
