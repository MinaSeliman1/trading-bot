"""
Sensitivity analysis on the PEAD viability verdict: does the pass/fail
result and the Sharpe magnitude hold up under small, legitimate walk-
forward window-width variations not chosen to improve the number? Runs
train_months in {12, 18, 24, 30, 36} (test_months=6 fixed) on both the
full 97-ticker universe and the independent 74-new-only subset.
"""
import logging
from pathlib import Path

from run_pead_viability_check import run_viability_check, discover_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_sensitivity")

LOG_PATH = "exploration_log_pead.md"
ORIGINAL_23 = ["JPM", "BAC", "GS", "JNJ", "UNH", "PFE", "ABBV", "PG", "KO", "WMT",
              "MCD", "CAT", "HON", "UPS", "XOM", "CVX", "VZ", "DIS", "HD", "COST",
              "LIN", "DUK", "V"]
TRAIN_MONTHS_GRID = [12, 18, 24, 30, 36]


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def main():
    all_tickers = discover_tickers()
    new_only = [t for t in all_tickers if t not in ORIGINAL_23]

    log_md("\n## Sensitivity analysis: walk-forward train-window width\n")
    log_md(f"- Sweeping train_months in {TRAIN_MONTHS_GRID} (test_months=6 fixed throughout -- "
          f"unchanged since the very first PEAD run). Not chosen post-hoc to improve any number -- "
          f"a symmetric bracket around the original 24mo choice. Corrected (real-price mark-to-market) "
          f"engine throughout.")
    log_md(f"- Also confirmed: BK and DFS (the 2 no-data symbols) never produced a cache file and are "
          f"therefore ALREADY excluded from every run by construction -- there is no data to "
          f"'include', so this dimension is not separately testable, and doesn't affect any result below.")

    results = {"full_97": [], "new_only_74": []}
    for train_months in TRAIN_MONTHS_GRID:
        r_full = run_viability_check(tickers=all_tickers,
                                     label=f"sensitivity: full universe, train={train_months}mo/test=6mo",
                                     train_months=train_months, test_months=6)
        results["full_97"].append((train_months, r_full))

        r_new = run_viability_check(tickers=new_only,
                                    label=f"sensitivity: new-only 74, train={train_months}mo/test=6mo",
                                    train_months=train_months, test_months=6)
        results["new_only_74"].append((train_months, r_new))

    log_md("\n### Sensitivity summary table\n")
    log_md("| train_months | Full-97 Sharpe | Full-97 MC_DD95 | Full-97 windows flagged | "
          "New-74 Sharpe | New-74 MC_DD95 | New-74 windows flagged |")
    log_md("|---|---|---|---|---|---|---|")
    for i, train_months in enumerate(TRAIN_MONTHS_GRID):
        _, r_full = results["full_97"][i]
        _, r_new = results["new_only_74"][i]
        def fmt(r):
            if r is None:
                return "N/A", "N/A", "N/A"
            m, mc = r["overall_metrics"], r["mc"]
            n_flag = sum(1 for w in r["window_results"] if not w["deviation_ok"])
            return f"{m['sharpe']:.3f}", f"{mc['max_dd_p95']*100:.1f}%", f"{n_flag}/{len(r['window_results'])}"
        fs, fd, fw = fmt(r_full)
        ns, nd, nw = fmt(r_new)
        log_md(f"| {train_months} | {fs} | {fd} | {fw} | {ns} | {nd} | {nw} |")

    full_sharpes = [r["overall_metrics"]["sharpe"] for _, r in results["full_97"] if r]
    new_sharpes = [r["overall_metrics"]["sharpe"] for _, r in results["new_only_74"] if r]
    full_pass = [r["viable"] for _, r in results["full_97"] if r]
    new_pass = [r["viable"] for _, r in results["new_only_74"] if r]

    log_md(f"\n- Full-97 Sharpe range across train-window widths: "
          f"{min(full_sharpes):.3f} to {max(full_sharpes):.3f} ({sum(full_pass)}/{len(full_pass)} pass)")
    log_md(f"- New-only-74 Sharpe range across train-window widths: "
          f"{min(new_sharpes):.3f} to {max(new_sharpes):.3f} ({sum(new_pass)}/{len(new_pass)} pass)")

    if min(full_sharpes) >= 0.5 and min(new_sharpes) >= 0.5:
        log_md("\n**CONCLUSION: the pass verdict is ROBUST to train-window width -- Sharpe stays above "
              "the 0.5 bar across all 5 widths tested on both universes, not just the originally-chosen "
              "24mo.**")
    elif max(full_sharpes) < 0.5 or max(new_sharpes) < 0.5:
        log_md("\n**CONCLUSION: the pass verdict does NOT survive -- Sharpe falls below 0.5 across most "
              "or all alternative train-window widths, meaning the original 24mo pass was fragile/"
              "coincidental, not a robust reading.**")
    else:
        log_md("\n**CONCLUSION: the pass verdict is FRAGILE -- Sharpe oscillates across the 0.5 bar "
              "depending on train-window width, a small and reasonable methodology choice not chosen "
              "to help the result. This is exactly the fragility signature the sensitivity check was "
              "designed to catch.**")

    return results


if __name__ == "__main__":
    main()
