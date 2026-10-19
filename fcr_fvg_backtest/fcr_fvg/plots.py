"""Static charts (matplotlib, Agg). Colour = instrument, line style = stop variant.
Palette: reference categorical slots 1,2,3,7 (validated adjacent CVD/normal-vision).
The aqua slot is below 3:1 on white, so every series is also named in a legend and
every chart has a CSV twin in the output folder (relief rule)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import drawdown_series

COLORS = {"SPY": "#2a78d6", "QQQ": "#eb6834", "ES": "#1baf7a", "NQ": "#6250d6"}
STYLES = {"A": "-", "B": "--"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    _style(ax)
    return fig, ax


def equity_curves(results: dict, path: str, initial: float):
    """results[(sym, variant)] = engine output."""
    fig, ax = _fig()
    for (sym, v), res in results.items():
        d = res["daily"]
        if d.empty:
            continue
        x = pd.to_datetime(d.index)
        ax.plot(x, (d["equity"] / initial - 1) * 100, color=COLORS.get(sym, "#444"), ls=STYLES[v], lw=1.6,
                label=f"{sym} – stop {v}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Net return since start (%)", color=MUTED)
    ax.set_title("Equity curve (net of commissions and slippage)", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def drawdowns(results: dict, path: str, initial: float):
    fig, ax = _fig()
    for (sym, v), res in results.items():
        d = res["daily"]
        if d.empty:
            continue
        dd = drawdown_series(d["equity"], initial)
        ax.plot(pd.to_datetime(d.index), dd, color=COLORS.get(sym, "#444"), ls=STYLES[v], lw=1.4, label=f"{sym} – stop {v}")
    ax.set_ylabel("Drawdown (%)", color=MUTED)
    ax.set_title("Drawdown below running peak", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def monthly_bars(results: dict, path: str):
    keys = sorted(results)
    syms = sorted({k[0] for k in keys}); variants = sorted({k[1] for k in keys})
    fig, axes = plt.subplots(len(syms), len(variants), figsize=(5.6 * len(variants), 2.4 * len(syms)),
                             dpi=130, squeeze=False, sharex=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, sym in enumerate(syms):
        for j, v in enumerate(variants):
            ax = axes[i][j]; ax.set_facecolor("#fcfcfb"); _style(ax)
            t = results.get((sym, v), {}).get("trades", pd.DataFrame())
            if len(t):
                m = t.groupby(pd.to_datetime(t["date"]).dt.to_period("M"))["net_pnl"].sum()
                col = [COLORS[sym] if x >= 0 else "#a0a0a0" for x in m.values]
                ax.bar(m.index.to_timestamp(), m.values, width=20, color=col)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{sym} – stop {v}", color=INK, loc="left", fontsize=10)
            ax.set_ylabel("Net P&L ($)", color=MUTED, fontsize=8)
    fig.suptitle("Monthly net P&L (colour = gain, grey = loss)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def r_histograms(results: dict, path: str):
    keys = sorted(results)
    syms = sorted({k[0] for k in keys}); variants = sorted({k[1] for k in keys})
    fig, axes = plt.subplots(len(syms), len(variants), figsize=(5.6 * len(variants), 2.4 * len(syms)),
                             dpi=130, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    bins = np.arange(-3.0, 3.75, 0.25)
    for i, sym in enumerate(syms):
        for j, v in enumerate(variants):
            ax = axes[i][j]; ax.set_facecolor("#fcfcfb"); _style(ax)
            t = results.get((sym, v), {}).get("trades", pd.DataFrame())
            if len(t):
                r = t["r_multiple"].clip(-3, 3.5)
                ax.hist(r, bins=bins, color=COLORS[sym], edgecolor="#fcfcfb", linewidth=1)
                ax.axvline(t["r_multiple"].mean(), color=INK, lw=1.2, ls=":")
            ax.set_title(f"{sym} – stop {v}", color=INK, loc="left", fontsize=10)
            ax.set_xlabel("Net R per trade (clipped to [-3, 3.5]; dotted = mean)", color=MUTED, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
