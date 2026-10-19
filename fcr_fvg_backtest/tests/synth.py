"""Synthetic random-walk generator for TESTS ONLY (property/invariant checks).
Output is never used to report performance."""
import numpy as np
import pandas as pd

NY = "America/New_York"


def random_walk_frame(dates, seed=0, start=100.0, vol=0.0006, tick=0.01, extended=True, tz_out="UTC"):
    rng = np.random.default_rng(seed)
    rows = []
    px = start
    for d in dates:
        mins = list(range(8 * 60, 20 * 60)) if extended else list(range(9 * 60 + 30, 16 * 60))
        for m in mins:
            ts = pd.Timestamp(f"{d} {m // 60:02d}:{m % 60:02d}", tz=NY)
            o = px
            path = o * np.exp(np.cumsum(rng.normal(0, vol / 2, 3)))
            c = path[-1]
            h = max(o, c, path.max()) * (1 + abs(rng.normal(0, vol / 4)))
            l = min(o, c, path.min()) * (1 - abs(rng.normal(0, vol / 4)))
            o, h, l, c = (round(x / tick) * tick for x in (o, h, l, c))
            h, l = max(h, o, c), min(l, o, c)
            rows.append((ts.tz_convert(tz_out), o, h, l, c, int(rng.integers(100, 10000))))
            px = c
        px = px * (1 + rng.normal(0, 0.003))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def bday_range(start, end):
    return [d.date() for d in pd.bdate_range(start, end)]
