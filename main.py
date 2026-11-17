"""Forecast a launch day's hourly Twitch viewers from its first hour.

Method, fitted at run time from the case's own comparables (past big
launches, each with a full release-day curve):

1.  Keep the comparables that were already live at hour 1 -- hour 1 at
    least 3% of the day's peak -- so simultaneous global unlocks, whose
    hour 1 is a pre-unlock trickle, do not bend the shape.  Comparables
    the case itself tiers as early-access-contaminated or weak are left
    out when enough remain.
2.  For each hour h = 2..24, regress the log ratio log(viewers_h /
    viewers_1) on log(viewers_1) across the pool.  The slope is near
    zero in the small hours and about -0.45 by the evening: a launch
    whose first hour is already enormous has less headroom left, while
    a modest first hour leaves room for the US unlock ramp.  The slope
    is shrunk by 0.9, and beyond the pool's log(viewers_1) range only
    half the slope applies, so an unprecedented first hour is not
    extrapolated linearly into a collapse.
3.  The forecast for hour h is lognormal: median p1 * exp(mu_h), spread
    from the regression's residual standard deviation, smoothed over
    neighbouring hours.  Quantiles come from the normal quantile
    function at the case's levels.

Everything is deterministic: sorted inputs, fixed arithmetic, no
randomness, no clock, no network.  Standard library only.

Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input), one JSON case
        (schema makefun/twitch-ccu-case/v1).
Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output):
        {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# A comparable counts as "already live at hour 1" when hour 1 reaches
# this share of its release-day peak.  Simultaneous-unlock launches sit
# below ~1%; rolling-midnight launches sit above ~8%.
LIVE_RATIO = 0.03
# Tiers the case data itself marks as unusable analogues.
BAD_TIERS = {"early-access-contaminated", "weak"}
TIER_RE = re.compile(r"Tier:\s*([A-Za-z][A-Za-z-]*)", re.IGNORECASE)
# Shrinkage on every regression slope, and the fraction of the slope
# that still applies past the pool's log-first-hour range.
SLOPE_SHRINK = 0.9
EXTRAPOLATION_SLOPE = 0.5
# Residual spread is floored (never pretend to near-certainty) and
# capped (never let one wild pool hour blow the band up).
SD_FLOOR = 0.15
SD_CAP = 1.0
# Fallback spread for the persistence forecast, in log units per
# square-root hour, used only when no usable pool exists.
PERSISTENCE_SIGMA = 0.15


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to about 1e-9 over the open unit interval; the scored
    levels 0.05/0.25/0.5/0.75/0.95 come out within a whisker of the
    textbook z values.  Deterministic and dependency-free.
    """
    p = min(max(p, 0.0001), 0.9999)
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when unparseable."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def round_half_up_2dp(x):
    """Round to 2 decimals, halves away from zero (stable bytes)."""
    scaled = math.floor(x * 100.0 + 0.5) / 100.0
    return int(scaled) if scaled.is_integer() else scaled


def release_day_curve(comparable):
    """Return {hour: viewers} for hours 1..24 of a comparable's day.

    Hours come from the hourly points whose timestamp falls on the
    release day's hour grid; points off the grid or with no positive
    viewer count are ignored.  None when the day start is unusable.
    """
    day_start = parse_time(comparable.get("day_start"))
    hourly = comparable.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return None
    curve = {}
    for point in hourly:
        if not isinstance(point, dict):
            continue
        moment = parse_time(point.get("from"))
        viewers = point.get("viewers")
        if moment is None or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        offset = (moment - day_start).total_seconds()
        if offset < 0 or offset >= 24 * 3600:
            continue
        if abs(offset - round(offset / 3600.0) * 3600.0) > 60:
            continue
        hour = int(round(offset / 3600.0)) + 1
        curve[hour] = float(viewers)
    return curve


def comparable_tier(comparable):
    """The house tier from the comparable's unlock note, if present."""
    unlock = comparable.get("unlock")
    if not isinstance(unlock, str):
        return None
    match = TIER_RE.search(unlock)
    return match.group(1).lower() if match else None


def build_pool(comparables):
    """Select the analogue pool, relaxing filters only if too small.

    Each pool member is (log_first_hour, {hour: log_ratio}) with
    log_ratio = log(viewers_h / viewers_1).  Returns an empty list
    when nothing usable exists.
    """
    candidates = []
    for comparable in comparables or []:
        if not isinstance(comparable, dict):
            continue
        curve = release_day_curve(comparable)
        if curve is None or 1 not in curve or len(curve) < 22:
            continue
        first = curve[1]
        peak = max(curve.values())
        ratio = first / peak if peak > 0 else 0.0
        ratios = {h: math.log(v / first) for h, v in curve.items()}
        candidates.append({
            "log_first": math.log(first),
            "ratios": ratios,
            "live_ratio": ratio,
            "tier": comparable_tier(comparable),
        })
    if not candidates:
        return []
    # Strictest first: already-live launches of the comparable tier.
    pool = [c for c in candidates
            if c["live_ratio"] >= LIVE_RATIO and c["tier"] not in BAD_TIERS]
    if len(pool) < 10:
        pool = [c for c in candidates if c["live_ratio"] >= LIVE_RATIO]
    if len(pool) < 8:
        pool = [c for c in candidates if c["live_ratio"] >= 0.01]
    if len(pool) < 5:
        pool = candidates
    return pool


def fit_hour(pool, hour, log_first_target):
    """OLS of log ratio on log first hour for one hour; (mu, sd).

    Returns None when fewer than five pool members carry this hour.
    """
    xs = []
    ys = []
    for member in pool:
        if hour in member["ratios"]:
            xs.append(member["log_first"])
            ys.append(member["ratios"][hour])
    if len(xs) < 5:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom > 0:
        slope = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys)) / denom
    slope *= SLOPE_SHRINK
    x = log_first_target
    low, high = min(xs), max(xs)
    if x > high:
        x = high + EXTRAPOLATION_SLOPE * (x - high)
    elif x < low:
        x = low + EXTRAPOLATION_SLOPE * (x - low)
    mu = mean_y + slope * (x - mean_x)
    resid_sq = sum((y - (mean_y + slope * (xx - mean_x))) ** 2
                   for xx, y in zip(xs, ys))
    sd = math.sqrt(resid_sq / n)
    return mu, min(max(sd, SD_FLOOR), SD_CAP)


def smooth(values):
    """Three-point moving average over neighbouring hours (edges: two)."""
    smoothed = {}
    for hour, value in values.items():
        neighbours = [values[h] for h in (hour - 1, hour, hour + 1)
                      if h in values]
        smoothed[hour] = sum(neighbours) / len(neighbours)
    return smoothed


def fit_shape(pool, hours, log_first_target):
    """Per-hour (mu, sd) of the log ratio, with gaps filled in.

    Hours the pool cannot fit borrow the nearest fitted hour; with no
    fitted hour at all, returns None and the caller falls back.
    """
    fitted = {}
    for hour in hours:
        result = fit_hour(pool, hour, log_first_target)
        if result is not None:
            fitted[hour] = result
    if not fitted:
        return None
    mus = smooth({h: fitted[h][0] for h in fitted})
    sds = smooth({h: fitted[h][1] for h in fitted})
    filled = {}
    known = sorted(fitted)
    for hour in hours:
        if hour in fitted:
            filled[hour] = (mus[hour], sds[hour])
        else:
            nearest = min(known, key=lambda h: abs(h - hour))
            filled[hour] = (mus[nearest], sds[nearest])
    return filled


def persistence_quantiles(p1, hour, z_values):
    """Flat forecast with a widening random-walk spread (fallback)."""
    spread = PERSISTENCE_SIGMA * math.sqrt(max(hour - 1, 0))
    return [p1 * math.exp(z * spread) for z in z_values]


def forecast_case(case, name):
    """Forecast one case dict; always returns a valid-shaped object."""
    levels = case.get("levels")
    if (not isinstance(levels, list) or not levels
            or any(not isinstance(level, (int, float))
                   or not 0.0 < level < 1.0 for level in levels)):
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    z_values = [norm_ppf(float(level)) for level in levels]
    forecast_rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in forecast_rows:
        if isinstance(row, dict) and isinstance(row.get("hour"), int):
            hours.append(row["hour"])
    if not hours:
        hours = list(range(2, 25))

    p1 = None
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                p1 = float(viewers)
                break
    if p1 is None:
        # No hour-1 print: prefer the hourly mean for the day's first
        # hour itself, else the last pre-checkpoint hourly mean, else
        # a neutral scale.  The shape still comes from the pool.
        hourly = case.get("history_hourly") or []
        day_start = parse_time(case.get("day_start"))
        fallback = None
        for point in hourly:
            if not isinstance(point, dict):
                continue
            viewers = point.get("viewers")
            if not (isinstance(viewers, (int, float)) and viewers > 0):
                continue
            fallback = float(viewers)
            if day_start is not None and parse_time(point.get("from")) == day_start:
                fallback = float(viewers)
                break
        p1 = fallback if fallback is not None else 1000.0

    shape = None
    pool = build_pool(case.get("comparables"))
    if len(pool) >= 3:
        shape = fit_shape(pool, hours, math.log(p1))

    rows = []
    for hour in hours:
        if shape is not None:
            mu, sd = shape[hour]
            quantiles = [p1 * math.exp(mu + z * sd) for z in z_values]
        else:
            quantiles = persistence_quantiles(p1, hour, z_values)
        quantiles = [min(max(q, 1.0), 1_000_000_000.0) for q in quantiles]
        for i in range(1, len(quantiles)):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        rows.append({"hour": hour,
                     "q": [round_half_up_2dp(q) for q in quantiles]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never crash a graded run
            print(f"{path.name}: model failed ({exc}); "
                  f"writing persistence fallback", flush=True)
            p1 = 1000.0
            for row in case.get("known") or []:
                if isinstance(row, dict) and row.get("hour") == 1:
                    viewers = row.get("viewers")
                    if isinstance(viewers, (int, float)) and viewers > 0:
                        p1 = float(viewers)
            z_values = [norm_ppf(level)
                        for level in (0.05, 0.25, 0.5, 0.75, 0.95)]
            forecast = {"hours": [
                {"hour": h,
                 "q": [round_half_up_2dp(q) for q in
                       persistence_quantiles(p1, h, z_values)]}
                for h in range(2, 25)]}
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
