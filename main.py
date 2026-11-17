"""Forecast a launch day's hourly Twitch viewers from its first hour.

Method, fitted at run time from the case's own comparables (past big
launches, each with a full release-day curve):

1.  Keep the comparables that were already live at hour 1 -- hour 1 at
    least 3% of the day's peak -- so simultaneous global unlocks, whose
    hour 1 is a pre-unlock trickle, do not bend the shape.  Comparables
    the case itself tiers as early-access-contaminated or weak are left
    out when enough remain.
2.  For each hour h = 2..24, regress the log ratio log(viewers_h /
    viewers_1) on log(viewers_1) across the pool, with two Huber
    reweighting passes so one wild analogue cannot drag the shape.
    The slope is near zero in the small hours and about -0.45 by the
    evening: a launch whose first hour is already enormous has less
    headroom left, while a modest first hour leaves room for the US
    unlock ramp.  The slope is shrunk by 0.9, and beyond the pool's
    log(viewers_1) range only half the slope applies, so an
    unprecedented first hour is not extrapolated linearly into a
    collapse.
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
from statistics import NormalDist

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


_NORMAL = NormalDist()


def norm_ppf(p):
    """Inverse standard-normal CDF, exact to machine precision.

    The standard library's NormalDist (Wichura's AS 241) replaces a
    hand-rolled rational approximation: no coefficient table to
    mistype, and the scored levels come out at the textbook z values.
    Clamped away from 0 and 1 so the function is total.
    """
    return _NORMAL.inv_cdf(min(max(p, 0.0001), 0.9999))


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


def is_number(value):
    """True for a finite JSON number (bools, NaN and inf excluded)."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


def format_quantiles(quantiles):
    """Clamp to the scored range, enforce order, round for output."""
    bounded = [min(max(q, 1.0), 1_000_000_000.0) for q in quantiles]
    for i in range(1, len(bounded)):
        bounded[i] = max(bounded[i], bounded[i - 1])
    return [round_half_up_2dp(q) for q in bounded]


def release_day_curve(comparable):
    """Return {hour: viewers} for hours 1..24 of a comparable's day.

    Hours come from the hourly points whose timestamp falls on the
    release day's hour grid; points off the grid or with no positive
    finite viewer count are ignored, and repeated points for one hour
    are averaged.  None when the day start is unusable.
    """
    day_start = parse_time(comparable.get("day_start"))
    hourly = comparable.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return None
    totals = {}
    counts = {}
    for point in hourly:
        if not isinstance(point, dict):
            continue
        moment = parse_time(point.get("from"))
        viewers = point.get("viewers")
        if moment is None or not is_number(viewers) or viewers <= 0:
            continue
        offset = (moment - day_start).total_seconds()
        if offset < 0 or offset >= 24 * 3600:
            continue
        if abs(offset - round(offset / 3600.0) * 3600.0) > 60:
            continue
        hour = int(round(offset / 3600.0)) + 1
        totals[hour] = totals.get(hour, 0.0) + float(viewers)
        counts[hour] = counts.get(hour, 0) + 1
    return {hour: totals[hour] / counts[hour] for hour in totals}


def comparable_tier(comparable):
    """The house tier from the comparable's unlock note, if present."""
    unlock = comparable.get("unlock")
    if not isinstance(unlock, str):
        return None
    match = TIER_RE.search(unlock)
    return match.group(1).lower() if match else None


def build_pool(comparables):
    """Select the analogue pool, relaxing filters only if too small.

    Each pool member is a dict with log_first (the log of hour 1's
    viewers), ratios ({hour: log(viewers_h / viewers_1)}), live_ratio
    and tier.  Returns an empty list when nothing usable exists.
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
    """Robust regression of log ratio on log first hour; (mu, sd).

    Ordinary least squares, then two Huber reweighting passes: pool
    members whose residual is far from the fit (a launch whose day
    ramped nothing like the rest) are downweighted, so one wild
    analogue cannot drag the hour's shape.  Returns None when fewer
    than five pool members carry this hour.
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
    for _ in range(2):
        residuals = [y - (mean_y + slope * (x - mean_x))
                     for x, y in zip(xs, ys)]
        ordered = sorted(abs(r) for r in residuals)
        mad = ordered[len(ordered) // 2] or 0.1
        cutoff = 1.345 * 1.4826 * mad
        weights = [1.0 if abs(r) <= cutoff else cutoff / abs(r)
                   for r in residuals]
        total = sum(weights)
        mean_x = sum(w * x for w, x in zip(weights, xs)) / total
        mean_y = sum(w * y for w, y in zip(weights, ys)) / total
        denom = sum(w * (x - mean_x) ** 2
                    for w, x in zip(weights, xs))
        slope = 0.0
        if denom > 0:
            slope = sum(w * (x - mean_x) * (y - mean_y)
                        for w, x, y in zip(weights, xs, ys)) / denom
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
    forecast_rows = case.get("forecast")
    hours = []
    seen = set()
    if isinstance(forecast_rows, list):
        for row in forecast_rows:
            if not isinstance(row, dict):
                continue
            hour = row.get("hour")
            if type(hour) is not int or hour in seen:
                continue
            seen.add(hour)
            hours.append(hour)
    # The scored shape is exactly hours 2..24, each once.  Anything
    # else in the case file (a duplicate, hour 1, an out-of-range
    # hour) falls back to the full set rather than emitting bad rows.
    if sorted(hours) != list(range(2, 25)):
        hours = list(range(2, 25))

    p1 = None
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if is_number(viewers) and viewers > 0:
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
            if not (is_number(viewers) and viewers > 0):
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
        rows.append({"hour": hour, "q": format_quantiles(quantiles)})
    return {"hours": rows}


def fallback_forecast(case):
    """Last-resort forecast: persistence from hour 1, valid shape.

    Used only when the model path fails for a case; never raises, so
    a graded run always leaves an output file for every input.
    """
    p1 = 1000.0
    if isinstance(case, dict):
        for row in case.get("known") or []:
            if isinstance(row, dict) and row.get("hour") == 1:
                viewers = row.get("viewers")
                if is_number(viewers) and viewers > 0:
                    p1 = float(viewers)
                    break
    z_values = [norm_ppf(level)
                for level in (0.05, 0.25, 0.5, 0.75, 0.95)]
    return {"hours": [
        {"hour": h,
         "q": format_quantiles(persistence_quantiles(p1, h, z_values))}
        for h in range(2, 25)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        print(f"no *.in files in {INPUT_DIR}", flush=True)
    for path in inputs:
        case = None
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not a JSON object")
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never crash a graded run
            print(f"{path.name}: model failed ({exc}); "
                  f"writing persistence fallback", flush=True)
            forecast = fallback_forecast(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
