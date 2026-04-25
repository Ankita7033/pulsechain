"""
PulseChain — Regional Baseline Modeling Layer  (Missing piece #2)
=================================================================
Learns district-level seasonal baselines independently per region+source.
Detects anomalies relative to expected regional behaviour, not global averages.

Why this matters:
  - Rural Montana has very different "normal" wastewater viral load than NYC
  - Flu season peaks 3-4 weeks later in the south vs northeast
  - A 20% increase is noise in some regions, alarming in others

Architecture:
  1. BaselineRecord  - stores (mean, std, seasonal_pattern) per (region, source, week)
  2. SeasonalProfile - 52-week sinusoidal model fit from historical data
  3. RegionalBaseline - compute region-specific z-scores and expected ranges

The seasonal model:
  expected(t) = mu + A * sin(2π(t - φ) / 52)
  where A = amplitude, φ = phase offset (week of annual peak)
"""

from __future__ import annotations
import math
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

logger = logging.getLogger("pulsechain.baseline")


# ── Pre-fit seasonal parameters per (region, source) ──────────────────────────
# In production these come from the DB (fitted on 2+ years of historical data).
# Format: {source: {region: {mu, sigma, amplitude, phase_week}}}
SEASONAL_PROFILES: Dict[str, Dict[str, Dict]] = {
    "wastewater": {
        "northeast": {"mu": 42500, "sigma": 8200,  "amplitude": 11000, "phase_week": 2},
        "southeast": {"mu": 38100, "sigma": 7400,  "amplitude": 8500,  "phase_week": 5},
        "midwest":   {"mu": 40200, "sigma": 7800,  "amplitude": 9800,  "phase_week": 3},
        "west":      {"mu": 35600, "sigma": 6900,  "amplitude": 7200,  "phase_week": 7},
    },
    "pharmacy": {
        "northeast": {"mu": 1240,  "sigma": 310,   "amplitude": 320,   "phase_week": 2},
        "southeast": {"mu": 1100,  "sigma": 280,   "amplitude": 260,   "phase_week": 5},
        "midwest":   {"mu": 1180,  "sigma": 295,   "amplitude": 290,   "phase_week": 3},
        "west":      {"mu": 1050,  "sigma": 265,   "amplitude": 220,   "phase_week": 7},
    },
    "absenteeism": {
        "northeast": {"mu": 0.082, "sigma": 0.018, "amplitude": 0.022, "phase_week": 2},
        "southeast": {"mu": 0.075, "sigma": 0.016, "amplitude": 0.018, "phase_week": 5},
        "midwest":   {"mu": 0.079, "sigma": 0.017, "amplitude": 0.020, "phase_week": 3},
        "west":      {"mu": 0.071, "sigma": 0.015, "amplitude": 0.015, "phase_week": 7},
    },
    "ed_triage": {
        "northeast": {"mu": 0.031, "sigma": 0.009, "amplitude": 0.009, "phase_week": 2},
        "southeast": {"mu": 0.028, "sigma": 0.008, "amplitude": 0.007, "phase_week": 5},
        "midwest":   {"mu": 0.029, "sigma": 0.008, "amplitude": 0.008, "phase_week": 3},
        "west":      {"mu": 0.026, "sigma": 0.007, "amplitude": 0.006, "phase_week": 7},
    },
    "search_trends": {
        "northeast": {"mu": 38.2, "sigma": 9.1,   "amplitude": 11.0,  "phase_week": 2},
        "southeast": {"mu": 34.8, "sigma": 8.6,   "amplitude": 9.5,   "phase_week": 5},
        "midwest":   {"mu": 36.5, "sigma": 8.9,   "amplitude": 10.2,  "phase_week": 3},
        "west":      {"mu": 32.1, "sigma": 8.0,   "amplitude": 8.1,   "phase_week": 7},
    },
}


@dataclass
class BaselineStats:
    region: str
    source: str
    week_of_year: int
    expected_value: float         # seasonal-adjusted expected value for this week
    sigma: float                  # expected standard deviation
    seasonal_component: float     # the seasonal add-on vs annual mean
    profile_source: str           # "fitted" | "synthetic" | "global_fallback"


@dataclass
class RegionalAnomalyScore:
    region: str
    source: str
    observed_value: float
    expected_value: float
    sigma: float
    z_score: float
    percentile: float             # where this observation falls in expected distribution
    is_anomaly: bool
    anomaly_direction: str        # "high" | "low" | "none"
    seasonal_context: str         # e.g. "peak flu season" / "off-season"
    week_of_year: int


def _week_of_year(dt: Optional[date] = None) -> int:
    if dt is None:
        dt = date.today()
    return dt.isocalendar()[1]


def _seasonal_expected(profile: dict, week: int) -> float:
    """
    Sinusoidal seasonal model:
      expected = mu + A * sin(2π * (week - phase) / 52)
    """
    mu        = profile["mu"]
    amplitude = profile["amplitude"]
    phase     = profile["phase_week"]
    seasonal  = amplitude * math.sin(2 * math.pi * (week - phase + 13) / 52.0)
    return mu + seasonal


def get_baseline(region: str, source: str, week: Optional[int] = None) -> BaselineStats:
    """
    Retrieve (or synthesise) a seasonal baseline for a given region+source+week.
    Falls back to global average if region has no fitted profile.
    """
    if week is None:
        week = _week_of_year()

    region_profiles = SEASONAL_PROFILES.get(source, {})
    profile = region_profiles.get(region)

    if profile:
        expected = _seasonal_expected(profile, week)
        seasonal_component = expected - profile["mu"]
        profile_src = "fitted"
        sigma = profile["sigma"]
    else:
        # Global fallback: average across all regions for this source
        all_profiles = list(SEASONAL_PROFILES.get(source, {}).values())
        if all_profiles:
            avg_mu  = sum(p["mu"]        for p in all_profiles) / len(all_profiles)
            avg_amp = sum(p["amplitude"] for p in all_profiles) / len(all_profiles)
            avg_phi = sum(p["phase_week"]for p in all_profiles) / len(all_profiles)
            avg_sig = sum(p["sigma"]     for p in all_profiles) / len(all_profiles)
            fake_profile = {"mu": avg_mu, "amplitude": avg_amp,
                            "phase_week": avg_phi, "sigma": avg_sig}
            expected = _seasonal_expected(fake_profile, week)
            seasonal_component = expected - avg_mu
            sigma = avg_sig
            profile_src = "global_fallback"
        else:
            expected = 0.0
            sigma = 1.0
            seasonal_component = 0.0
            profile_src = "synthetic"

    return BaselineStats(
        region=region,
        source=source,
        week_of_year=week,
        expected_value=round(expected, 4),
        sigma=round(sigma, 4),
        seasonal_component=round(seasonal_component, 4),
        profile_source=profile_src,
    )


def compute_regional_z(
    region: str,
    source: str,
    observed: float,
    week: Optional[int] = None,
) -> RegionalAnomalyScore:
    """
    Compute a region-aware z-score for an observed signal value.
    This is the main function called by the n8n anomaly detector.
    """
    if week is None:
        week = _week_of_year()

    baseline = get_baseline(region, source, week)
    sigma = max(baseline.sigma, 1e-9)
    z = (observed - baseline.expected_value) / sigma

    # Standard normal CDF approximation for percentile
    def std_norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    percentile = std_norm_cdf(z) * 100.0

    is_anomaly = abs(z) >= 2.0
    direction  = "high" if z >= 2.0 else ("low" if z <= -2.0 else "none")

    # Contextual season label
    peak_profile = SEASONAL_PROFILES.get(source, {}).get(region, {})
    peak_week    = peak_profile.get("phase_week", 2)
    weeks_from_peak = abs(week - peak_week)
    if weeks_from_peak > 26:
        weeks_from_peak = 52 - weeks_from_peak
    season_ctx = ("peak flu season"    if weeks_from_peak <= 4  else
                  "near peak season"   if weeks_from_peak <= 8  else
                  "off-season")

    return RegionalAnomalyScore(
        region=region,
        source=source,
        observed_value=round(observed, 6),
        expected_value=round(baseline.expected_value, 6),
        sigma=round(sigma, 6),
        z_score=round(z, 4),
        percentile=round(percentile, 2),
        is_anomaly=is_anomaly,
        anomaly_direction=direction,
        seasonal_context=season_ctx,
        week_of_year=week,
    )


def score_all_signals(
    region: str,
    observations: Dict[str, float],
    week: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Score all observed signals for a region at once.
    Returns a dict of {source: anomaly_score_dict} suitable for JSON serialisation.
    """
    results = {}
    for source, value in observations.items():
        score = compute_regional_z(region, source, value, week)
        results[source] = {
            "z_score":          score.z_score,
            "observed":         score.observed_value,
            "expected":         score.expected_value,
            "sigma":            score.sigma,
            "percentile":       score.percentile,
            "is_anomaly":       score.is_anomaly,
            "direction":        score.anomaly_direction,
            "seasonal_context": score.seasonal_context,
            "week":             score.week_of_year,
        }
    return results


if __name__ == "__main__":
    print("=== Regional Baseline Demo ===\n")
    observations = {
        "wastewater":    85000.0,   # 2× normal for northeast in winter
        "pharmacy":      1850.0,
        "absenteeism":   0.14,
        "ed_triage":     0.052,
        "search_trends": 61.0,
    }

    results = score_all_signals("northeast", observations, week=3)  # January = week 3
    for src, s in results.items():
        flag = "🚨" if s["is_anomaly"] else "  "
        print(f"  {flag} {src:15s}  z={s['z_score']:+.2f}  "
              f"obs={s['observed']:.3g}  exp={s['expected']:.3g}  "
              f"[{s['seasonal_context']}]")
