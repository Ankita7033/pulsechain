"""
PulseChain — Explainable Alert Generator  (Missing piece #3)
============================================================
Every alert must explain itself. Public-health officials will not act on
a black-box score. This module produces a structured, human-readable
explanation of exactly which signals triggered an alert, by how much,
and why the system responded the way it did.

Output contract (all alerts must include this block):
  {
    "summary":    "Tier 2 Advisory — 3 signals elevated above seasonal baseline",
    "drivers": [
      {"signal": "wastewater",  "z_score": +2.8, "contribution": 0.42, "context": "peak flu season"},
      {"signal": "pharmacy",    "z_score": +1.6, "contribution": 0.28, "context": "..."},
      {"signal": "absenteeism", "z_score": +1.2, "contribution": 0.19, "context": "..."}
    ],
    "suppressors": [
      {"signal": "ed_triage", "z_score": +0.3, "note": "clinical signals not yet elevated"}
    ],
    "missing_signals": ["search_trends"],
    "confidence":  0.82,
    "credible_interval": [0.61, 0.94],
    "lead_time_estimate": "wastewater leads clinical by ~7 days",
    "recommended_action": "Increase ED surveillance sampling; notify regional health dept",
    "audit_hash": "sha256:abc123..."
  }
"""

from __future__ import annotations
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pulsechain.explainer")

# ── Signal metadata used in explanations ────────────────────────────────────
SIGNAL_META = {
    "wastewater": {
        "label":        "Wastewater viral load",
        "lead_days":    7,
        "description":  "Environmental RNA surveillance — leads clinical cases by ~7 days",
        "unit":         "copies/mL",
    },
    "pharmacy": {
        "label":        "Pharmacy antipyretic sales",
        "lead_days":    4,
        "description":  "Bulk OTC fever-reducer purchases — proxy for symptomatic illness burden",
        "unit":         "purchase index",
    },
    "absenteeism": {
        "label":        "School/workplace absenteeism",
        "lead_days":    3,
        "description":  "Reported absence rates above seasonal expected — early community spread",
        "unit":         "fraction absent",
    },
    "ed_triage": {
        "label":        "ED chief-complaint keyword density",
        "lead_days":    2,
        "description":  "ILI-related keywords in triage notes — near-real-time clinical signal",
        "unit":         "keyword rate",
    },
    "search_trends": {
        "label":        "Internet symptom search volume",
        "lead_days":    5,
        "description":  "Population symptom searches — very early behavioural signal, lower specificity",
        "unit":         "trend score",
    },
}

TIER_LABELS = {0: "Normal", 1: "Tier 1 — Emergency", 2: "Tier 2 — Advisory", 3: "Tier 3 — Watch"}

RECOMMENDED_ACTIONS = {
    1: [
        "Immediately notify state/federal health authority",
        "Activate emergency response protocol",
        "Dispatch field investigation team",
        "Initiate public communication plan",
        "Increase all signal sampling to 5-minute intervals",
    ],
    2: [
        "Notify regional health department within 2 hours",
        "Increase ED surveillance sampling frequency",
        "Alert hospital infection-control teams",
        "Begin enhanced wastewater sampling at upstream sites",
        "Prepare public advisory draft",
    ],
    3: [
        "Flag region for heightened monitoring",
        "Cross-verify with lab confirmation data",
        "Schedule check-in review within 24 hours",
        "Notify local epidemiology team (non-urgent)",
    ],
    0: [
        "Continue routine monitoring",
    ],
}


@dataclass
class SignalDriver:
    signal: str
    label: str
    z_score: float
    contribution_fraction: float   # fraction of total evidence from this signal
    log_bf: float                  # raw Bayes factor from fusion engine
    direction: str                 # "elevating" | "suppressing" | "neutral"
    seasonal_context: str
    lead_days: int
    description: str


@dataclass
class AlertExplanation:
    alert_id: str
    region: str
    region_name: str
    tier: int
    tier_label: str
    summary: str
    posterior_probability: float
    confidence: float
    credible_interval: Tuple[float, float]
    drivers: List[SignalDriver]           # signals raising the alert
    suppressors: List[SignalDriver]       # signals actively arguing against
    neutral: List[SignalDriver]           # signals present but uninformative
    missing_signals: List[str]
    lead_time_estimate: str
    recommended_actions: List[str]
    generated_at: str
    audit_hash: str                       # tamper-evident fingerprint of this explanation


def _format_z(z: float) -> str:
    sign = "+" if z >= 0 else ""
    return f"{sign}{z:.2f}σ"


def _lead_time_estimate(drivers: List[SignalDriver]) -> str:
    if not drivers:
        return "Insufficient signal data for lead-time estimation."
    earliest = min(drivers, key=lambda d: -d.lead_days)
    return (
        f"{earliest.label} is leading — "
        f"clinical confirmation expected in approximately {earliest.lead_days} days."
    )


def _audit_hash(region: str, tier: int, posterior: float, timestamp: str) -> str:
    raw = f"{region}:{tier}:{posterior:.4f}:{timestamp}"
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_explanation(
    region: str,
    region_name: str,
    tier: int,
    posterior: float,
    confidence: float,
    credible_interval: Tuple[float, float],
    signal_log_bfs: Dict[str, float],        # from Bayesian fusion engine
    signal_z_scores: Dict[str, float],       # from regional baseline model
    seasonal_contexts: Dict[str, str],       # from regional baseline model
    missing_signals: List[str],
) -> AlertExplanation:
    """
    Build a full structured explanation for an alert.
    Designed to be called directly after the Bayesian fusion step.
    """
    now = datetime.now(timezone.utc).isoformat()
    alert_id = _audit_hash(region, tier, posterior, now)

    # Classify signals into drivers / suppressors / neutral
    drivers: List[SignalDriver] = []
    suppressors: List[SignalDriver] = []
    neutral: List[SignalDriver] = []

    total_positive_lbf = sum(lbf for lbf in signal_log_bfs.values() if lbf > 0.05) or 1.0

    for source, log_bf in signal_log_bfs.items():
        meta = SIGNAL_META.get(source, {
            "label": source, "lead_days": 0,
            "description": "", "unit": ""
        })
        z = signal_z_scores.get(source, 0.0)
        ctx = seasonal_contexts.get(source, "")

        contrib = max(0.0, log_bf) / total_positive_lbf if log_bf > 0.05 else 0.0

        sd = SignalDriver(
            signal=source,
            label=meta["label"],
            z_score=z,
            contribution_fraction=round(contrib, 3),
            log_bf=log_bf,
            direction=("elevating" if log_bf > 0.05 else
                       "suppressing" if log_bf < -0.05 else "neutral"),
            seasonal_context=ctx,
            lead_days=meta["lead_days"],
            description=meta["description"],
        )
        if log_bf > 0.05:
            drivers.append(sd)
        elif log_bf < -0.05:
            suppressors.append(sd)
        else:
            neutral.append(sd)

    # Sort drivers by contribution (descending)
    drivers.sort(key=lambda d: -d.contribution_fraction)

    # Build human summary
    n_drivers = len(drivers)
    tier_lbl  = TIER_LABELS.get(tier, "Unknown")
    if n_drivers == 0:
        summary = f"{tier_lbl} — alert from prior risk elevation; no strong single-signal driver."
    elif n_drivers == 1:
        d = drivers[0]
        summary = (f"{tier_lbl} — {d.label} elevated {_format_z(d.z_score)} "
                   f"above regional seasonal baseline.")
    else:
        top = ", ".join(f"{d.label} {_format_z(d.z_score)}" for d in drivers[:3])
        summary = (f"{tier_lbl} — {n_drivers} signals elevated above regional baseline: {top}.")

    if suppressors:
        sup_names = ", ".join(s.label for s in suppressors)
        summary += f" Note: {sup_names} currently below expected — confidence moderated."

    lead_est = _lead_time_estimate(drivers)
    actions  = RECOMMENDED_ACTIONS.get(tier, RECOMMENDED_ACTIONS[0])

    return AlertExplanation(
        alert_id=alert_id,
        region=region,
        region_name=region_name,
        tier=tier,
        tier_label=tier_lbl,
        summary=summary,
        posterior_probability=posterior,
        confidence=confidence,
        credible_interval=credible_interval,
        drivers=drivers,
        suppressors=suppressors,
        neutral=neutral,
        missing_signals=missing_signals,
        lead_time_estimate=lead_est,
        recommended_actions=actions,
        generated_at=now,
        audit_hash=alert_id,
    )


def to_dict(explanation: AlertExplanation) -> dict:
    """Serialise to JSON-safe dict for API response and n8n output."""
    return {
        "alert_id":              explanation.alert_id,
        "region":                explanation.region,
        "region_name":           explanation.region_name,
        "tier":                  explanation.tier,
        "tier_label":            explanation.tier_label,
        "summary":               explanation.summary,
        "posterior_probability": explanation.posterior_probability,
        "confidence":            explanation.confidence,
        "credible_interval":     list(explanation.credible_interval),
        "drivers": [
            {
                "signal":       d.signal,
                "label":        d.label,
                "z_score":      d.z_score,
                "display":      _format_z(d.z_score),
                "contribution": d.contribution_fraction,
                "direction":    d.direction,
                "context":      d.seasonal_context,
                "lead_days":    d.lead_days,
            }
            for d in explanation.drivers
        ],
        "suppressors": [
            {
                "signal":  s.signal,
                "label":   s.label,
                "z_score": s.z_score,
                "note":    f"Currently {_format_z(s.z_score)} — argues against outbreak",
            }
            for s in explanation.suppressors
        ],
        "missing_signals":     explanation.missing_signals,
        "lead_time_estimate":  explanation.lead_time_estimate,
        "recommended_actions": explanation.recommended_actions,
        "generated_at":        explanation.generated_at,
        "audit_hash":          explanation.audit_hash,
    }


def run(payload: dict) -> dict:
    """FastAPI / n8n Code node entry point."""
    exp = build_explanation(
        region=payload.get("region", "unknown"),
        region_name=payload.get("region_name", payload.get("region", "Unknown")),
        tier=int(payload.get("alert_tier", 0)),
        posterior=float(payload.get("posterior_probability", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        credible_interval=tuple(payload.get("credible_interval", [0.0, 1.0])),
        signal_log_bfs=payload.get("signal_log_bfs", {}),
        signal_z_scores=payload.get("signal_z_scores", {}),
        seasonal_contexts=payload.get("seasonal_contexts", {}),
        missing_signals=payload.get("missing_signals", []),
    )
    return to_dict(exp)


if __name__ == "__main__":
    sample = {
        "region":                "northeast",
        "region_name":           "Northeastern US",
        "alert_tier":            2,
        "posterior_probability": 0.61,
        "confidence":            0.82,
        "credible_interval":     [0.44, 0.76],
        "signal_log_bfs": {
            "wastewater":    1.82,
            "pharmacy":      0.94,
            "absenteeism":   0.61,
            "ed_triage":    -0.21,
            "search_trends": 0.38,
        },
        "signal_z_scores": {
            "wastewater":    2.8,
            "pharmacy":      1.6,
            "absenteeism":   1.2,
            "ed_triage":     0.3,
            "search_trends": 1.4,
        },
        "seasonal_contexts": {
            "wastewater":    "peak flu season",
            "pharmacy":      "peak flu season",
            "absenteeism":   "peak flu season",
            "ed_triage":     "peak flu season",
            "search_trends": "peak flu season",
        },
        "missing_signals": [],
    }
    result = run(sample)
    print(f"\n{'='*60}")
    print(f"ALERT: {result['tier_label']}")
    print(f"{'='*60}")
    print(f"{result['summary']}\n")
    print("Drivers:")
    for d in result["drivers"]:
        bar = "█" * int(d["contribution"] * 20)
        print(f"  {d['label']:35s}  {d['display']:>8}  [{bar:<20}] {d['contribution']:.0%}")
    if result["suppressors"]:
        print("\nSuppressors:")
        for s in result["suppressors"]:
            print(f"  {s['label']:35s}  {s['z_score']:+.2f}σ  {s['note']}")
    print(f"\nConfidence:   {result['confidence']:.0%}")
    print(f"90% CI:       [{result['credible_interval'][0]:.0%} – {result['credible_interval'][1]:.0%}]")
    print(f"Lead time:    {result['lead_time_estimate']}")
    print(f"\nRecommended actions:")
    for a in result["recommended_actions"]:
        print(f"  • {a}")
    print(f"\nAudit hash:   {result['audit_hash']}")
