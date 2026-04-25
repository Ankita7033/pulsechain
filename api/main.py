"""
PulseChain FastAPI Backend
Provides synthetic data endpoints + Bayesian fusion + Enhanced analysis.
All endpoints called by the n8n workflow.
"""

import random
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="PulseChain API",
    description="Epidemiological Signal Processing API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REGIONS = ["northeast", "southeast", "midwest", "west"]
REGION_NAMES = {
    "northeast": "Northeastern US",
    "southeast": "Southeastern US",
    "midwest":   "Midwestern US",
    "west":      "Western US",
}


# ─── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ─── Synthetic Data Sources ─────────────────────────────────────────────────

def _ts():
    return datetime.now(timezone.utc).isoformat()


@app.get("/v1/synthetic/pharmacy")
def synthetic_pharmacy(region: str = "all"):
    regions = REGIONS if region == "all" else [region]
    data = []
    for r in regions:
        # Occasionally spike a region to trigger anomaly detection
        base = random.uniform(40, 60)
        if random.random() < 0.1:
            base = random.uniform(75, 95)
        data.append({
            "source":         "pharmacy",
            "region":         r,
            "region_name":    REGION_NAMES.get(r, r),
            "purchase_index": round(base, 2),
            "timestamp":      _ts(),
        })
    return {"data": data, "count": len(data)}


@app.get("/v1/synthetic/search_trends")
def synthetic_search_trends(region: str = "all"):
    regions = REGIONS if region == "all" else [region]
    data = []
    for r in regions:
        base = random.uniform(40, 60)
        if random.random() < 0.1:
            base = random.uniform(70, 90)
        data.append({
            "source":       "search_trends",
            "region":       r,
            "region_name":  REGION_NAMES.get(r, r),
            "search_index": round(base, 2),
            "timestamp":    _ts(),
        })
    return {"data": data, "count": len(data)}


@app.get("/v1/synthetic/absenteeism")
def synthetic_absenteeism(region: str = "all"):
    regions = REGIONS if region == "all" else [region]
    data = []
    for r in regions:
        base = random.uniform(0.05, 0.12)
        if random.random() < 0.1:
            base = random.uniform(0.18, 0.30)
        data.append({
            "source":            "absenteeism",
            "region":            r,
            "region_name":       REGION_NAMES.get(r, r),
            "absenteeism_rate":  round(base, 4),
            "timestamp":         _ts(),
        })
    return {"data": data, "count": len(data)}


@app.get("/v1/synthetic/ed_triage")
def synthetic_ed_triage(region: str = "all"):
    regions = REGIONS if region == "all" else [region]
    data = []
    for r in regions:
        base = random.uniform(35, 55)
        if random.random() < 0.1:
            base = random.uniform(65, 85)
        data.append({
            "source":       "ed_triage",
            "region":       r,
            "region_name":  REGION_NAMES.get(r, r),
            "triage_index": round(base, 2),
            "timestamp":    _ts(),
        })
    return {"data": data, "count": len(data)}


# ─── Bayesian Fusion Engine ─────────────────────────────────────────────────

class FuseRequest(BaseModel):
    region:          Optional[str] = "unknown"
    region_name:     Optional[str] = None
    signals:         Optional[Dict[str, Any]] = {}
    prior:           Optional[float] = 0.02
    observed_values: Optional[Dict[str, float]] = {}
    explain:         Optional[bool] = True
    forecast_spread: Optional[bool] = True
    horizon_days:    Optional[int] = 14


@app.post("/v1/fuse")
def fuse(req: FuseRequest):
    """Simple Bayesian fusion of multi-source signals."""
    observed = req.observed_values or {}
    n_signals = len(observed)
    if n_signals == 0:
        posterior = req.prior
    else:
        # Naive Bayes: multiply likelihood ratios
        log_odds = math.log(req.prior / (1 - req.prior))
        for source, value in observed.items():
            # Each signal above threshold adds positive log-odds
            if value > 60:
                log_odds += 1.5
            elif value > 40:
                log_odds += 0.3
        posterior = 1 / (1 + math.exp(-log_odds))
        posterior = max(0.0, min(0.99, posterior))

    alert_tier = (
        1 if posterior >= 0.75 else
        2 if posterior >= 0.50 else
        3 if posterior >= 0.25 else 0
    )
    tier_labels = {0: "NORMAL", 1: "EMERGENCY", 2: "ADVISORY", 3: "WATCH"}

    return {
        "region":                req.region,
        "region_name":           req.region_name or req.region,
        "posterior_probability": round(posterior, 4),
        "alert_tier":            alert_tier,
        "tier_label":            tier_labels[alert_tier],
        "confidence":            0.75 if n_signals > 1 else 0.60,
        "dominant_signal":       max(observed, key=observed.get) if observed else "unknown",
        "drivers": [
            {"source": s, "value": v, "label": s.replace("_", " ").title(), "display": f"{v:.1f}"}
            for s, v in observed.items()
        ],
        "recommended_actions": _actions(alert_tier),
        "explanation_summary": (
            f"{tier_labels[alert_tier]} signal detected in {req.region_name or req.region}. "
            f"Posterior outbreak probability: {posterior*100:.1f}%."
        ),
        "explanation": {
            "summary": f"Bayesian fusion across {n_signals} signal(s) yields {posterior*100:.1f}% outbreak probability.",
            "markdown": f"## Fusion Result\n**Posterior:** {posterior*100:.1f}%\n**Tier:** {tier_labels[alert_tier]}"
        },
        "fusion": {
            "method":        "naive_bayes",
            "log_odds_ratio": round(math.log((posterior + 1e-9) / (1 - posterior + 1e-9)), 4),
        },
        "spread_forecast": _spread(req.region, posterior) if req.forecast_spread else None,
        "lead_time_estimate": "3-7 days",
        "assessed_at": _ts(),
    }


# ─── Enhanced Analysis ──────────────────────────────────────────────────────

class EnhancedRequest(BaseModel):
    region:          Optional[str] = "unknown"
    region_name:     Optional[str] = None
    signal_values:   Optional[Dict[str, float]] = {}
    signals:         Optional[Dict[str, Any]] = {}
    z_score:         Optional[float] = 0
    source:          Optional[str] = "unknown"
    signal_value:    Optional[float] = 0
    prior:           Optional[float] = 0.02
    horizon_days:    Optional[int] = 14
    pathogen:        Optional[str] = "influenza"


@app.post("/v1/enhanced/analyze")
def enhanced_analyze(req: EnhancedRequest):
    """Enhanced analysis — wraps fusion with additional context."""
    observed = req.signal_values or {}
    if req.source and req.signal_value:
        observed[req.source] = req.signal_value

    fuse_req = FuseRequest(
        region=req.region,
        region_name=req.region_name,
        signals=req.signals,
        prior=req.prior,
        observed_values=observed,
        explain=True,
        forecast_spread=True,
        horizon_days=req.horizon_days,
    )
    base = fuse(fuse_req)

    # Enrich with pathogen-specific context
    base["pathogen"] = req.pathogen
    base["bayesian_fusion"] = {
        "posterior_probability": base["posterior_probability"],
        "alert_tier":           base["alert_tier"],
        "confidence":           base["confidence"],
        "dominant_signal":      base["dominant_signal"],
        "drivers":              base["drivers"],
        "recommended_actions":  base["recommended_actions"],
        "explanation_summary":  base["explanation_summary"],
        "spread_forecast":      base["spread_forecast"],
        "credible_interval":    [
            round(max(0, base["posterior_probability"] - 0.15), 4),
            round(min(1, base["posterior_probability"] + 0.15), 4),
        ],
    }
    return base


# ─── Risk Score (legacy endpoint) ──────────────────────────────────────────

@app.get("/v1/score/{region}")
def risk_score(region: str):
    score = round(random.uniform(0.1, 0.9), 4)
    return {
        "region":     region,
        "risk_score": score,
        "timestamp":  _ts(),
    }


# ─── Helpers ────────────────────────────────────────────────────────────────

def _actions(tier: int) -> List[str]:
    return {
        1: ["Activate emergency response protocol", "Notify regional health authority", "Deploy rapid response team"],
        2: ["Monitor situation closely", "Prepare response resources", "Brief health officials"],
        3: ["Continue surveillance", "Review trending indicators", "Alert on-call epidemiologist"],
        0: ["Maintain routine monitoring"],
    }.get(tier, ["Maintain routine monitoring"])


def _spread(region: str, posterior: float) -> Dict:
    neighbors = {
        "northeast": ["midwest", "southeast"],
        "southeast": ["northeast", "midwest"],
        "midwest":   ["northeast", "west", "southeast"],
        "west":      ["midwest"],
    }.get(region, [])
    return {
        "spread_risks": [
            {
                "target":            n,
                "spread_probability": round(posterior * random.uniform(0.3, 0.6), 4),
                "eta_days":          random.randint(3, 10),
            }
            for n in neighbors
        ]
    }