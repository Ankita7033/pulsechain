"""
PulseChain — Enhanced Scoring API Routes
Exposes all 4 new engines via FastAPI endpoints.
These are called by n8n workflow nodes.
"""
from __future__ import annotations
import logging
import sys, os

# Add ml modules to path
for p in ["ml/bayesian","ml/baseline","ml/explainer","ml/spread"]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../", p))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

logger = logging.getLogger("pulsechain.api.enhanced")
router = APIRouter(prefix="/v1/enhanced", tags=["enhanced-scoring"])


# ── Request / Response models ─────────────────────────────────────────

class SignalBundle(BaseModel):
    """Signals keyed by source name, each with z_score and metadata."""
    region:       str
    region_name:  str
    signal_data:  Dict[str, Dict]   # {source: {z_score, trust_score, freshness}}
    signal_values: Optional[Dict[str, float]] = None  # raw values for baseline
    prior:        float = 0.02
    doy:          Optional[int] = None   # day of year (default: today)
    dow:          Optional[int] = None   # day of week (default: today)

class SpreadRequest(BaseModel):
    source_region:     str
    source_risk_score: float
    pathogen:          str = "influenza"
    horizon_days:      int = 21
    containment:       float = 0.0

class FullAnalysisRequest(BaseModel):
    """Single endpoint that runs all 4 engines in sequence."""
    region:        str
    region_name:   str
    signal_values: Dict[str, float]   # raw signal values
    trust_scores:  Optional[Dict[str, float]] = None
    pathogen:      str = "influenza"
    run_spread:    bool = True
    prior:         float = 0.02
    doy:           Optional[int] = None
    dow:           Optional[int] = None


# ── Endpoint 1: Bayesian Fusion only ─────────────────────────────────
@router.post("/fuse")
async def bayesian_fuse(req: SignalBundle):
    """
    Run Bayesian signal fusion on pre-computed z-scores.
    Called by n8n after anomaly detection to upgrade score from
    weighted-sum to posterior probability.
    """
    try:
        from fusion_engine import run_bayesian_fusion
        result = run_bayesian_fusion(req.signal_data, prior=req.prior, region=req.region)
        return result
    except Exception as e:
        logger.exception("Bayesian fusion error")
        raise HTTPException(500, f"Bayesian fusion failed: {e}")


# ── Endpoint 2: Regional baseline z-scores ───────────────────────────
@router.post("/baseline")
async def regional_baseline(req: SignalBundle):
    """
    Compute regionally- and seasonally-adjusted z-scores.
    Called by n8n before fusion to replace global z-scores with
    district-specific seasonal ones.
    """
    try:
        from regional_baseline import get_regional_z_scores
        if not req.signal_values:
            raise HTTPException(400, "signal_values required for baseline scoring")
        result = get_regional_z_scores(req.region, req.signal_values, req.doy, req.dow)
        return {"region": req.region, "scores": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Baseline scoring error")
        raise HTTPException(500, f"Baseline scoring failed: {e}")


# ── Endpoint 3: Alert explanation ────────────────────────────────────
@router.post("/explain")
async def explain_alert(
    region:          str,
    region_name:     str,
    fusion_result:   dict,
    baseline_scores: dict,
    signal_values:   Dict[str, float],
    trust_scores:    Optional[Dict[str, float]] = None,
):
    """
    Generate structured explanation for a fired alert.
    Called by n8n notification dispatch workflow to enrich alert payload.
    """
    try:
        from alert_explainer import generate_explanation
        result = generate_explanation(
            region, region_name, fusion_result,
            baseline_scores, signal_values, trust_scores
        )
        return result
    except Exception as e:
        logger.exception("Explanation generation error")
        raise HTTPException(500, f"Explanation failed: {e}")


# ── Endpoint 4: Spread forecast ───────────────────────────────────────
@router.post("/spread")
async def spread_forecast(req: SpreadRequest):
    """
    Run cross-region spread model from a detected outbreak source.
    Called by n8n after Tier 1/2 alert fires.
    """
    try:
        from spread_model import run_spread_model
        result = run_spread_model(
            source_region=req.source_region,
            source_risk_score=req.source_risk_score,
            pathogen=req.pathogen,
            horizon_days=req.horizon_days,
            containment=req.containment,
        )
        return result
    except Exception as e:
        logger.exception("Spread model error")
        raise HTTPException(500, f"Spread model failed: {e}")


# ── Endpoint 5: Full pipeline (all 4 engines in one call) ─────────────
@router.post("/analyze")
async def full_analysis(req: FullAnalysisRequest):
    """
    Runs the complete enhanced pipeline:
      1. Regional baseline → regional z-scores
      2. Bayesian fusion   → posterior probability + CI
      3. Alert explainer   → structured explanation
      4. Spread model      → propagation forecast (if tier >= 2)

    This is the primary endpoint called by n8n workflow 05 (anomaly detector)
    when a signal anomaly is detected. Replaces the simple /v1/score call
    with a full probabilistic + explainable assessment.
    """
    try:
        from regional_baseline import get_regional_z_scores
        from fusion_engine     import run_bayesian_fusion
        from alert_explainer   import generate_explanation
        from spread_model      import run_spread_model

        # ── Step 1: Regional baseline ─────────────────────────────
        baseline_scores = get_regional_z_scores(
            req.region, req.signal_values, req.doy, req.dow
        )

        # Build signal_data dict for fusion (regional z-scores + trust)
        trust = req.trust_scores or {}
        signal_data = {
            src: {
                "z_score":     baseline_scores[src]["regional_z"],
                "trust_score": trust.get(src, 0.9),
                "freshness":   1.0,
            }
            for src in req.signal_values
            if src in baseline_scores
        }

        # ── Step 2: Bayesian fusion ───────────────────────────────
        fusion = run_bayesian_fusion(signal_data, prior=req.prior, region=req.region)

        # ── Step 3: Explanation ────────────────────────────────────
        explanation = generate_explanation(
            region=req.region,
            region_name=req.region_name,
            fusion_result=fusion,
            baseline_scores=baseline_scores,
            signal_values=req.signal_values,
            trust_scores=trust,
        )

        # ── Step 4: Spread model (only if actionable tier) ────────
        spread = None
        if req.run_spread and fusion.get("alert_tier", 0) >= 2:
            spread = run_spread_model(
                source_region=req.region,
                source_risk_score=fusion["posterior_probability"],
                pathogen=req.pathogen,
                horizon_days=21,
                containment=0.0,
            )

        return {
            "region":           req.region,
            "region_name":      req.region_name,
            # Core outputs
            "alert_tier":       fusion["alert_tier"],
            "posterior_probability": fusion["posterior_probability"],
            "credible_interval": fusion["credible_interval"],
            "confidence":        fusion["confidence"],
            "dominant_signal":   fusion["dominant_signal"],
            # Full results
            "baseline":         baseline_scores,
            "fusion":           fusion,
            "explanation":      explanation,
            "spread_forecast":  spread,
            # Quick-access for n8n routing
            "drivers": [
                {
                    "source":     src,
                    "regional_z": baseline_scores[src]["regional_z"],
                    "is_anomaly": baseline_scores[src]["is_anomaly"],
                    "contribution": fusion["signal_contributions"].get(src, 0),
                    "is_peak_season": baseline_scores[src].get("is_peak_season", False),
                }
                for src in req.signal_values
                if src in baseline_scores
            ],
        }

    except Exception as e:
        logger.exception("Full analysis pipeline error")
        raise HTTPException(500, f"Full analysis failed: {e}")
