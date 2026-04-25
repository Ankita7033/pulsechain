"""
PulseChain API — Enhanced Risk Scoring Routes
Integrates all four new engines into unified endpoints.
"""
import sys, os
sys.path.insert(0, "/app/ml")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

router = APIRouter(prefix="/v1", tags=["fusion"])

# ── Request / Response models ──────────────────────────────────────────────────

class SignalInput(BaseModel):
    z_score: float
    trust: float = 1.0
    freshness: float = 1.0

class FusionRequest(BaseModel):
    region: str
    region_name: Optional[str] = None
    signals: Dict[str, SignalInput]
    prior: float = Field(0.02, ge=0.001, le=0.5)
    observed_values: Optional[Dict[str, float]] = None  # raw values for baseline
    explain: bool = True
    forecast_spread: bool = True
    horizon_days: int = 14

class RegionStateInput(BaseModel):
    region: str
    outbreak_probability: float
    active_signals: int = 0
    days_since_first_signal: int = 0

class SpreadRequest(BaseModel):
    regions: List[RegionStateInput]
    horizon_days: int = 14


# ── Unified fusion endpoint ───────────────────────────────────────────────────

@router.post("/fuse")
async def full_fusion(req: FusionRequest):
    """
    Master endpoint: runs all four engines in sequence and returns
    a complete structured response.

    Called by n8n workflow 05 (Anomaly Detector) on every anomaly event.
    """
    import importlib, sys

    # Lazy imports so the module works without all deps installed
    def _import(path):
        spec = importlib.util.spec_from_file_location("mod", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # ── Step 1: Regional baseline z-scores ───────────────────────────
    baseline_mod = _import(os.path.join(base_path, "ml/baseline/regional_baseline.py"))
    from datetime import date
    week = date.today().isocalendar()[1]

    regional_scores = {}
    seasonal_contexts = {}

    if req.observed_values:
        regional_scores = baseline_mod.score_all_signals(
            req.region, req.observed_values, week=week
        )
        # Override z_scores from regional baseline (more accurate than global)
        for src, score in regional_scores.items():
            if src in req.signals.__root__ if hasattr(req.signals, '__root__') else req.signals:
                pass  # keep trust/freshness from request, update z_score
        seasonal_contexts = {src: v["seasonal_context"] for src, v in regional_scores.items()}

    # Build signal_data for Bayesian fusion using regional z-scores when available
    signal_data = {}
    for src, inp in req.signals.items():
        z = regional_scores.get(src, {}).get("z_score", inp.z_score) if regional_scores else inp.z_score
        signal_data[src] = {"z_score": z, "trust": inp.trust, "freshness": inp.freshness}

    # ── Step 2: Bayesian fusion ───────────────────────────────────────
    bayesian_mod = _import(os.path.join(base_path, "ml/bayesian/fusion_engine.py"))
    fusion_result = bayesian_mod.run(signal_data, prior=req.prior, region=req.region)

    # ── Step 3: Explainable alert ─────────────────────────────────────
    explanation = None
    if req.explain and fusion_result["alert_tier"] > 0:
        explainer_mod = _import(os.path.join(base_path, "ml/explainer/alert_explainer.py"))
        z_for_explainer = {src: d["z_score"] for src, d in signal_data.items()}
        explanation = explainer_mod.run({
            "region":                req.region,
            "region_name":           req.region_name or req.region,
            "alert_tier":            fusion_result["alert_tier"],
            "posterior_probability": fusion_result["posterior_probability"],
            "confidence":            fusion_result["confidence"],
            "credible_interval":     fusion_result["credible_interval"],
            "signal_log_bfs":        fusion_result["signal_log_bfs"],
            "signal_z_scores":       z_for_explainer,
            "seasonal_contexts":     seasonal_contexts,
            "missing_signals":       fusion_result["missing_signals"],
        })

    # ── Step 4: Spread forecast ───────────────────────────────────────
    spread = None
    if req.forecast_spread and fusion_result["alert_tier"] > 0:
        spread_mod = _import(os.path.join(base_path, "ml/spread/cross_region_spread.py"))
        spread = spread_mod.run({
            "regions": [
                {"region": req.region,
                 "outbreak_probability": fusion_result["posterior_probability"],
                 "active_signals": len([s for s, d in signal_data.items() if d["z_score"] >= 2.0]),
                 "days_since_first_signal": 0}
            ],
            "horizon_days": req.horizon_days,
        })

    return {
        "region":             req.region,
        "bayesian_fusion":    fusion_result,
        "regional_baselines": regional_scores if regional_scores else None,
        "explanation":        explanation,
        "spread_forecast":    spread,
        "pipeline_version":   "2.0",
    }


# ── Individual engine endpoints (for testing / n8n sub-calls) ─────────────────

@router.post("/fuse/bayesian")
async def bayesian_only(region: str, signals: Dict[str, SignalInput], prior: float = 0.02):
    """Run just the Bayesian fusion step."""
    import importlib.util, os
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    spec = importlib.util.spec_from_file_location("bf", os.path.join(base, "ml/bayesian/fusion_engine.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = {src: {"z_score": inp.z_score, "trust": inp.trust, "freshness": inp.freshness}
            for src, inp in signals.items()}
    return mod.run(data, prior=prior, region=region)


@router.post("/fuse/explain")
async def explain_only(payload: dict):
    """Run just the explainer step (pass fusion output directly)."""
    import importlib.util, os
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    spec = importlib.util.spec_from_file_location("exp", os.path.join(base, "ml/explainer/alert_explainer.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.run(payload)


@router.post("/fuse/spread")
async def spread_only(req: SpreadRequest):
    """Run just the spread forecast step."""
    import importlib.util, os
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    spec = importlib.util.spec_from_file_location("sp", os.path.join(base, "ml/spread/cross_region_spread.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.run({
        "regions": [r.dict() for r in req.regions],
        "horizon_days": req.horizon_days,
    })


@router.get("/fuse/baseline/{region}")
async def baseline_for_region(region: str, week: Optional[int] = None):
    """Return current seasonal baselines for a region across all 5 signals."""
    import importlib.util, os
    from datetime import date as d_
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    spec = importlib.util.spec_from_file_location("bl", os.path.join(base, "ml/baseline/regional_baseline.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    wk   = week or d_.today().isocalendar()[1]
    return {
        src: {
            "expected": mod.get_baseline(region, src, wk).expected_value,
            "sigma":    mod.get_baseline(region, src, wk).sigma,
            "seasonal": mod.get_baseline(region, src, wk).seasonal_component,
            "week":     wk,
        }
        for src in ["wastewater", "pharmacy", "absenteeism", "ed_triage", "search_trends"]
    }
