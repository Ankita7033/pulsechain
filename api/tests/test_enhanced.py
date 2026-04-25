"""Tests for all 4 enhanced engines."""
import sys, os
for p in ["ml/bayesian","ml/baseline","ml/explainer","ml/spread"]:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", p))

# Bayesian Fusion
def test_posterior_bounded():
    from fusion_engine import run_bayesian_fusion
    r = run_bayesian_fusion({"wastewater": {"z_score": 8.0, "trust_score": 1.0, "freshness": 1.0}})
    assert 0 < r["posterior_probability"] < 1

def test_false_alarm_suppressed():
    from fusion_engine import run_bayesian_fusion
    r = run_bayesian_fusion({
        "wastewater":   {"z_score": 0.1, "trust_score": 0.9, "freshness": 1.0},
        "pharmacy":     {"z_score": 0.2, "trust_score": 0.9, "freshness": 1.0},
        "search_trends":{"z_score": 2.9, "trust_score": 0.7, "freshness": 1.0},
    })
    assert r["posterior_probability"] < 0.20

def test_multi_signal_raises_posterior():
    from fusion_engine import run_bayesian_fusion
    r1 = run_bayesian_fusion({"wastewater":{"z_score":3.0,"trust_score":0.9,"freshness":1.0}})
    r5 = run_bayesian_fusion({s:{"z_score":3.0,"trust_score":0.9,"freshness":1.0}
                               for s in ["wastewater","pharmacy","absenteeism","ed_triage","search_trends"]})
    assert r5["posterior_probability"] > r1["posterior_probability"]

def test_low_trust_reduces_impact():
    from fusion_engine import run_bayesian_fusion
    rh = run_bayesian_fusion({"wastewater":{"z_score":4.0,"trust_score":0.95,"freshness":1.0}})
    rz = run_bayesian_fusion({"wastewater":{"z_score":4.0,"trust_score":0.0, "freshness":0.0}})
    assert rh["posterior_probability"] > rz["posterior_probability"]

def test_credible_interval_valid():
    from fusion_engine import run_bayesian_fusion
    r = run_bayesian_fusion({"wastewater":{"z_score":3.0,"trust_score":0.9,"freshness":1.0}})
    lo, hi = r["credible_interval"]
    assert lo < r["posterior_probability"] < hi

def test_missing_signals_reported():
    from fusion_engine import run_bayesian_fusion
    r = run_bayesian_fusion({"wastewater":{"z_score":2.0,"trust_score":0.9,"freshness":1.0}})
    assert len(r["missing_signals"]) == 4

# Regional Baseline
def test_peak_vs_offseason():
    from regional_baseline import get_regional_z_scores
    pk = get_regional_z_scores("northeast",{"wastewater":65000},doy=15, dow=1)
    of = get_regional_z_scores("northeast",{"wastewater":65000},doy=180,dow=1)
    assert abs(pk["wastewater"]["regional_z"]) < abs(of["wastewater"]["regional_z"])

def test_regional_differs_from_global():
    from regional_baseline import get_regional_z_scores
    s = get_regional_z_scores("northeast",{"wastewater":65000},doy=15,dow=1)
    assert s["wastewater"]["regional_z"] != s["wastewater"]["global_z"]

def test_weekend_suppression():
    from regional_baseline import RegionalBaselineStore
    store = RegionalBaselineStore()
    wd = store.get("northeast","absenteeism",doy=15,dow=1)
    we = store.get("northeast","absenteeism",doy=15,dow=6)
    assert we.seasonal_mean < wd.seasonal_mean

def test_drift_detection():
    from regional_baseline import BaselineDriftDetector
    d = BaselineDriftDetector(k=0.5, h=4.0)
    detected = False
    for _ in range(40):
        if d.update("northeast","wastewater",1.5)["drift_detected"]:
            detected = True; break
    assert detected

def test_all_sources_returned():
    from regional_baseline import get_regional_z_scores
    v = {"wastewater":42500,"pharmacy":1240,"absenteeism":0.082,"ed_triage":0.031,"search_trends":38.2}
    assert len(get_regional_z_scores("northeast",v)) == 5

# Alert Explainer
def test_explainer_markdown():
    from alert_explainer import generate_explanation
    fusion = {"alert_tier":2,"posterior_probability":0.62,"prior_probability":0.02,
              "confidence":0.78,"credible_interval":[0.50,0.74],"log_odds_ratio":3.5,
              "dominant_signal":"wastewater","signal_contributions":{},"bayes_factors":{}}
    e = generate_explanation("northeast","Northeastern US",fusion,{},{"wastewater":68000.0})
    assert "Signal Drivers" in e["markdown"] and len(e["markdown"]) > 100

def test_explainer_tier_actions():
    from alert_explainer import generate_explanation
    fusion = {"alert_tier":1,"posterior_probability":0.85,"prior_probability":0.02,
              "confidence":0.92,"credible_interval":[0.78,0.92],"log_odds_ratio":5.2,
              "dominant_signal":"wastewater","signal_contributions":{},"bayes_factors":{}}
    e = generate_explanation("northeast","Northeastern US",fusion,{},{"wastewater":85000.0})
    assert "Recommended Actions" in e["markdown"]

# Spread Model
def test_source_arrival_prob_1():
    from spread_model import run_spread_model
    r = run_spread_model("northeast",0.7,"influenza",14)
    src = next(x for x in r["risk_matrix"] if x["is_source"])
    assert src["arrival_probability"] == 1.0

def test_all_regions_present():
    from spread_model import run_spread_model
    r = run_spread_model("northeast",0.7,"influenza",14)
    assert {x["region"] for x in r["risk_matrix"]} == {"northeast","southeast","midwest","west"}

def test_covid_higher_than_flu():
    from spread_model import run_spread_model
    flu   = run_spread_model("northeast",0.8,"influenza",21)
    covid = run_spread_model("northeast",0.8,"covid_variant",21)
    sf = next(r for r in flu["risk_matrix"]   if r["is_source"])
    sc = next(r for r in covid["risk_matrix"] if r["is_source"])
    assert sc["peak_pct"] > sf["peak_pct"]

def test_spread_narrative_nonempty():
    from spread_model import run_spread_model
    r = run_spread_model("midwest",0.6,"generic",14)
    assert len(r["narrative"]) > 50
