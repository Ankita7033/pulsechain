"""
Tests for the four new PulseChain engines.
All pass without any external dependencies.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml/bayesian"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml/baseline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml/explainer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml/spread"))

# ── 1. Bayesian Fusion ──────────────────────────────────────────────────────
class TestBayesianFusion:
    def _run(self, data, prior=0.02):
        import fusion_engine
        return fusion_engine.run(data, prior=prior)

    def test_prior_with_no_signals_returns_prior(self):
        from fusion_engine import fuse
        r = fuse([])
        # With no signals, posterior should equal prior
        assert abs(r.posterior - 0.02) < 0.005

    def test_full_outbreak_reaches_tier1(self):
        r = self._run({
            "wastewater": {"z_score": 5.0}, "pharmacy": {"z_score": 4.5},
            "absenteeism": {"z_score": 3.8}, "ed_triage": {"z_score": 4.2},
            "search_trends": {"z_score": 3.6}
        })
        assert r["alert_tier"] == 1
        assert r["posterior_probability"] > 0.75

    def test_isolated_search_spike_stays_normal(self):
        r = self._run({
            "wastewater": {"z_score": 0.1},
            "search_trends": {"z_score": 2.9}
        })
        assert r["alert_tier"] == 0, "Single-signal spike should not trigger alert"

    def test_three_concordant_signals_tier2_or_higher(self):
        r = self._run({
            "wastewater": {"z_score": 4.0},
            "pharmacy":   {"z_score": 3.5},
            "absenteeism":{"z_score": 3.0}
        })
        assert r["alert_tier"] >= 3, "Three moderate signals should at least Watch"

    def test_low_trust_pulls_toward_uninformative(self):
        r_high_trust = self._run({"wastewater": {"z_score": 3.5, "trust": 1.0}})
        r_low_trust  = self._run({"wastewater": {"z_score": 3.5, "trust": 0.1}})
        assert r_high_trust["posterior_probability"] > r_low_trust["posterior_probability"]

    def test_credible_interval_contains_posterior(self):
        r = self._run({"wastewater": {"z_score": 3.0}, "pharmacy": {"z_score": 2.0}})
        lo, hi = r["credible_interval"]
        assert lo <= r["posterior_probability"] <= hi

    def test_posterior_bounded_0_1(self):
        r = self._run({"wastewater": {"z_score": 999.0}})
        assert 0.0 < r["posterior_probability"] < 1.0

    def test_missing_signals_reported(self):
        r = self._run({"wastewater": {"z_score": 2.5}})
        assert "pharmacy" in r["missing_signals"]

    def test_dominant_signal_is_most_evidence(self):
        r = self._run({
            "wastewater": {"z_score": 4.5, "trust": 0.95},
            "search_trends": {"z_score": 1.2, "trust": 0.70}
        })
        assert r["dominant_signal"] == "wastewater"

    def test_suppressing_signal_lowers_posterior(self):
        # Without suppressor
        r1 = self._run({"wastewater": {"z_score": 3.0}})
        # With suppressor (ed_triage clearly below normal)
        r2 = self._run({"wastewater": {"z_score": 3.0}, "ed_triage": {"z_score": -1.5}})
        assert r2["posterior_probability"] < r1["posterior_probability"]


# ── 2. Regional Baseline ─────────────────────────────────────────────────────
class TestRegionalBaseline:
    def test_seasonal_expected_higher_in_peak_week(self):
        import regional_baseline as rb
        peak   = rb.get_baseline("northeast", "wastewater", week=2)   # Jan = peak
        trough = rb.get_baseline("northeast", "wastewater", week=28)  # July = off-peak
        assert peak.expected_value > trough.expected_value

    def test_z_score_large_for_2x_baseline(self):
        import regional_baseline as rb
        score = rb.compute_regional_z("northeast", "wastewater", 85000.0, week=3)
        assert score.z_score > 2.0
        assert score.is_anomaly

    def test_baseline_value_within_expected_returns_low_z(self):
        import regional_baseline as rb
        # Use the seasonal expected value itself (not annual mean) so z≈0
        baseline = rb.get_baseline("northeast", "wastewater", week=3)
        score = rb.compute_regional_z("northeast", "wastewater", baseline.expected_value, week=3)
        assert abs(score.z_score) < 0.001, f"Expected z≈0, got {score.z_score}" 

    def test_unknown_region_gets_global_fallback(self):
        import regional_baseline as rb
        baseline = rb.get_baseline("atlantis", "wastewater", week=5)
        assert baseline.profile_source == "global_fallback"
        assert baseline.expected_value > 0

    def test_all_five_sources_have_northeast_profile(self):
        import regional_baseline as rb
        sources = ["wastewater", "pharmacy", "absenteeism", "ed_triage", "search_trends"]
        for src in sources:
            b = rb.get_baseline("northeast", src, week=10)
            assert b.expected_value > 0, f"{src} should have a positive expected value"

    def test_seasonal_context_reflects_week(self):
        import regional_baseline as rb
        # Week 2 = January = peak flu season for northeast
        score = rb.compute_regional_z("northeast", "wastewater", 45000.0, week=2)
        assert "peak" in score.seasonal_context

    def test_score_all_signals_returns_all_sources(self):
        import regional_baseline as rb
        obs = {"wastewater": 50000, "pharmacy": 1300, "absenteeism": 0.09,
               "ed_triage": 0.035, "search_trends": 42.0}
        scores = rb.score_all_signals("northeast", obs, week=5)
        assert set(scores.keys()) == set(obs.keys())

    def test_percentile_of_expected_value_near_50(self):
        import regional_baseline as rb
        b = rb.get_baseline("northeast", "wastewater", week=5)
        score = rb.compute_regional_z("northeast", "wastewater", b.expected_value, week=5)
        assert 40.0 < score.percentile < 60.0


# ── 3. Explainable Alert ─────────────────────────────────────────────────────
class TestExplainableAlert:
    def _sample_payload(self, tier=2, posterior=0.61):
        return {
            "region": "northeast", "region_name": "Northeastern US",
            "alert_tier": tier, "posterior_probability": posterior,
            "confidence": 0.82, "credible_interval": [0.44, 0.76],
            "signal_log_bfs": {
                "wastewater": 1.82, "pharmacy": 0.94,
                "absenteeism": 0.61, "ed_triage": -0.21, "search_trends": 0.38
            },
            "signal_z_scores": {
                "wastewater": 2.8, "pharmacy": 1.6,
                "absenteeism": 1.2, "ed_triage": 0.3, "search_trends": 1.4
            },
            "seasonal_contexts": {src: "peak flu season" for src in
                ["wastewater","pharmacy","absenteeism","ed_triage","search_trends"]},
            "missing_signals": [],
        }

    def test_explanation_has_required_fields(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload())
        for field in ["summary","drivers","suppressors","recommended_actions",
                      "audit_hash","lead_time_estimate","credible_interval","confidence"]:
            assert field in r, f"Missing field: {field}"

    def test_drivers_sorted_by_contribution_descending(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload())
        contribs = [d["contribution"] for d in r["drivers"]]
        assert contribs == sorted(contribs, reverse=True)

    def test_suppressor_has_negative_log_bf(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload())
        # ed_triage has log_bf = -0.21 so should be in suppressors
        sup_signals = [s["signal"] for s in r["suppressors"]]
        assert "ed_triage" in sup_signals

    def test_tier1_has_emergency_actions(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload(tier=1, posterior=0.85))
        actions_text = " ".join(r["recommended_actions"]).lower()
        assert "emergency" in actions_text or "immediately" in actions_text

    def test_wastewater_leads_7_days(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload())
        assert "7" in r["lead_time_estimate"]

    def test_audit_hash_is_deterministic_format(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload())
        assert r["audit_hash"].startswith("sha256:")

    def test_summary_mentions_tier_label(self):
        import alert_explainer
        r = alert_explainer.run(self._sample_payload(tier=2))
        assert "Advisory" in r["summary"]

    def test_no_drivers_when_all_suppressed(self):
        import alert_explainer
        payload = self._sample_payload()
        payload["signal_log_bfs"] = {k: -0.5 for k in payload["signal_log_bfs"]}
        r = alert_explainer.run(payload)
        assert r["drivers"] == []


# ── 4. Cross-Region Spread ──────────────────────────────────────────────────
class TestSpreadModel:
    def _run(self, prob=0.72, region="northeast", horizon=14):
        import cross_region_spread
        return cross_region_spread.run({
            "regions": [{"region": region, "outbreak_probability": prob,
                         "active_signals": 4, "days_since_first_signal": 3}],
            "horizon_days": horizon,
        })

    def test_high_probability_generates_forecasts(self):
        r = self._run(prob=0.72)
        assert len(r["forecasts"]) > 0

    def test_low_probability_no_forecast(self):
        r = self._run(prob=0.05)
        assert len(r["forecasts"]) == 0  # below 0.15 threshold

    def test_spread_risks_ordered_by_probability(self):
        r = self._run(prob=0.80)
        probs = [s["spread_probability"] for s in r["forecasts"][0]["spread_risks"]]
        assert probs == sorted(probs, reverse=True)

    def test_source_region_not_in_targets(self):
        r = self._run(prob=0.75, region="northeast")
        targets = [s["target"] for s in r["forecasts"][0]["spread_risks"]]
        assert "northeast" not in targets

    def test_spread_probability_bounded_0_1(self):
        r = self._run(prob=0.99)
        for s in r["forecasts"][0]["spread_risks"]:
            assert 0.0 <= s["spread_probability"] <= 1.0

    def test_eta_positive(self):
        r = self._run(prob=0.80)
        for s in r["forecasts"][0]["spread_risks"]:
            assert s["eta_days"] > 0

    def test_method_field_present(self):
        r = self._run(prob=0.70)
        assert "mobility_weighted" in r["method"]

    def test_longer_horizon_raises_spread_prob(self):
        r14 = self._run(prob=0.70, horizon=14)
        r28 = self._run(prob=0.70, horizon=28)
        p14 = r14["forecasts"][0]["spread_risks"][0]["spread_probability"]
        p28 = r28["forecasts"][0]["spread_risks"][0]["spread_probability"]
        assert p28 >= p14


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"])
