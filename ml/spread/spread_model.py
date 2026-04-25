"""
PulseChain — Cross-Region Spread Modeling Engine
=================================================
Models outbreak propagation across connected regions using:

1. Mobility Graph       — weighted edges from transport/commute flow data
2. SEIR compartmental   — deterministic epidemic model per node
3. Risk propagation     — probability of outbreak arriving in neighboring regions
4. Spread timeline      — expected arrival day per downstream region
5. Intervention impact  — how containment in source region slows spread

Mathematical basis:
- SIR/SEIR model per region node
- Metapopulation coupling via mobility matrix M[i][j]
- Force of infection: lambda_i = beta * I_i/N_i + sum_j(m_ij * lambda_j)
- Arrival time estimate: tau_ij = log(N_j / epsilon) / (R_eff * gamma * m_ij)

Reference:
    Colizza et al. (2006) "The role of the airline transportation network
    in the prediction and predictability of global epidemics" PNAS
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pulsechain.spread")

# ── Mobility graph (normalised daily flow between regions) ────────────
# m[origin][dest] = fraction of origin population that travels to dest daily
# Derived from Census commute + airline data (synthetic here)
MOBILITY_MATRIX: Dict[str, Dict[str, float]] = {
    "northeast": {
        "southeast": 0.0042,   # ~1M/day NYC↔Atlanta air corridor
        "midwest":   0.0038,   # NYC↔Chicago
        "west":      0.0031,   # NYC↔LA
    },
    "southeast": {
        "northeast": 0.0039,
        "midwest":   0.0029,
        "west":      0.0025,
    },
    "midwest": {
        "northeast": 0.0041,
        "southeast": 0.0028,
        "west":      0.0033,
    },
    "west": {
        "northeast": 0.0034,
        "southeast": 0.0026,
        "midwest":   0.0035,
    },
}

# Region populations (millions)
REGION_POPULATION: Dict[str, float] = {
    "northeast": 57.2,
    "southeast": 92.0,
    "midwest":   68.5,
    "west":      78.8,
}

REGION_NAMES = {
    "northeast": "Northeastern US",
    "southeast": "Southeastern US",
    "midwest":   "Midwestern US",
    "west":      "Western US",
}

# Disease parameters (calibrated for seasonal influenza)
DISEASE_PARAMS = {
    "influenza": {
        "R0":           1.4,     # basic reproduction number
        "serial_interval": 3.0,  # days between successive cases
        "incubation":   1.5,     # days (SEIR exposed period)
        "infectious":   2.5,     # days (infectious period)
        "ifr":          0.001,   # infection fatality rate
    },
    "covid_variant": {
        "R0":           2.8,
        "serial_interval": 4.5,
        "incubation":   3.0,
        "infectious":   5.0,
        "ifr":          0.005,
    },
    "generic": {
        "R0":           1.6,
        "serial_interval": 4.0,
        "incubation":   2.0,
        "infectious":   3.0,
        "ifr":          0.002,
    },
}


@dataclass
class RegionState:
    region:     str
    population: float    # millions
    susceptible: float   # fraction 0-1
    exposed:    float    # fraction 0-1 (SEIR E compartment)
    infectious: float    # fraction 0-1
    recovered:  float    # fraction 0-1
    arrival_day: Optional[int] = None    # day outbreak arrived
    arrival_probability: float = 0.0    # P(outbreak arrives within horizon)
    peak_day:   Optional[int] = None
    peak_prevalence: float = 0.0
    at_risk:    bool = False

    @property
    def effective_R(self) -> float:
        """R_eff = R0 * S (susceptible fraction)."""
        return self.susceptible


@dataclass
class SpreadForecast:
    source_region:     str
    horizon_days:      int
    pathogen:          str
    generated_at:      str
    source_risk_score: float

    regional_forecasts: Dict[str, Dict] = field(default_factory=dict)
    spread_order:       List[str]       = field(default_factory=list)
    arrival_times:      Dict[str, int]  = field(default_factory=dict)  # region -> days
    peak_times:         Dict[str, int]  = field(default_factory=dict)
    intervention_impact: Dict[str, float] = field(default_factory=dict)
    narrative:          str = ""
    risk_matrix:        List[dict] = field(default_factory=list)


class SEIRSpreadModel:
    """
    Metapopulation SEIR model with mobility coupling.

    State: S, E, I, R per region
    Coupling: infectious individuals travel between regions
    """

    def __init__(self, pathogen: str = "generic"):
        self.params = DISEASE_PARAMS.get(pathogen, DISEASE_PARAMS["generic"])
        self.pathogen = pathogen

    def _seir_step(self, state: RegionState, dt: float = 1.0,
                   imported_force: float = 0.0) -> RegionState:
        """One time step of SEIR dynamics."""
        p = self.params
        beta  = p["R0"] / p["infectious"]
        sigma = 1.0 / p["incubation"]
        gamma = 1.0 / p["infectious"]

        S, E, I, R = state.susceptible, state.exposed, state.infectious, state.recovered
        N = 1.0  # working in fractions

        # Force of infection (local + imported)
        lam = beta * I + imported_force

        dS = -lam * S * dt
        dE = (lam * S - sigma * E) * dt
        dI = (sigma * E - gamma * I) * dt
        dR = gamma * I * dt

        new_S = max(0.0, S + dS)
        new_E = max(0.0, E + dE)
        new_I = max(0.0, I + dI)
        new_R = min(1.0, R + dR)

        # Re-normalise to handle floating point drift
        total = new_S + new_E + new_I + new_R
        if total > 0:
            new_S, new_E, new_I, new_R = (
                new_S/total, new_E/total, new_I/total, new_R/total
            )

        return RegionState(
            region=state.region,
            population=state.population,
            susceptible=new_S, exposed=new_E,
            infectious=new_I, recovered=new_R,
            arrival_day=state.arrival_day,
            arrival_probability=state.arrival_probability,
        )

    def _compute_imported_force(
        self, dest: str, states: Dict[str, RegionState],
        containment: float = 0.0
    ) -> float:
        """
        Force of infection imported from neighboring regions via mobility.
        containment (0-1): how much source region has reduced travel.
        """
        total_force = 0.0
        for origin, m_row in MOBILITY_MATRIX.items():
            if origin == dest:
                continue
            mobility_rate = m_row.get(dest, 0.0) * (1.0 - containment)
            if origin in states:
                total_force += mobility_rate * states[origin].infectious
        return total_force

    def _expected_arrival_day(self, source: str, dest: str,
                               source_infectious: float) -> Optional[int]:
        """
        Estimate days until first imported case arrives in dest region.
        Based on: tau = 1 / (mobility_rate * infectious_count)
        """
        mobility = MOBILITY_MATRIX.get(source, {}).get(dest, 0.0)
        if mobility == 0 or source_infectious == 0:
            return None
        pop = REGION_POPULATION.get(source, 50.0) * 1e6
        daily_travelers = mobility * pop * source_infectious
        if daily_travelers < 0.001:
            return None
        tau = 1.0 / daily_travelers
        return max(1, int(math.ceil(tau)))

    def run(
        self,
        source_region:     str,
        source_risk_score: float,
        horizon_days:      int = 21,
        seed_infectious:   float = 0.001,  # 0.1% of population infectious
        containment:       float = 0.0,    # 0=none, 1=full travel ban
    ) -> SpreadForecast:
        """
        Run metapopulation SEIR simulation.

        Returns spread forecast with arrival times, peak predictions,
        and risk matrix for all regions.
        """
        # Initialize states
        states: Dict[str, RegionState] = {}
        for region, pop in REGION_POPULATION.items():
            if region == source_region:
                # Source: seeded with outbreak
                states[region] = RegionState(
                    region=region, population=pop,
                    susceptible=1.0 - seed_infectious,
                    exposed=seed_infectious * 0.3,
                    infectious=seed_infectious * 0.7,
                    recovered=0.0,
                    arrival_day=0,
                    arrival_probability=1.0,
                )
            else:
                # Destinations: susceptible, clean
                states[region] = RegionState(
                    region=region, population=pop,
                    susceptible=1.0, exposed=0.0,
                    infectious=0.0, recovered=0.0,
                )

        # Track time series
        time_series: Dict[str, List[float]] = {r: [] for r in states}
        arrival_days: Dict[str, Optional[int]] = {r: None for r in states}
        peak_days:    Dict[str, Optional[int]] = {r: None for r in states}
        peak_prev:    Dict[str, float]          = {r: 0.0  for r in states}

        arrival_days[source_region] = 0

        # Simulate day by day
        for day in range(horizon_days):
            # Compute imported forces for all destinations
            import_forces = {
                dest: self._compute_imported_force(dest, states, containment)
                for dest in states
            }

            new_states = {}
            for region, state in states.items():
                imported = import_forces.get(region, 0.0)
                new_state = self._seir_step(state, dt=1.0, imported_force=imported)
                new_states[region] = new_state

                # Track arrival
                if arrival_days[region] is None and new_state.exposed > 1e-6:
                    arrival_days[region] = day

                # Track peak
                if new_state.infectious > peak_prev[region]:
                    peak_prev[region]  = new_state.infectious
                    peak_days[region]  = day

                time_series[region].append(round(new_state.infectious * 100, 4))

            states = new_states

        # Compute arrival probabilities (P(arrival within horizon))
        arrival_probs = {}
        for region, ad in arrival_days.items():
            if region == source_region:
                arrival_probs[region] = 1.0
            elif ad is not None:
                # Earlier arrival = higher probability
                arrival_probs[region] = round(1.0 - ad / horizon_days, 3)
            else:
                # Estimate from mobility
                simple_arr = self._expected_arrival_day(
                    source_region, region,
                    states[source_region].infectious
                )
                if simple_arr and simple_arr <= horizon_days:
                    arrival_probs[region] = round(
                        max(0.1, 1.0 - simple_arr / horizon_days), 3
                    )
                else:
                    arrival_probs[region] = 0.1

        # Intervention impact: how much containment slows arrival
        intervention_impact = {}
        for dest in states:
            if dest == source_region:
                continue
            mob = MOBILITY_MATRIX.get(source_region, {}).get(dest, 0.0)
            # Fraction of spread risk eliminated by containment
            impact = containment * mob / max(sum(MOBILITY_MATRIX.get(source_region, {}).values()), 1e-9)
            intervention_impact[dest] = round(impact, 3)

        # Sort regions by arrival day
        spread_order = sorted(
            [r for r in states if r != source_region],
            key=lambda r: (arrival_days.get(r) or 999)
        )

        # Build regional forecast dicts
        regional_forecasts = {}
        for region, state in states.items():
            regional_forecasts[region] = {
                "region_name":        REGION_NAMES.get(region, region),
                "arrival_day":        arrival_days.get(region),
                "arrival_probability":arrival_probs.get(region, 0),
                "peak_day":           peak_days.get(region),
                "peak_prevalence_pct":round(peak_prev.get(region, 0) * 100, 3),
                "final_infectious_pct":round(states[region].infectious * 100, 4),
                "infectious_timeseries": time_series[region],
                "is_source":          region == source_region,
                "containment_reduces_risk_by": intervention_impact.get(region, 0),
            }

        # Risk matrix
        risk_matrix = [
            {
                "region": r,
                "region_name": REGION_NAMES.get(r, r),
                "arrival_day": arrival_days.get(r),
                "arrival_probability": arrival_probs.get(r, 0),
                "peak_day":    peak_days.get(r),
                "peak_pct":    round(peak_prev.get(r, 0) * 100, 3),
                "is_source":   r == source_region,
            }
            for r in [source_region] + spread_order
        ]

        narrative = self._build_narrative(
            source_region, spread_order, arrival_days, arrival_probs,
            peak_days, peak_prev, horizon_days, containment, states
        )

        return SpreadForecast(
            source_region=source_region,
            horizon_days=horizon_days,
            pathogen=self.pathogen,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_risk_score=source_risk_score,
            regional_forecasts=regional_forecasts,
            spread_order=spread_order,
            arrival_times={r: d for r, d in arrival_days.items() if d is not None},
            peak_times={r: d for r, d in peak_days.items() if d is not None},
            intervention_impact=intervention_impact,
            narrative=narrative,
            risk_matrix=risk_matrix,
        )

    def _build_narrative(self, source, order, arrivals, probs,
                         peaks, peak_prev, horizon, containment, states) -> str:
        p = self.params
        src_name = REGION_NAMES.get(source, source)
        lines = [
            f"Spread model ({self.pathogen}) seeded in {src_name}.",
            f"R0={p['R0']}, serial interval={p['serial_interval']}d, "
            f"horizon={horizon}d, containment={containment:.0%}.",
            "",
            "Projected spread order:",
        ]
        for r in order:
            ad = arrivals.get(r)
            pr = probs.get(r, 0)
            pk = peaks.get(r)
            pp = peak_prev.get(r, 0) * 100
            rname = REGION_NAMES.get(r, r)
            if ad:
                lines.append(f"  Day {ad:2d}: {rname:<20} P(arrival)={pr:.0%}  "
                              f"peak ~day {pk or '?'} ({pp:.2f}% prevalence)")
            else:
                lines.append(f"  Day ??:  {rname:<20} P(arrival)={pr:.0%}  "
                              f"(low connectivity)")
        if containment > 0:
            lines += ["",
                      f"With {containment:.0%} travel containment from {src_name}:"]
            for r in order:
                rname = REGION_NAMES.get(r, r)
                impact = states.get(r, None)
                lines.append(f"  {rname}: arrival risk reduced by ~"
                              f"{containment * 40:.0f}%")
        return "\n".join(lines)


# ── Module-level instance ─────────────────────────────────────────────
def run_spread_model(
    source_region:     str,
    source_risk_score: float,
    pathogen:          str = "influenza",
    horizon_days:      int = 21,
    containment:       float = 0.0,
) -> dict:
    """FastAPI / n8n entry point."""
    model    = SEIRSpreadModel(pathogen=pathogen)
    forecast = model.run(source_region, source_risk_score,
                         horizon_days=horizon_days,
                         seed_infectious=min(0.005, source_risk_score * 0.008),
                         containment=containment)
    return {
        "source_region":     forecast.source_region,
        "horizon_days":      forecast.horizon_days,
        "pathogen":          forecast.pathogen,
        "generated_at":      forecast.generated_at,
        "spread_order":      forecast.spread_order,
        "arrival_times":     forecast.arrival_times,
        "peak_times":        forecast.peak_times,
        "risk_matrix":       forecast.risk_matrix,
        "regional_forecasts":forecast.regional_forecasts,
        "intervention_impact":forecast.intervention_impact,
        "narrative":         forecast.narrative,
    }


if __name__ == "__main__":
    print("=== Cross-Region Spread Model Demo ===\n")

    for pathogen in ["influenza", "covid_variant"]:
        print(f"Pathogen: {pathogen.upper()}")
        result = run_spread_model(
            source_region="northeast",
            source_risk_score=0.72,
            pathogen=pathogen,
            horizon_days=21,
            containment=0.0,
        )
        print(f"  Spread order: {' → '.join(result['spread_order'])}")
        print(f"  Arrival times (days):")
        for row in result["risk_matrix"]:
            tag = " [SOURCE]" if row["is_source"] else ""
            print(f"    {row['region_name']:<22} "
                  f"arrive=day {str(row['arrival_day'] or '?'):<4} "
                  f"P={row['arrival_probability']:.0%}  "
                  f"peak=day {str(row['peak_day'] or '?'):<4} "
                  f"({row['peak_pct']:.3f}%){tag}")
        print()

    # With containment
    print("Effect of 60% travel containment from Northeast:")
    result_contain = run_spread_model(
        source_region="northeast",
        source_risk_score=0.72,
        pathogen="influenza",
        horizon_days=21,
        containment=0.60,
    )
    for row in result_contain["risk_matrix"]:
        if not row["is_source"]:
            no_contain_row = next((r for r in run_spread_model(
                "northeast", 0.72, "influenza", 21, 0.0
            )["risk_matrix"] if r["region"] == row["region"]), None)
            baseline_ad = no_contain_row["arrival_day"] if no_contain_row else "?"
            print(f"  {row['region_name']:<22} "
                  f"arrival: day {baseline_ad or '?'} → day {row['arrival_day'] or '?'} "
                  f"(containment effect)")
