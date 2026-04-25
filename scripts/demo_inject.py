#!/usr/bin/env python3
"""Inject synthetic outbreak signals for demo. Uses HTTP fallback if Kafka unavailable."""
import argparse, asyncio, hashlib, json, random, sys
from datetime import datetime, timezone

BASELINES = {
    "wastewater": {"mean": 42500, "std": 8200},
    "pharmacy": {"mean": 1240, "std": 310},
    "absenteeism": {"mean": 0.082, "std": 0.018},
    "ed_triage": {"mean": 0.031, "std": 0.009},
    "search_trends": {"mean": 38.2, "std": 9.1},
}
INTENSITY = {
    "low":    {"mult": 2.0, "sigs": ["wastewater"]},
    "medium": {"mult": 3.5, "sigs": ["wastewater","pharmacy","search_trends"]},
    "high":   {"mult": 6.0, "sigs": list(BASELINES.keys())},
}
REGIONS = {"northeast":"Northeastern US","southeast":"Southeastern US","midwest":"Midwestern US","west":"Western US"}

async def inject(region, intensity, duration, interval, api_url):
    cfg = INTENSITY[intensity]
    regions = list(REGIONS.keys()) if region == "all" else [region]
    steps = max(1, duration // interval)
    print(f"\n{'='*52}\n  PulseChain v2 Outbreak Injector\n{'='*52}")
    print(f"  Pathogen region: {region}  Intensity: {intensity}  Mult: ×{cfg['mult']}")
    print(f"  Signals: {', '.join(cfg['sigs'])}\n  Endpoints: {api_url}\n")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            for r in regions:
                for step in range(steps):
                    p = step / max(steps-1,1)
                    wave = (1+(cfg["mult"]-1)*(p/.4) if p<.4 else cfg["mult"] if p<.7 else cfg["mult"]-(cfg["mult"]-1)*((p-.7)/.3))
                    ts = datetime.now(timezone.utc).isoformat()
                    for src in cfg["sigs"]:
                        b = BASELINES[src]
                        val = max(0, b["mean"] * wave * (1+random.gauss(0,.07)))
                        z = (val - b["mean"]) / b["std"]
                        eid = hashlib.sha256(f"{src}:{r}:{ts}".encode()).hexdigest()
                        try:
                            resp = await client.post(f"{api_url}/v1/score", json={
                                "source":src,"region":r,"z_score":z,"severity_score":max(0,z),
                                "event_id":eid,"observed_value":round(val,4)
                            })
                            tier = resp.json().get("alert_tier","?") if resp.status_code==200 else "err"
                            bar = "█"*int((step/steps)*30)+"░"*(30-int((step/steps)*30))
                            print(f"\r  [{bar}] {src:12s} z={z:+.2f} → tier {tier}", end="", flush=True)
                        except Exception as e:
                            print(f"\r  HTTP error: {e}", end="", flush=True)
                    await asyncio.sleep(interval)
        print(f"\n\n  ✅ Done! Check http://localhost:8080 for live updates.")
        if intensity=="high": print("  🚨 Expect Tier 1 Emergency alert in n8n + dashboard")
        elif intensity=="medium": print("  ⚠️  Expect Tier 2 Advisory alert")
        else: print("  👁  Expect Tier 3 Watch mode")
    except ImportError:
        print("Install httpx: pip install httpx")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region",    default="northeast", choices=list(REGIONS.keys())+["all"])
    p.add_argument("--pathogen",  default="flu")
    p.add_argument("--intensity", default="high",      choices=["low","medium","high"])
    p.add_argument("--duration",  default=60,  type=int)
    p.add_argument("--interval",  default=5,   type=int)
    p.add_argument("--api-url",   default="http://localhost:8000")
    args = p.parse_args()
    asyncio.run(inject(args.region, args.intensity, args.duration, args.interval, args.api_url))

if __name__ == "__main__":
    main()
