"""Generate REPORT.md from runs."""
import json, glob, statistics
from pathlib import Path
from datetime import datetime

def collect_runs(runs_dir="runs"):
    summaries = []
    for p in Path(runs_dir).rglob("summary.json"):
        try:
            summaries.append(json.loads(open(p).read()))
        except: pass
    events = []
    for p in Path(runs_dir).rglob("events.jsonl"):
        try:
            ev = [json.loads(l) for l in open(p) if l.strip()]
            events.extend(ev)
        except: pass
    return summaries, events

if __name__ == "__main__":
    summaries, events = collect_runs()
    out = Path("REPORT.md")
    with open(out, "w") as f:
        f.write(f"# Voice Agent Baseline Report\n\nGenerated {datetime.now().isoformat()}\n\n")
        f.write(f"## Summary\n\n- Runs: {len(summaries)}\n- Events: {len(events)}\n\n")
        if summaries:
            vrams = [s.get("peak_vram_mb") for s in summaries if s.get("peak_vram_mb")]
            if vrams:
                f.write(f"## VRAM\n\n- Peak median {statistics.median(vrams):.0f} MB\n- Peak p95 {sorted(vrams)[int(len(vrams)*0.95)]:.0f} MB\n\n")
        # latency from events
        # parse llm_first_token etc.
        f.write("## Latency\n\nSee per-run events.jsonl for breakdown: speech_end -> EOU -> ASR final -> LLM first token\n\n")
        f.write("## Recommendations\n\n- Run `python eval/sweep_endpointing.py` for 15-config grid\n- See `runs/sweep/report.csv`\n")
    print(f"Wrote {out}")
