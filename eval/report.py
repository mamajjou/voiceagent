"""Generate REPORT.md from runs/ + sweep output.

Aggregates run summaries and sweep CSV into a Markdown report with the
latency waterfall (speech_end -> EOU -> ASR final -> Qwen TTFT) and stats.
"""
import json, glob, statistics, argparse, csv
from pathlib import Path
from datetime import datetime
import numpy as np

def collect_runs(runs_dir="runs"):
    summaries = []
    for p in Path(runs_dir).rglob("summary.json"):
        try: summaries.append(json.loads(open(p).read()))
        except: pass
    return summaries

def load_sweep(csv_path):
    rows = []
    if Path(csv_path).exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    return rows

def build_report(sweep_rows, summaries):
    now = datetime.now().isoformat()
    out = ["# Voice Agent Baseline Report", "", f"Generated {now}", ""]
    out += [f"Iterations: {len(summaries)} runs | Sweep rows: {len(sweep_rows)}", ""]

    # Latency waterfall from sweep
    if sweep_rows:
        out += ["## Endpointing sweep (speech_end -> ASR final -> Qwen TTFT)", ""]
        out += ["| rc | lookahead | EOU | ASR_final_med | ASR_final_p90 | TTFT_med | TTFT_p90 | WER |",
                "|-----|------|-----|------|------|------|------|------|"]
        for r in sweep_rows:
            out.append(f"| {r.get('rc')} | {r.get('rc_ms')}ms | {r.get('eou_ms')}ms | "
                       f"{r.get('asr_final_median_ms') or '-'} | {r.get('asr_final_p90_ms') or '-'} | "
                       f"{r.get('ttft_median_ms') or '-'} | {r.get('ttft_p90_ms') or '-'} | "
                       f"{(r.get('wer_mean') or '-')} |")
        out += [""]

    # VRAM from summaries
    vrams = [s.get("peak_vram_mb") for s in summaries if s.get("peak_vram_mb")]
    if vrams:
        out += ["## VRAM", "", f"- Peak median: **{statistics.median(vrams):.0f} MB**",
                f"- Peak p95: **{sorted(vrams)[int(len(vrams)*0.95)]:.0f} MB**", ""]

    # Per-run summary table
    if summaries:
        out += ["## Runs", ""]
        out += ["| session | audio | lang | ASR text | Qwen text | peak VRAM |", "|---|---|---|---|---|---|"]
        for s in summaries:
            out.append(f"| {s.get('session','')[:8]} | {s.get('audio_id','')} | {s.get('language','')} | "
                       f"`{(s.get('asr_final_text') or '')[:40]}` | `{(s.get('llm_text') or '')[:40]}` | "
                       f"{s.get('peak_vram_mb')} |")
        out += [""]

    out += ["## Notes", "", "- See `runs/*/events.jsonl` for per-event timestamps.",
            "- See `eval/score_asr.py` for WER / partial-stability analysis.",
            "- Waterfall: `speech_end -> EOU -> ASR final -> Qwen first token`."]
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--sweep-csv", default="runs/sweep/report.csv")
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()
    summaries = collect_runs(args.runs_dir)
    sweep_rows = load_sweep(args.sweep_csv)
    text = build_report(sweep_rows, summaries)
    Path(args.out).write_text(text)
    print(f"Wrote {args.out} ({len(summaries)} runs, {len(sweep_rows)} sweep rows)")
