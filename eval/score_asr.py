"""Score ASR: WER, stability, endpoint latency."""
import json, argparse
from pathlib import Path
import jiwer
import numpy as np

def _lcp(a: str, b: str) -> int:
    """Longest common prefix length of two strings."""
    i = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            i += 1
        else:
            break
    return i

def _ned(a: str, b: str) -> float:
    """Normalized edit distance (jiwer) in [0, 1]."""
    if not b:
        return 0.0 if not a else 1.0
    try:
        return float(jiwer.cer(a, b))
    except Exception:
        return 1.0

def stability_metrics(partials: list[str], final: str, partial_times=None):
    """Partial-transcript stability statistics.

    Args:
        partials: sequence of partial hypothesis strings (growing).
        final: the committed final transcript.
        partial_times: optional monotonic timestamps aligned with partials.

    Returns:
        dict with mean/final LCP ratio, mean normalized edit distance,
        number of revisions, and (if times given) time-to-stability metrics.
    """
    lcps = [_lcp(p, final) for p in partials]
    neds = [_ned(p, final) for p in partials]
    if not partials:
        return {
            "n_partials": 0, "mean_lcp_ratio": 0.0, "final_lcp_ratio": 0.0,
            "mean_ned": 1.0, "n_revisions": 0, "first_stable_time": None,
            "revision_rate": 0.0,
        }
    denom = max(len(final), 1)
    mean_lcp_ratio = float(np.mean(lcps)) / denom
    final_lcp_ratio = float(lcps[-1]) / denom
    mean_ned = float(np.mean(neds))
    # Revisions: number of times the partial text changed (non-monotonic, word refit)
    revisions = sum(1 for i in range(1, len(partials)) if partials[i] != partials[i-1])
    # Time to stability: first time the final's words are 'stably' present. Approximate:
    # the timestamp when the longest-common-prefix ratio first reaches 90% of final.
    first_stable_time = None
    if partial_times:
        for t, lp in zip(partial_times, lcps):
            if denom and lp / denom >= 0.9:
                first_stable_time = float(t)
                break
    revision_rate = revisions / max(len(partials), 1)
    return {
        "n_partials": len(partials), "mean_lcp_ratio": round(mean_lcp_ratio, 4),
        "final_lcp_ratio": round(final_lcp_ratio, 4), "mean_ned": round(mean_ned, 4),
        "n_revisions": revisions, "first_stable_time": first_stable_time,
        "revision_rate": round(revision_rate, 4),
    }

if __name__ == "__main__":
    # Self-test
    final = "What is the capital of France?"
    partials = ["What", "What is", "What is the", "What is the capital", "What is the capital of France"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/manifest.jsonl")
    ap.add_argument("--runs-dir", default="runs/replay")
    args = ap.parse_args()
    entries = {json.loads(l)["id"]: json.loads(l) for l in open(args.manifest) if l.strip()}
    wers = []
    for id, entry in entries.items():
        run = Path(args.runs_dir) / id / "events.jsonl"
        if not run.exists():
            continue
        events = [json.loads(l) for l in open(run) if l.strip()]
        finals = [e["text"] for e in events if e["event"]=="asr_final"]
        if not finals: continue
        hyp = finals[-1]
        ref = entry.get("reference_text","")
        if ref:
            w = jiwer.wer(ref, hyp)
            wers.append(w)
            print(f"{id}: WER={w:.3f} ref='{ref[:60]}' hyp='{hyp[:60]}'")
    if wers:
        print(f"Average WER: {np.mean(wers):.3f} median {np.median(wers):.3f} p90 {np.percentile(wers,90):.3f}")
