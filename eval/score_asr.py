"""Score ASR: WER, stability, endpoint latency."""
import json, argparse
from pathlib import Path
import jiwer
import numpy as np

def stability_metrics(partials: list[str], final: str):
    # longest common prefix etc.
    def lcp(a,b):
        i=0
        for ca, cb in zip(a,b):
            if ca==cb: i+=1
            else: break
        return i
    lcps = [lcp(p, final) for p in partials]
    # normalized edit distance
    import Levenshtein
    eds = []
    for p in partials:
        try:
            eds.append(Levenshtein.distance(p, final)/max(len(final),1))
        except:
            eds.append(1.0)
    return {"mean_lcp": float(np.mean(lcps)) if lcps else 0, "final_lcp": lcps[-1] if lcps else 0, "mean_ed": float(np.mean(eds)) if eds else 1}

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
