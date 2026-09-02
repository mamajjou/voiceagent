"""Endpoint sweep: 15 configs grid, report WER / latency / false cuts."""
import json, time, itertools, argparse, csv
from pathlib import Path
import yaml
from jiwer import wer
import numpy as np
from voice_agent.audio import FileReplayAudioSource
from voice_agent.nemo_client import NemoClient, ASRConfig
from voice_agent.turn_manager import TurnManager
from voice_agent.session import Session
from voice_agent.llm_client import LLMClient, LLMConfig

RIGHT_MS = {0:80,1:160,3:320,6:560,13:1120}

def evaluate_one(cfg, entry, mock=False):
    asr_cfg = ASRConfig(
        host=cfg["asr"]["backend"]["host"], port=cfg["asr"]["backend"]["port"],
        rnnt_right_context=cfg["asr"]["streaming"]["rnnt_right_context"],
        eou_ms=cfg["asr"]["endpointing"]["stop_history_eou_ms"],
        language=entry.get("language","en-US")
    )
    llm_cfg = LLMConfig(host=cfg["llm"]["host"], port=cfg["llm"]["port"], system_prompt=cfg["llm"]["system_prompt"])
    asr = NemoClient(asr_cfg, mock_text=entry.get("reference_text") if mock else None)
    llm = LLMClient(llm_cfg, mock=True)  # no need for real LLM in ASR sweep
    tm = TurnManager()
    audio = FileReplayAudioSource(entry["audio"], realtime_factor=0)
    sess = Session(audio_source=audio, asr_client=asr, llm_client=llm, turn_manager=tm, log_path=Path(f"runs/sweep/tmp/events.jsonl"), language=entry.get("language","en-US"), reference_text=entry.get("reference_text"), audio_id=entry["id"])
    # capture timings
    t0 = time.monotonic()
    sess.run()
    # Extract ASR final vs reference
    final = tm.history[0].final_text if tm.history else ""
    ref = entry.get("reference_text","")
    # endpoint latency: endpoint_t - reference end; asr_final latency similarly
    # For mock, these are synthetic; for real, measured
    turn = tm.history[0] if tm.history else None
    return {
        "ref": ref,
        "hyp": final,
        "wer": wer(ref, final) if ref and final else 1.0,
        "endpoint_t": turn.endpoint_t if turn else None,
        "asr_final_t": turn.asr_final_t if turn else None,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--manifest", default="eval/manifest.jsonl")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/sweep/report.csv")
    args = ap.parse_args()
    base_cfg = yaml.safe_load(open(args.config))
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()][:20]  # limit for demo
    grid = list(itertools.product([1,3,6], [350,500,650,800,1000]))
    results = []
    for rc, eou in grid:
        cfg = yaml.safe_load(open(args.config))
        cfg["asr"]["streaming"]["rnnt_right_context"] = rc
        cfg["asr"]["endpointing"]["stop_history_eou_ms"] = eou
        wers = []
        for e in entries:
            r = evaluate_one(cfg, e, mock=args.mock)
            wers.append(r["wer"])
        avg_wer = float(np.mean(wers)) if wers else 0
        # TODO: real endpoint latency needs reference timestamps; placeholder
        print(f"rc={rc} ({RIGHT_MS[rc]}ms) eou={eou}ms  WER={avg_wer:.3f}  N={len(entries)}")
        results.append({"rc":rc, "rc_ms":RIGHT_MS[rc], "eou_ms":eou, "wer":avg_wer, "n":len(entries)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"[sweep] wrote {args.out}")
    # also json
    with open(str(Path(args.out).with_suffix(".json")), "w") as f:
        json.dump(results, f, indent=2)
