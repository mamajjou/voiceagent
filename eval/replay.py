"""Replay manifest through full pipeline and log metrics."""
import json, argparse, time
from pathlib import Path
import yaml
from voice_agent.audio import FileReplayAudioSource
from voice_agent.nemo_client import NemoClient, ASRConfig
from voice_agent.llm_client import LLMClient, LLMConfig
from voice_agent.turn_manager import TurnManager
from voice_agent.session import Session

def load_cfg(path):
    import yaml
    return yaml.safe_load(open(path))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--manifest", default="eval/manifest.jsonl")
    ap.add_argument("--runs-dir", default="runs/replay")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--realtime", type=float, default=0.0, help="0=fast, 1.0=realtime")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    asr_cfg = ASRConfig(host=cfg["asr"]["backend"]["host"], port=cfg["asr"]["backend"]["port"], rnnt_right_context=cfg["asr"]["streaming"]["rnnt_right_context"], eou_ms=cfg["asr"]["endpointing"]["stop_history_eou_ms"], language=cfg["asr"].get("language","en-US"))
    llm_cfg = LLMConfig(host=cfg["llm"]["host"], port=cfg["llm"]["port"], system_prompt=cfg["llm"]["system_prompt"], temperature=cfg["llm"]["generation"]["temperature"])
    asr = NemoClient(asr_cfg, mock_text=None if not args.mock else "mock")
    llm = LLMClient(llm_cfg, mock=args.mock)
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        entries = entries[:args.limit]
    print(f"[replay] {len(entries)} entries, realtime={args.realtime}, mock={args.mock}")
    for e in entries:
        audio = FileReplayAudioSource(e["audio"], realtime_factor=args.realtime)
        tm = TurnManager()
        # for mock, set per-entry mock text
        if args.mock:
            asr.mock_text = e.get("reference_text","")
        sess = Session(audio_source=audio, asr_client=asr, llm_client=llm, turn_manager=tm, log_path=Path(f"{args.runs_dir}/{e['id']}/events.jsonl"), language=e.get("language","en-US"), reference_text=e.get("reference_text"), audio_id=e["id"])
        sess.run()
