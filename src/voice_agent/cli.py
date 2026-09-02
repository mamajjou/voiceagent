"""CLI entrypoint."""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path
import yaml

from rich.console import Console

from .audio import FileReplayAudioSource, MicrophoneAudioSource
from .nemo_client import NemoClient, ASRConfig
from .llm_client import LLMClient, LLMConfig
from .turn_manager import TurnManager
from .session import Session
from .telemetry import GPUTelemetry

console = Console()

def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config {p}: expected mapping")
    # minimal defaults so CLI doesn't crash on partial configs
    cfg.setdefault("audio", {}).setdefault("chunk_ms", 20)
    cfg.setdefault("audio", {}).setdefault("sample_rate", 16000)
    return cfg

def build_clients(args, cfg):
    asr_cfg = ASRConfig(
        host=cfg["asr"]["backend"]["host"],
        port=cfg["asr"]["backend"]["port"],
        model_path=cfg["asr"]["model"]["path"],
        gpu=cfg["asr"]["backend"]["gpu"],
        rnnt_right_context=cfg["asr"]["streaming"]["rnnt_right_context"],
        eou_ms=cfg["asr"]["endpointing"]["stop_history_eou_ms"],
        language=args.language or cfg["asr"].get("language", "en-US"),
        enable_endpointing=cfg["asr"]["endpointing"]["enable"],
        vad_based=cfg["asr"]["endpointing"].get("vad_based", False),
    )
    llm_cfg = LLMConfig(
        host=cfg["llm"]["host"],
        port=cfg["llm"]["port"],
        model=cfg["llm"].get("model_path", "Qwen3.8-27B"),
        context=cfg["llm"].get("context", 8192),
        system_prompt=cfg["llm"]["system_prompt"],
        temperature=cfg["llm"]["generation"]["temperature"],
        top_p=cfg["llm"]["generation"]["top_p"],
        top_k=cfg["llm"]["generation"]["top_k"],
        presence_penalty=cfg["llm"]["generation"]["presence_penalty"],
        max_tokens=cfg["llm"]["generation"]["max_tokens"],
        enable_thinking=cfg["llm"]["generation"]["enable_thinking"],
    )
    mock = args.mock or cfg.get("mock", False)
    asr = NemoClient(asr_cfg, mock_text=args.mock_text if mock else None)
    llm = LLMClient(llm_cfg, mock=mock or args.mock_llm)
    return asr, llm, asr_cfg, llm_cfg

def main():
    parser = argparse.ArgumentParser(description="Voice agent naive baseline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--source", choices=["file", "mic"], default="file")
    parser.add_argument("--audio", type=str, help="path to wav for file source")
    parser.add_argument("--manifest", type=str, help="jsonl manifest to replay")
    parser.add_argument("--language", type=str, default=None, help="en-US or de-DE")
    parser.add_argument("--realtime", type=float, default=1.0, help="realtime factor (1.0 = live, 0 = fast)")
    parser.add_argument("--mock", action="store_true", help="mock ASR/LLM for testing without GPU")
    parser.add_argument("--mock-llm", action="store_true", help="mock LLM only")
    parser.add_argument("--mock-text", type=str, default="Hello, this is a mock transcription for testing.")
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args()
    # UTF-8 handling: ensure German umlauts pass through; console may need utf-8
    # Python's open already uses utf-8 where we specify encoding; argparse preserves unicode

    cfg = load_config(args.config)
    asr, llm, asr_cfg, llm_cfg = build_clients(args, cfg)

    # Health checks
    async def check():
        print(f"[check] ASR {asr_cfg.host}:{asr_cfg.port} ...")
        ok = await asr.check_health()
        print(f"  ASR health: {ok}")
        print(f"[check] LLM {llm_cfg.host}:{llm_cfg.port} ...")
        ok2 = await llm.check_health()
        print(f"  LLM health: {ok2}")
    try:
        asyncio.run(check())
    except:
        pass

    if args.source == "file":
        if args.manifest:
            # multi-turn from manifest
            entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
            print(f"[cli] replaying {len(entries)} manifest entries")
            tm = TurnManager()
            # keep one LLM history across turns
            for entry in entries:
                audio = FileReplayAudioSource(entry["audio"], realtime_factor=args.realtime, chunk_ms=cfg["audio"]["chunk_ms"])
                sess = Session(
                    audio_source=audio,
                    asr_client=asr,
                    llm_client=llm,
                    turn_manager=tm,
                    log_path=Path(f"{args.runs_dir}/{entry['id']}/events.jsonl"),
                    language=entry.get("language", args.language or "en-US"),
                    reference_text=entry.get("reference_text"),
                    audio_id=entry["id"],
                )
                sess.run()
        elif args.audio:
            audio = FileReplayAudioSource(args.audio, realtime_factor=args.realtime, chunk_ms=cfg["audio"]["chunk_ms"])
            tm = TurnManager()
            sess = Session(
                audio_source=audio,
                asr_client=asr,
                llm_client=llm,
                turn_manager=tm,
                log_path=Path(f"{args.runs_dir}/single/events.jsonl"),
                language=args.language or cfg["asr"].get("language", "en-US"),
                reference_text=args.mock_text if args.mock else None,
                audio_id=Path(args.audio).stem,
            )
            sess.run()
        else:
            print("Provide --audio or --manifest for file source")
            parser.print_help()
    elif args.source == "mic":
        audio = MicrophoneAudioSource(chunk_ms=cfg["audio"]["chunk_ms"], sample_rate=cfg["audio"]["sample_rate"])
        tm = TurnManager()
        sess = Session(
            audio_source=audio,
            asr_client=asr,
            llm_client=llm,
            turn_manager=tm,
            log_path=Path(f"{args.runs_dir}/mic/events.jsonl"),
            language=args.language or cfg["asr"].get("language", "en-US"),
        )
        sess.run()

if __name__ == "__main__":
    main()
