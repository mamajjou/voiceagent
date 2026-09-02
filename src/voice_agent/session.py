"""Session orchestrator: AudioSource → ASR → TurnManager → LLM → logger."""
from __future__ import annotations
import time
import json
import uuid
import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

from .audio import AudioSource, FileReplayAudioSource
from .nemo_client import NemoClient, ASRConfig, ASRPartial
from .llm_client import LLMClient, LLMConfig
from .turn_manager import TurnManager, State
from .telemetry import GPUTelemetry, snapshot_vram, query_nvidia_smi, get_cuda_version, get_git_revision, file_hash

console = Console()

def _safe_int(x):
    try: return int(x)
    except: return None

class Session:
    def __init__(
        self,
        audio_source: AudioSource,
        asr_client: NemoClient,
        llm_client: LLMClient,
        turn_manager: TurnManager,
        log_path: Path,
        language: str = "en-US",
        reference_text: Optional[str] = None,
        audio_id: Optional[str] = None,
        asr_config: Optional[dict] = None,
        llm_config: Optional[dict] = None,
        reference_end_s: Optional[float] = None,
    ):
        self.audio_source = audio_source
        self.asr = asr_client
        self.llm = llm_client
        self.tm = turn_manager
        self.log_path = Path(log_path)
        self.language = language
        self.reference_text = reference_text
        self.audio_id = audio_id or str(uuid.uuid4())[:8]
        self.session_id = str(uuid.uuid4())[:8]
        self.start_t = None
        self.events: list[dict] = []
        self.telemetry = GPUTelemetry(log_path=self.log_path.parent / "gpu.jsonl")
        self.asr_config = asr_config
        self.llm_config = llm_config
        self.reference_end_s = reference_end_s

        # wire turn commit to LLM
        self.tm.on_commit = self._on_turn_commit_sync_wrapper
        self._pending_llm_text = ""
        self._llm_first_token_t = None

    def _log(self, event: str, **kwargs):
        t = time.monotonic() - (self.start_t or time.monotonic())
        entry = {"t": round(t, 3), "event": event, "session": self.session_id, **kwargs}
        self.events.append(entry)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except: pass
        return entry

    def _on_turn_commit_sync_wrapper(self, text: str, lang: str):
        pass

    async def run_async(self):
        self.start_t = time.monotonic()
        # ensure parent dir exists and truncate events
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try: open(self.log_path, "w").close()
        except: pass
        self.telemetry.start()

        # Gather system info
        gpu_info = query_nvidia_smi()
        cuda_version = get_cuda_version()
        git_rev = get_git_revision()
        # Model hashes (first 10MB + size)
        asr_path = None
        llm_path = None
        try:
            asr_path = getattr(self.asr.cfg, "model_path", None) if hasattr(self.asr, "cfg") else None
            llm_path = getattr(self.llm.cfg, "model", None) if hasattr(self.llm, "cfg") else None
            # fallback to config dict
            if not asr_path and self.asr_config:
                asr_path = self.asr_config.get("model", {}).get("path")
            if not llm_path and self.llm_config:
                llm_path = self.llm_config.get("model_path")
        except: pass

        vram_before = snapshot_vram()
        self._log("audio_start", audio_id=self.audio_id, language=self.language, reference_text=self.reference_text, reference_end_s=self.reference_end_s)
        # Log configs and system
        self._log("system_info",
                  gpu_name=gpu_info.get("gpu_name"),
                  driver_version=gpu_info.get("driver_version"),
                  cuda_version=cuda_version,
                  git_revision=git_rev,
                  asr_model_path=asr_path,
                  asr_model_hash=file_hash(asr_path) if asr_path else None,
                  llm_model_path=llm_path,
                  llm_model_hash=file_hash(llm_path) if llm_path else None,
                  asr_config=self.asr_config,
                  llm_config=self.llm_config,
                  )
        self._log("vram_before", vram_mb=vram_before, vram_total_mb=gpu_info.get("vram_total_mb"))

        # If reference_end_s known, schedule reference_speech_end event (wall time = reference_end_s after audio_start)
        # We log it at the right wall time after audio playback? For FileReplay, we know duration, but for true latency we need to log when reference speech would have ended
        # For now, log immediately for offline analysis
        if self.reference_end_s is not None:
            # This will be used to compute endpoint latency = endpoint_t - reference_end_s
            self._log("reference_speech_end", reference_end_s=self.reference_end_s)

        vram_after_qwen = None
        # Try to get VRAM after Qwen if server already running (snapshot)
        # This is best-effort; true "after Qwen" should be after Qwen load, but Qwen is already loaded on host
        vram_after_qwen = snapshot_vram()
        self._log("vram_after_qwen", vram_mb=vram_after_qwen)

        loop = asyncio.get_event_loop()

        def on_partial(p: ASRPartial):
            ts = time.monotonic() - self.start_t
            if p.is_final:
                console.print(f"[{ts:06.2f}] user ✓ {p.text}")
                self._log("asr_final", text=p.text, language=p.language)
                self._log("endpoint", text=p.text)
            else:
                console.print(f"[{ts:06.2f}] user ~ {p.text}")
                self._log("asr_partial", text=p.text, is_final=p.is_final)
            try:
                loop.call_soon_threadsafe(self.tm.on_partial, p)
            except:
                self.tm.on_partial(p)

        # Collect frames - for FileReplay this paces; for mic it's live
        # To avoid double-pacing (list() would pace), we stream directly
        # But we also want to log audio_end after frames done, so we need to handle both

        # For file replay, we need to know duration; for mic, we don't
        # Use async generator that logs audio_end when done
        async def frame_gen():
            count = 0
            for f in self.audio_source.frames():
                count += 1
                yield f
                if f.is_last:
                    break
            self._log("audio_end", frames=count)

        # Run ASR streaming
        final_text = await self.asr.stream_frames(frame_gen(), on_partial, language=self.language)

        vram_after_asr = snapshot_vram()
        self._log("vram_after_asr", vram_mb=vram_after_asr)

        # After ASR completes, if turn manager is in LLM_GENERATING, run LLM
        llm_full = ""
        prompt_tokens = None
        gen_tokens = None
        prompt_tps = None
        gen_tps = None
        if self.tm.state == State.LLM_GENERATING and self.tm.current:
            committed_text = self.tm.current.final_text or final_text
            # Log Qwen request with UTF-8/tokenization timing (trivial)
            tok_start = time.monotonic()
            # Tokenization is inside llm_client; we approximate as zero
            self._log("llm_request", text=committed_text, utf8_bytes=len(committed_text.encode("utf-8")))
            self._log("utf8_tokenization", latency_ms=round((time.monotonic()-tok_start)*1000, 2))
            console.print(f"[endpoint +{time.monotonic()-self.start_t:.2f}s] LLM request")
            llm_start = time.monotonic()
            first_token_t = None
            full = ""
            # Capture timings from llm_client if it exposes them; we will compute from events
            def on_token(tok: str):
                nonlocal first_token_t, full
                if first_token_t is None:
                    first_token_t = time.monotonic()
                    self._log("llm_first_token", text=tok, latency_s=round(first_token_t - llm_start, 3))
                    console.print(f"[first token +{first_token_t - llm_start:.2f}s] ", end="")
                full += tok
                console.print(tok, end="", highlight=False)

            # stream_chat may return timings via llm_client
            full = await self.llm.stream_chat(on_token)
            # If on_token not called but full returned (non-streaming fallback)
            if full and not on_token:
                pass
            console.print()
            done_t = time.monotonic()
            # Try to extract token counts from llm_client or estimate
            prompt_tokens = len(committed_text.split())
            gen_tokens = len(full.split())
            # Try to get real token counts if llm_client stored them
            # For now estimate; llm_client could expose usage
            total_time = done_t - llm_start
            if first_token_t:
                prefill_time = first_token_t - llm_start
                decode_time = done_t - first_token_t if first_token_t else 0
                # Approximate tok/s
                if prefill_time > 0 and prompt_tokens:
                    prompt_tps = round(prompt_tokens / max(prefill_time, 0.001), 1)
                if decode_time > 0 and gen_tokens:
                    gen_tps = round(gen_tokens / max(decode_time, 0.001), 1)
            llm_full = full
            self._log("llm_done", text=full, prompt_tokens=prompt_tokens, generated_tokens=gen_tokens,
                      prompt_tps=prompt_tps, generation_tps=gen_tps,
                      total_time_s=round(total_time, 3))
            # Update history
            try:
                self.llm.add_user(committed_text)
                self.llm.add_assistant(full)
            except: pass
            self.tm.complete_llm(full, first_token_t=first_token_t, done_t=done_t)
            self._log("turn_complete", llm_text=full)

        self.telemetry.stop()
        peak = self.telemetry.peak_vram_mb()
        self._log("session_end", peak_vram_mb=peak, telemetry_summary=self.telemetry.summary())

        # Build summary
        # Compute latencies if reference_end_s known and we have endpoint/first token times
        # Find events
        def find_event(name):
            for e in self.events:
                if e["event"] == name:
                    return e
            return None
        asr_final_ev = find_event("asr_final")
        endpoint_ev = find_event("endpoint")
        first_tok_ev = find_event("llm_first_token")
        ref_end = self.reference_end_s
        # If ref_end is None, try to infer from audio duration or last word timing (not available here)
        summary = {
            "session": self.session_id,
            "audio_id": self.audio_id,
            "language": self.language,
            "reference_text": self.reference_text,
            "reference_end_s": ref_end,
            "asr_final_text": asr_final_ev.get("text") if asr_final_ev else final_text,
            "llm_text": llm_full,
            "peak_vram_mb": peak,
            "vram_before_mb": vram_before,
            "vram_after_qwen_mb": vram_after_qwen,
            "vram_after_asr_mb": vram_after_asr,
            "vram_total_mb": gpu_info.get("vram_total_mb"),
            "gpu_name": gpu_info.get("gpu_name"),
            "cuda_version": cuda_version,
            "git_revision": git_rev,
            "asr_model_path": asr_path,
            "asr_model_hash": file_hash(asr_path) if asr_path else None,
            "llm_model_path": llm_path,
            "llm_model_hash": file_hash(llm_path) if llm_path else None,
            "asr_config": self.asr_config,
            "llm_config": self.llm_config,
            "events": len(self.events),
            "telemetry": self.telemetry.summary(),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "prompt_tps": prompt_tps,
            "generation_tps": gen_tps,
        }
        # Add endpoint latency if computable
        if ref_end is not None and endpoint_ev:
            summary["endpoint_latency_s"] = round(endpoint_ev["t"] - ref_end, 3)
        if ref_end is not None and asr_final_ev:
            summary["asr_final_latency_s"] = round(asr_final_ev["t"] - ref_end, 3)
        if ref_end is not None and first_tok_ev:
            summary["llm_ttft_s"] = round(first_tok_ev["t"] - ref_end, 3)

        # Write summary.json
        try:
            with open(self.log_path.parent / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except: pass

        console.print(f"[session] peak VRAM {peak:.0f} MB, events {len(self.events)} -> {self.log_path}")
        return summary

    def run(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self.run_async())
                return fut.result()
        else:
            return loop.run_until_complete(self.run_async())

    async def run_turn(self, audio_source: AudioSource, reference_text: Optional[str] = None, reference_end_s: Optional[float] = None):
        self.audio_source = audio_source
        self.reference_text = reference_text
        self.reference_end_s = reference_end_s
        return await self.run_async()
