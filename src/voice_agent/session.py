"""Session orchestrator: AudioSource → ASR → TurnManager → LLM → logger."""
from __future__ import annotations
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console

from .audio import AudioSource, FileReplayAudioSource
from .nemo_client import NemoClient, ASRConfig, ASRPartial
from .llm_client import LLMClient, LLMConfig
from .turn_manager import TurnManager, State
from .telemetry import GPUTelemetry, snapshot_vram

console = Console()

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
        self.events = []
        self.telemetry = GPUTelemetry(log_path=self.log_path.parent / "gpu.jsonl")

        # wire turn commit to LLM
        self.tm.on_commit = self._on_turn_commit_sync_wrapper
        self._pending_llm_text = ""
        self._llm_first_token_t = None

    def _log(self, event: str, **kwargs):
        t = time.monotonic() - (self.start_t or time.monotonic())
        entry = {"t": round(t, 3), "event": event, "session": self.session_id, **kwargs}
        self.events.append(entry)
        # also write to jsonl
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except:
            pass
        return entry

    def _on_turn_commit_sync_wrapper(self, text: str, lang: str):
        # called from TurnManager when ASR final+EOU committed
        # We need to trigger LLM in a thread-safe way; for sync session we block
        # This is called within on_partial context, so we handle LLM here synchronously
        # But to avoid deadlock, we set state and let the outer loop handle it?
        # Simpler: do LLM inline (blocking) - works for file replay
        pass  # handled in run loop via polling state

    async def run_async(self):
        self.start_t = time.monotonic()
        self.telemetry.start()
        self._log("audio_start", audio_id=self.audio_id, language=self.language, reference_text=self.reference_text)
        vram_before = snapshot_vram()
        self._log("vram_before", vram_mb=vram_before)

        # We need to stream audio frames and ASR concurrently
        # For file replay, we can iterate frames and send to ASR
        # ASR client will call on_partial for each hypothesis

        # Shared state for LLM trigger
        loop = asyncio.get_event_loop()
        llm_task = None

        def on_partial(p: ASRPartial):
            # Called from ASR client's thread
            ts = time.monotonic() - self.start_t
            if p.is_final:
                console.print(f"[{ts:06.2f}] user ✓ {p.text}")
                self._log("asr_final", text=p.text, language=p.language)
                self._log("endpoint", text=p.text)
                # mark reference_speech_end if known from manifest
                # For now, log asr_final time; reference end will be compared offline
            else:
                console.print(f"[{ts:06.2f}] user ~ {p.text}")
                self._log("asr_partial", text=p.text)
            # feed to turn manager
            # need thread-safe call
            try:
                loop.call_soon_threadsafe(self.tm.on_partial, p)
            except:
                self.tm.on_partial(p)

        # Collect frames
        frames = list(self.audio_source.frames())  # for file replay, this paces
        self._log("audio_end", frames=len(frames))

        # Stream to ASR (async)
        # Wrap frames as async generator
        async def frame_gen():
            for f in frames:
                yield f

        # Run ASR streaming
        final_text = await self.asr.stream_frames(frame_gen(), on_partial, language=self.language)
        # After ASR completes, if turn manager is in LLM_GENERATING, run LLM
        if self.tm.state == State.LLM_GENERATING and self.tm.current:
            committed_text = self.tm.current.final_text or final_text
            self._log("llm_request", text=committed_text)
            console.print(f"[endpoint +{time.monotonic()-self.start_t:.2f}s] LLM request")
            llm_start = time.monotonic()
            first_token_t = None
            full = ""

            def on_token(tok: str):
                nonlocal first_token_t, full
                if first_token_t is None:
                    first_token_t = time.monotonic()
                    self._log("llm_first_token", text=tok, latency_s=round(first_token_t - llm_start, 3))
                    console.print(f"[first token +{first_token_t - llm_start:.2f}s] ", end="")
                full += tok
                console.print(tok, end="", highlight=False)
                # also log token? too verbose

            full = await self.llm.stream_chat(on_token)
            console.print()  # newline
            done_t = time.monotonic()
            self._log("llm_done", text=full, prompt_tokens=len(committed_text.split()), generated_tokens=len(full.split()))
            self.llm.add_user(committed_text)
            self.llm.add_assistant(full)
            self.tm.complete_llm(full, first_token_t=first_token_t, done_t=done_t)
            # latency breakdown
            # reference_speech_end unknown here; will be computed offline if manifest provides end_s
            self._log("turn_complete", llm_text=full)

        self.telemetry.stop()
        self._log("session_end", peak_vram_mb=self.telemetry.peak_vram_mb())
        # write summary
        summary = {
            "session": self.session_id,
            "audio_id": self.audio_id,
            "language": self.language,
            "reference_text": self.reference_text,
            "peak_vram_mb": self.telemetry.peak_vram_mb(),
            "vram_before": vram_before,
            "events": len(self.events),
        }
        with open(self.log_path.parent / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        console.print(f"[session] peak VRAM {self.telemetry.peak_vram_mb():.0f} MB, events {len(self.events)} -> {self.log_path}")
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

    # For multi-turn chaining
    async def run_turn(self, audio_source: AudioSource, reference_text: Optional[str] = None):
        # reuse same LLM history, new audio source
        self.audio_source = audio_source
        self.reference_text = reference_text
        return await self.run_async()
