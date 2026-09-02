"""NeMo-Speech.cpp client.

Talks to nemo-speech.cpp server over WebSocket/HTTP.
The server is expected to run: nemo-speech-server with Nemotron 3.5 GGUF.

We support mock mode for testing without GPU (replays reference text as ASR).
"""
from __future__ import annotations
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional
import httpx
import websockets
import websockets.exceptions

@dataclass
class ASRPartial:
    text: str
    timestamp: float  # event time monotonic
    is_final: bool
    is_endpoint: bool = False
    language: str = "en-US"
    stability: Optional[float] = None

@dataclass
class ASRConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    model_path: str = "/workspace/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
    gpu: int = 0
    rnnt_right_context: int = 3
    eou_ms: int = 650
    language: str = "en-US"
    enable_endpointing: bool = True
    vad_based: bool = False

class NemoClient:
    """Streaming ASR client with reconnection and mock fallback."""

    def __init__(self, cfg: ASRConfig, mock_text: Optional[str] = None):
        self.cfg = cfg
        self.mock_text = mock_text  # if set, use mock streaming
        self._ws = None
        self._session_id = str(uuid.uuid4())[:8]

    @property
    def ws_url(self) -> str:
        return f"ws://{self.cfg.host}:{self.cfg.port}/v1/stream"

    @property
    def http_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    async def check_health(self) -> bool:
        if self.mock_text is not None:
            return True
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.http_url}/health")
                return r.status_code == 200
        except:
            # try websocket
            try:
                async with websockets.connect(self.ws_url, open_timeout=2) as ws:
                    await ws.close()
                    return True
            except:
                return False

    async def stream_frames(
        self,
        audio_frames,  # async or sync iterable of AudioFrame
        on_partial: Callable[[ASRPartial], None],
        language: Optional[str] = None,
    ) -> str:
        """Stream audio frames, call on_partial for each hypothesis, return final text."""
        lang = language or self.cfg.language
        if self.mock_text is not None:
            return await self._mock_stream(audio_frames, on_partial, lang)
        # Real streaming via websocket
        try:
            return await self._ws_stream(audio_frames, on_partial, lang)
        except Exception as e:
            print(f"[nemo] ws stream failed {e}, falling back to mock if available")
            if self.mock_text:
                return await self._mock_stream(audio_frames, on_partial, lang)
            raise

    async def _ws_stream(self, audio_frames, on_partial, lang) -> str:
        # Protocol: send json config then binary PCM chunks
        # This is the expected NeMo-Speech.cpp protocol; we handle version drift.
        final_text = ""
        async with websockets.connect(self.ws_url, max_size=10*1024*1024) as ws:
            # Send initial config
            init = {
                "session_id": self._session_id,
                "language": lang,
                "sample_rate": 16000,
                "rnnt_right_context": self.cfg.rnnt_right_context,
                "endpointing": {
                    "enable": self.cfg.enable_endpointing,
                    "stop_history_eou_ms": self.cfg.eou_ms,
                    "vad_based": self.cfg.vad_based,
                }
            }
            await ws.send(json.dumps(init))
            # Concurrent send and receive
            async def sender():
                # audio_frames may be sync generator; run in thread if needed
                # we support async generator or sync generator wrapped
                if hasattr(audio_frames, "__aiter__"):
                    async for frame in audio_frames:
                        await ws.send(frame.pcm16)
                        if frame.is_last:
                            # signal end
                            await ws.send(json.dumps({"eof": True}))
                            break
                else:
                    # sync iterable - need to run in executor to not block
                    import asyncio
                    loop = asyncio.get_event_loop()
                    def sync_send():
                        for frame in audio_frames:
                            # we need to async send from sync context -> use run_coroutine_threadsafe?
                            # Instead, we make this sender async but iterate sync
                            pass
                    # fallback: iterate sync directly (blocking but ok for small files if we yield sometimes)
                    for frame in audio_frames:
                        await ws.send(frame.pcm16)
                        await asyncio.sleep(0.005)  # yield
                        if frame.is_last:
                            await ws.send(json.dumps({"eof": True}))
                            break
                # small delay to allow final to arrive
                await asyncio.sleep(0.2)

            async def receiver():
                nonlocal final_text
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        continue
                    try:
                        data = json.loads(msg)
                    except:
                        continue
                    # expected fields: text, is_final, is_endpoint, type
                    text = data.get("text") or data.get("transcript") or ""
                    is_final = data.get("is_final") or data.get("final") or False
                    is_endpoint = data.get("is_endpoint") or data.get("endpoint") or data.get("eou") or False
                    # also handle "partial" vs "final"
                    event_type = data.get("type") or ("final" if is_final else "partial")
                    if text is not None:
                        partial = ASRPartial(
                            text=text,
                            timestamp=time.monotonic(),
                            is_final=is_final,
                            is_endpoint=is_endpoint or is_final,
                            language=lang,
                        )
                        on_partial(partial)
                        if is_final:
                            final_text = text
                            if is_endpoint:
                                break
                    if data.get("eof") or data.get("done"):
                        break
            await asyncio.gather(sender(), receiver())
        return final_text

    async def _mock_stream(self, audio_frames, on_partial, lang) -> str:
        """Mock: gradually reveal mock_text as partials."""
        # consume frames to simulate realtime, but emit partials based on time
        text = self.mock_text or ""
        words = text.split()
        # consume frames in background to simulate latency
        # we just sleep proportionally to audio duration
        # emit partials word-by-word
        start = time.monotonic()
        # Drain frames quickly if realtime_factor=0 else paced already
        # For mock, we don't actually need to send audio
        # Just simulate timing
        if hasattr(audio_frames, "__aiter__"):
            async for _ in audio_frames:
                pass
        else:
            for _ in audio_frames:
                pass
        # Now emit partials
        for i in range(1, len(words)+1):
            partial_text = " ".join(words[:i])
            is_final = (i == len(words))
            p = ASRPartial(text=partial_text, timestamp=time.monotonic(), is_final=is_final, is_endpoint=is_final, language=lang)
            on_partial(p)
            await asyncio.sleep(0.06)  # 60ms per word
        return text

    # Synchronous wrapper for orchestrator that uses threads
    def stream_frames_sync(self, frames, on_partial, language=None) -> str:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self.stream_frames(frames_async_wrap(frames), on_partial, language))
                return fut.result()
        else:
            return loop.run_until_complete(self.stream_frames(frames_async_wrap(frames), on_partial, language))

def frames_async_wrap(sync_frames):
    async def gen():
        for f in sync_frames:
            yield f
    return gen()

# Helper to test server without audio
async def health_check(cfg: ASRConfig) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"http://{cfg.host}:{cfg.port}/health")
            return {"status": r.status_code, "body": r.text[:500]}
        except Exception as e:
            return {"status": "error", "error": str(e)}
