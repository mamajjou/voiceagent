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
        return f"ws://{self.cfg.host}:{self.cfg.port}/v1/realtime"

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
        # NeMo-Speech.cpp WebSocket /v1/realtime protocol
        # Server sends session.created on connect, client sends session.update,
        # then binary PCM16 frames, then input_audio_buffer.commit.
        # Server sends conversation.item.input_audio_transcription.delta (partial)
        # and .completed (final).
        final_text = ""
        async with websockets.connect(self.ws_url, max_size=10*1024*1024) as ws:
            # Wait for session.created (optional)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                try:
                    data = json.loads(msg)
                    # ignore session.created
                except:
                    pass
            except:
                pass
            # Send session.update with ASR config
            # Map rnnt_right_context to endpointing via server defaults; rnnt_right_context itself is model-internal
            # and not exposed via session.update, but we pass endpointing_ms from eou_ms
            session_update = {
                "type": "session.update",
                "session": {
                    "sample_rate": 16000,
                    "language": lang,
                    "endpointing_ms": self.cfg.eou_ms if self.cfg.enable_endpointing else 0,
                    "automatic_punctuation": True,
                    "word_timestamps": True,
                }
            }
            await ws.send(json.dumps(session_update))
            # Concurrent send and receive
            async def sender():
                if hasattr(audio_frames, "__aiter__"):
                    async for frame in audio_frames:
                        await ws.send(frame.pcm16)
                        if frame.is_last:
                            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            break
                else:
                    for frame in audio_frames:
                        await ws.send(frame.pcm16)
                        await asyncio.sleep(0.001)
                        if frame.is_last:
                            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            break
                await asyncio.sleep(0.2)

            async def receiver():
                nonlocal final_text
                accum = ""
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
                    t = data.get("type", "")
                    # delta is incremental, accumulate for full hypothesis
                    if t == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta") or data.get("text") or ""
                        if delta:
                            accum += delta
                            p = ASRPartial(text=accum, timestamp=time.monotonic(), is_final=False, is_endpoint=False, language=lang)
                            on_partial(p)
                    elif t == "conversation.item.input_audio_transcription.completed":
                        transcript = data.get("transcript") or data.get("text") or ""
                        # words may be in data["words"] but we just use transcript
                        p = ASRPartial(text=transcript, timestamp=time.monotonic(), is_final=True, is_endpoint=True, language=lang)
                        on_partial(p)
                        final_text = transcript
                        break
                    elif t == "session.updated":
                        continue
                    elif t == "input_audio_buffer.committed":
                        continue
                    elif t == "error":
                        print(f"[nemo] server error {data}")
                        break
                    # Handle legacy fallback keys
                    elif "delta" in data and "type" not in data:
                        p = ASRPartial(text=data["delta"], timestamp=time.monotonic(), is_final=False, language=lang)
                        on_partial(p)
                    elif "transcript" in data and data.get("type") == "completed":
                        p = ASRPartial(text=data["transcript"], timestamp=time.monotonic(), is_final=True, is_endpoint=True, language=lang)
                        on_partial(p)
                        final_text = data["transcript"]
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
