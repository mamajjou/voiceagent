"""Server-based realtime streaming latency benchmark.

Replays a WAV through the nemo-speech WebSocket /v1/realtime endpoint at
realtime_factor=1.0 (20ms chunks), times the speech-end -> EOU -> ASR-final
latency, and sweeps endpointing_ms (EOU threshold).

Server must be running at 127.0.0.1:8090 with the configured model.
"""
import asyncio, json, time, os, sys
import numpy as np
import soundfile as sf
import websockets

WS = 'ws://127.0.0.1:8090/v1/realtime'
SR = 16000
CHUNK_MS = 20

def load_pcm16(path):
    data, sr = sf.read(path, always_2d=False, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        dur = len(data) / sr
        target = int(dur * SR)
        old = np.linspace(0, 1, len(data))
        new = np.linspace(0, 1, target)
        data = np.interp(new, old, data).astype(np.float32)
    data = np.clip(data, -1, 1)
    return (data * 32767).astype(np.int16)

async def run_one(path, eou_ms, lang):
    pcm = load_pcm16(path)
    total_bytes = len(pcm)
    audio_dur = total_bytes / SR
    n_chunks = (total_bytes + CHUNK_MS * SR // 1000 - 1) // (CHUNK_MS * SR // 1000)
    chunk_bytes = CHUNK_MS * SR // 1000 * 2

    events = []
    async with websockets.connect(WS, max_size=50*1024*1024) as ws:
        # session.created
        try:
            await asyncio.wait_for(ws.recv(), 2.0)
        except Exception:
            pass
        # send endpointing via session.update (rnn_right_context is server-fixed)
        await ws.send(json.dumps({
            'type': 'session.update',
            'session': {'sample_rate': SR, 'language': lang,
                        'endpointing_ms': eou_ms, 'automatic_punctuation': True,
                        'word_timestamps': True}
        }))

        # Reset monotonic clock reference
        t_ref = time.monotonic()
        speech_end_wall = t_ref + audio_dur

        async def sender():
            for i in range(n_chunks):
                s = i * chunk_bytes
                e = min(s + chunk_bytes, total_bytes)
                chunk = pcm[s:e]
                # realtime pacing
                target = t_ref + (i * CHUNK_MS / 1000.0)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                await ws.send(chunk.tobytes())
            # commit after all audio sent (endpoint handled by server via silence)
            await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)

        final = {'text': '', 't': None, 'endpoint': False}
        partials = []

        async def receiver():
            accum = ''
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), 8.0)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                t = data.get('type', '')
                if t == 'conversation.item.input_audio_transcription.delta':
                    dl = data.get('delta') or data.get('text') or ''
                    if dl:
                        accum += dl
                        partials.append((time.monotonic() - t_ref, accum, False))
                elif t == 'conversation.item.input_audio_transcription.completed':
                    txt = data.get('transcript') or data.get('text') or ''
                    final['text'] = txt
                    final['t'] = time.monotonic() - t_ref
                    partials.append((final['t'], txt, True))
                    break
                elif t == 'error':
                    break

        await asyncio.gather(sender(), receiver())

    end_latency = final['t'] - audio_dur if final['t'] is not None else None
    return {
        'audio_dur': audio_dur,
        'final_t': final['t'],
        'end_latency': end_latency,
        'text': final['text'],
        'n_partials': len(partials),
        'partials': partials,
    }

async def main():
    corpus = [
        ('/tmp/de.wav', 'de'), ('/tmp/de2.wav', 'de'), ('/tmp/de3.wav', 'de'),
        ('/tmp/en1.wav', 'en'), ('/tmp/en2.wav', 'en'), ('/tmp/en3.wav', 'en'),
    ]
    eou_grid = [350, 500, 650, 800, 1000]
    for eou in eou_grid:
        print(f"\n===== EOU={eou}ms =====", flush=True)
        latencies = []
        for path, lang in corpus:
            r = await run_one(path, eou, lang)
            lat = r['end_latency']
            latencies.append(lat if lat is not None else 999)
            print(f"  {path} dur={r['audio_dur']:.2f}s final_t={r['final_t']:.3f} end_latency={lat:.3f} '{r['text'][:50]}'", flush=True)
        good = [l for l in latencies if l < 10]
        if good:
            print(f"  MEDIAN end_latency={sorted(good)[len(good)//2]:.3f}s  n={len(good)}", flush=True)

asyncio.run(main())
