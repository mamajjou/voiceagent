"""Measure true server-side EOU latency: stream speech + trailing silence,
do NOT send commit; let server detect endpoint via stop_history_eou_ms.

Sweeps eou_ms by relaunching is not possible (server-fixed), so we test the
server started with eou=650 and record whether server-side endpointing fires
on its own, and how long after speech-end the final arrives.
"""
import asyncio, json, time, os
import numpy as np
import soundfile as sf
import websockets

WS = 'ws://127.0.0.1:8090/v1/realtime'
SR = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = SR * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

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

async def run_one(path, lang, pad_silence_ms=1500, eou_ms=650):
    pcm = load_pcm16(path)
    audio_dur = len(pcm) / SR
    # build padded stream: speech + trailing silence
    silent = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
    events = []

    async with websockets.connect(WS, max_size=50*1024*1024) as ws:
        try:
            await asyncio.wait_for(ws.recv(), 2.0)
        except Exception:
            pass
        # NOTE: do not set endpointing_ms (server-fixed via launch flag)
        await ws.send(json.dumps({'type': 'session.update', 'session': {
            'sample_rate': SR, 'language': lang, 'automatic_punctuation': True}}))

        t_ref = time.monotonic()
        speech_end_wall = t_ref + audio_dur

        # chunks: speech then N silent chunks
        n_speech = (len(pcm) + CHUNK_BYTES - 1) // CHUNK_BYTES
        n_sil = pad_silence_ms // CHUNK_MS

        async def sender():
            for i in range(n_speech):
                s = i * CHUNK_BYTES
                e = min(s + CHUNK_BYTES, len(pcm))
                target = t_ref + (i * CHUNK_MS / 1000.0)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                await ws.send(pcm[s:e].tobytes())
            # trailing silence, paced
            base = n_speech * CHUNK_MS / 1000.0
            for j in range(n_sil):
                target = t_ref + base + (j * CHUNK_MS / 1000.0)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                await ws.send(silent.tobytes())
            # do NOT commit; let server detect silence

        final = {'text': '', 't': None}
        last_delta_t = None
        async def receiver():
            accum = ''
            last_partial = {'t': None}
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), 12.0)
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
                        last_partial['t'] = time.monotonic() - t_ref
                elif t == 'conversation.item.input_audio_transcription.completed':
                    final['text'] = data.get('transcript') or data.get('text') or ''
                    final['t'] = time.monotonic() - t_ref
                    break
                elif t == 'error':
                    break

        await asyncio.gather(sender(), receiver())

    eou_latency = final['t'] - audio_dur if final['t'] else None
    return audio_dur, eou_latency, final['text'], final['t']

async def main():
    corpus = [('/tmp/de.wav','de'), ('/tmp/de2.wav','de'), ('/tmp/en1.wav','en'), ('/tmp/en2.wav','en')]
    print("Server launched with --asr.endpointing.stop_history_eou_ms=650 (assumed).", flush=True)
    print("Streaming speech + 1.5s trailing silence, NO commit. If server-side EOU works,")
    print("final should arrive ~= speech_end + eou_ms. If end_latency ~= pad_silence, commit-like.", flush=True)
    for path, lang in corpus:
        dur, eou_lat, text, ft = await run_one(path, lang)
        print(f"{path} dur={dur:.2f}s final_t={ft if ft is not None else 'NONE'}  EOU_lat={eou_lat if eou_lat is not None else 'NONE'}  '{text[:60]}'", flush=True)

asyncio.run(main())
