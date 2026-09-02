"""Definitive EOU sweep for the naive voice baseline.

Simulates the orchestrator: stream speech + trailing silence at realtime,
detect end-of-speech (energy-based), wait a silence window (eou_ms), then
commit to finalize. Measures speech-end -> ASR-final latency.

Server is fixed at rnnt_right_context=3 (320ms) via launch flag. We sweep the
silence window (EOU) that the TurnManager applies before commit.

energy-based VAD: compute RMS per 20ms chunk; speech_end is the last frame
above an adaptive threshold.
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

def detect_speech_end(pcm):
    """Return (speech_end_sample_index) via energy VAD on 20ms frames."""
    n_chunks = (len(pcm) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    rms = []
    for i in range(n_chunks):
        s = i * CHUNK_SAMPLES
        e = min(s + CHUNK_SAMPLES, len(pcm))
        chunk = pcm[s:e].astype(np.float32) / 32768.0
        r = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        rms.append(r)
    rms = np.array(rms)
    # adaptive threshold: speech is frames above max(0.01, 10% of peak)
    peak = rms.max()
    thr = max(0.008, 0.10 * peak)
    speech = rms > thr
    if not speech.any():
        return None
    # last speech frame index (sample boundary)
    last_speech_idx = int(np.max(np.nonzero(speech)))
    # speech end sample = end of that frame
    return min((last_speech_idx + 1) * CHUNK_SAMPLES, len(pcm))

async def run_one(path, lang, eou_ms):
    pcm = load_pcm16(path)
    audio_dur = len(pcm) / SR
    speech_end_sample = detect_speech_end(pcm)
    speech_end_s = (speech_end_sample if speech_end_sample else len(pcm)) / SR
    n_chunks = (len(pcm) + CHUNK_BYTES - 1) // CHUNK_BYTES
    # pad enough trailing silence beyond speech_end + eou
    pad_chunks = (eou_ms // CHUNK_MS) + 20
    silent = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

    async with websockets.connect(WS, max_size=50*1024*1024) as ws:
        try:
            await asyncio.wait_for(ws.recv(), 2.0)
        except Exception:
            pass
        await ws.send(json.dumps({'type': 'session.update', 'session': {
            'sample_rate': SR, 'language': lang, 'automatic_punctuation': True}}))
        t_ref = time.monotonic()

        async def sender():
            # speech frames paced to realtime
            for i in range(n_chunks):
                s = i * CHUNK_BYTES
                e = min(s + CHUNK_BYTES, len(pcm))
                target = t_ref + (i * CHUNK_MS / 1000.0)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                await ws.send(pcm[s:e].tobytes())
            # continue streaming silence; TurnManager sees silence, waits eou_ms
            base = n_chunks * CHUNK_MS / 1000.0
            for j in range(pad_chunks):
                target = t_ref + base + (j * CHUNK_MS / 1000.0)
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                await ws.send(silent.tobytes())
            # after sending silence through eou window, commit
            # commit arrives at t_ref + speech_end_s + eou_ms
            await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)

        final = {'text': '', 't': None, 'endpoint': False}
        partials = []
        async def receiver():
            accum = ''
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), 10.0)
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
                        partials.append((time.monotonic() - t_ref, accum))
                elif t == 'conversation.item.input_audio_transcription.completed':
                    final['text'] = data.get('transcript') or data.get('text') or ''
                    final['t'] = time.monotonic() - t_ref
                    break
                elif t == 'error':
                    break

        await asyncio.gather(sender(), receiver())

    final_lat = final['t'] - speech_end_s if final['t'] is not None else None
    return {
        'audio_dur': audio_dur, 'speech_end_s': speech_end_s,
        'final_t': final['t'], 'final_lat': final_lat,
        'text': final['text'], 'n_partials': len(partials),
    }

async def main():
    corpus = [
        ('/tmp/de.wav','de','Guten Tag. Wie geht es dir?'),
        ('/tmp/de2.wav','de','Guten Morgen, wie geht es Ihnen heute?'),
        ('/tmp/de3.wav','de','Ich bin gestern nach Berlin gefahren.'),
        ('/tmp/de4.wav','de','Wie lautet die Hauptstadt von Deutschland?'),
        ('/tmp/en1.wav','en','She had your dark suit and greasy washwater all year.'),
        ('/tmp/en2.wav','en','Could you explain why the sky is blue?'),
        ('/tmp/en3.wav','en','What is the capital of France?'),
        ('/tmp/en4.wav','en','How does a transformer attention mechanism work?'),
    ]
    eou_grid = [350, 500, 650, 800, 1000]
    print("TurnManager sweep: speech_end -> EOU silence window -> commit -> ASR final\n", flush=True)
    all_results = []
    for eou in eou_grid:
        lats = []
        print(f"===== EOU={eou}ms =====", flush=True)
        for path, lang, ref in corpus:
            r = await run_one(path, lang, eou)
            fl = r['final_lat']
            lats.append(fl if fl is not None else 999)
            print(f"  {os.path.basename(path)} speech_end={r['speech_end_s']:.2f}s final_t={r['final_t']:.3f} final_lat={fl:.3f}s n_part={r['n_partials']}", flush=True)
        good = sorted(l for l in lats if l < 10)
        if good:
            med = good[len(good)//2]
            print(f"  -> MEDIAN final_latency (speech_end->ASR_final) = {med:.3f}s  n={len(good)}", flush=True)
        all_results.append((eou, good))

    # aggregate medians across EOU
    print("\n=== SWEEP SUMMARY (speech_end -> ASR final, server rc=320ms) ===", flush=True)
    print(f"{'EOU ms':>7} {'median':>8} {'p90':>8} {'p95':>8}", flush=True)
    for eou, good in all_results:
        if good:
            print(f"{eou:>7} {sorted(good)[len(good)//2]:>8.3f} {sorted(good)[int(len(good)*0.9)]:>8.3f} {sorted(good)[min(len(good)-1,int(len(good)*0.95))]:>8.3f}", flush=True)

asyncio.run(main())
