"""Precise latency waterfall + partial stability + VRAM, using reference
speech-end from audio energy (strict threshold). Run for rc=320ms server.

Reports for each clip: speech_end(ground truth via 0.01 RMS), EOU window,
final_t, and the waterALL breakdown. Also computes partial stability (LCP).
"""
import asyncio, json, time, os, subprocess
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
        tgt = int(dur * SR)
        old = np.linspace(0, 1, len(data))
        new = np.linspace(0, 1, tgt)
        data = np.interp(new, old, data).astype(np.float32)
    data = np.clip(data, -1, 1)
    return (data * 32767).astype(np.int16)

def ref_speech_end(pcm, thr=0.01):
    """Last sample with |amp| > thr (absolute), robust to gTTS tails."""
    n = len(pcm)
    # frame RMS
    nch = (n + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    last = 0
    for i in range(nch):
        s = i*CHUNK_SAMPLES; e = min(s+CHUNK_SAMPLES, n)
        c = pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2)) > thr:
            last = i+1
    return min(last*CHUNK_SAMPLES, n)

def vram():
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        parts = r.stdout.strip().split(',')
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None

def lcp(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i

async def run_one(path, lang, eou_ms, ref):
    pcm = load_pcm16(path)
    audio_dur = len(pcm) / SR
    se_idx = ref_speech_end(pcm)
    se_s = se_idx / SR
    n_chunks = (len(pcm) + CHUNK_BYTES - 1) // CHUNK_BYTES
    pad_chunks = (eou_ms // CHUNK_MS) + 30
    silent = np.zeros(CHUNK_SAMPLES, dtype=np.int16)

    async with websockets.connect(WS, max_size=50*1024*1024) as ws:
        try:
            await asyncio.wait_for(ws.recv(), 2.0)
        except Exception:
            pass
        await ws.send(json.dumps({'type':'session.update','session':{
            'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        t_ref = time.monotonic()
        async def sender():
            for i in range(n_chunks):
                s=i*CHUNK_BYTES; e=min(s+CHUNK_BYTES,len(pcm))
                tgt=t_ref+(i*CHUNK_MS/1000.0)
                now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            base=n_chunks*CHUNK_MS/1000.0
            for j in range(pad_chunks):
                tgt=t_ref+base+(j*CHUNK_MS/1000.0)
                now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        final={'text':'','t':None}
        partials=[]
        accum=''
        async def receiver():
            nonlocal accum
            while True:
                try:
                    msg=await asyncio.wait_for(ws.recv(),10.0)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg,bytes): continue
                data=json.loads(msg); t=data.get('type','')
                if t=='conversation.item.input_audio_transcription.delta':
                    dl=data.get('delta') or data.get('text') or ''
                    if dl:
                        accum+=dl
                        partials.append((time.monotonic()-t_ref, accum))
                elif t=='conversation.item.input_audio_transcription.completed':
                    final['text']=data.get('transcript') or data.get('text') or ''
                    final['t']=time.monotonic()-t_ref
                    break
                elif t=='error': break
        await asyncio.gather(sender(), receiver())
    final_lat = final['t'] - se_s if final['t'] is not None else None
    # partial stability: LCP of final partial vs final aligned by accumulating
    # compute longest common prefix of each partial vs final text
    final_text = final['text']
    lcp_vals = [lcp(p, final_text) for _,p in partials]
    max_lcp = max(lcp_vals) if lcp_vals else 0
    return {'audio_dur':audio_dur,'se_s':se_s,'final_t':final['t'],
            'final_lat':final_lat,'text':final['text'],'n_part':len(partials),
            'max_lcp':max_lcp,'ref':ref}

async def main():
    corpus=[
        ('/tmp/de.wav','de'),
        ('/tmp/de2.wav','de'),
        ('/tmp/de3.wav','de'),
        ('/tmp/de4.wav','de'),
        ('/tmp/en1.wav','en'),
        ('/tmp/en2.wav','en'),
        ('/tmp/en3.wav','en'),
        ('/tmp/en4.wav','en'),
    ]
    eou=650
    print(f"=== Waterfall at EOU={eou}ms (server rc=320ms), speech_end=ground-truth RMS===",flush=True)
    lats=[]
    for path,lang in corpus:
        r=await run_one(path,lang,eou,'')
        lats.append(r['final_lat'] if r['final_lat'] is not None else 999)
        print(f"{os.path.basename(path)} audio_dur={r['audio_dur']:.2f}s se={r['se_s']:.2f}s final_t={r['final_t']:.3f} final_lat={r['final_lat']:.3f}s n_part={r['n_part']} maxLCP={r['max_lcp']}/{len(r['text'])}",flush=True)
    good=sorted(l for l in lats if l<10)
    print(f"\nMEDIAN speech_end->ASR_final = {good[len(good)//2]:.3f}s  p90={good[int(len(good)*0.9)]:.3f}s  n={len(good)}",flush=True)

asyncio.run(main())
