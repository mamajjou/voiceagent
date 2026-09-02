"""True EOU latency: stream speech (up to speech_end), pad exactly eou_ms of
silence, commit AT speech_end+eou_ms. Measures the real user-visible latency:
speech_end -> (eoU silence window) -> ASR final.

The file's built-in trailing silence is TRUNCATED: we only stream samples up
to speech_end, then synthesize the EOU silence window ourselves.
"""
import asyncio, json, time, os
import numpy as np
import soundfile as sf
import websockets

WS='ws://127.0.0.1:8090/v1/realtime'
SR=16000; CHUNK_MS=20
CHUNK_SAMPLES=SR*CHUNK_MS//1000
CHUNK_BYTES=CHUNK_SAMPLES*2

def load_pcm16(path, truncate_at):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1)
    data=(data*32767).astype(np.int16)
    return data[:truncate_at]

def ref_speech_end(path,thr=0.01):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1); pcm=(data*32767).astype(np.int16)
    n=len(pcm); nch=(n+CHUNK_SAMPLES-1)//CHUNK_SAMPLES; last=0
    for i in range(nch):
        s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,n)
        c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*CHUNK_SAMPLES,n)

async def run_one(path,lang,eou_ms):
    se=ref_speech_end(path)
    # truncate audio to speech_end (drop file's internal trailing silence)
    pcm=load_pcm16(path, se)
    total=len(pcm); se_s=se/SR
    n_chunks=(total+CHUNK_SAMPLES-1)//CHUNK_SAMPLES
    eou_chunks=int(eou_ms/CHUNK_MS)
    silent=np.zeros(CHUNK_SAMPLES,dtype=np.int16)

    async with websockets.connect(WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{
            'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        t_ref=time.monotonic()
        async def sender():
            # stream speech chunks (paced)
            for i in range(n_chunks):
                s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,total)
                tgt=t_ref+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            # stream exactly eou_ms of silence, then always commit
            base=n_chunks*CHUNK_MS/1000.0
            for j in range(max(eou_chunks,1)):
                tgt=t_ref+base+(j*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
                await asyncio.sleep(0.001)
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        final={'text':'','t':None}
        async def receiver():
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),10.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.completed':
                    final['text']=d.get('transcript') or d.get('text') or ''
                    final['t']=time.monotonic()-t_ref; break
                elif t=='error': break
        await asyncio.gather(sender(),receiver())
    lat=final['t']-se_s if final['t'] is not None else None
    return se_s, final['t'], lat, final['text']

    # Note: if final['t'] is None (stream timed out), lat is None.

async def main():
    corpus=[('/tmp/de.wav','de'),('/tmp/de2.wav','de'),('/tmp/de3.wav','de'),
            ('/tmp/en1.wav','en'),('/tmp/en2.wav','en'),('/tmp/en3.wav','en'),('/tmp/en4.wav','en')]
    print("== TRUE EOU latency (stream speech -> eou silence -> commit) ==",flush=True)
    summary={}
    for eou in [0,350,650,1000]:
        vals=[]; print(f"\n-- EOU={eou}ms --",flush=True)
        for path,lang in corpus:
            se,ft,lat,text=await run_one(path,lang,eou)
            vals.append(lat if lat is not None else 999)
            lt = f"{lat:.3f}" if lat is not None else "NONE"
            ftx = f"{ft:.3f}" if ft is not None else "NONE"
            print(f"{os.path.basename(path)} se={se:.2f} f_t={ftx} lat={lt} '{text[:40]}'",flush=True)
        g=sorted(v for v in vals if v<10)
        summary[eou]=(g[len(g)//2], g[int(len(g)*0.9)])
        print(f"MEDIAN {g[len(g)//2]:.3f}s p90={g[int(len(g)*0.9)]:.3f}s",flush=True)
    print("\n=== FINAL EOU LATENCY (speech_end -> ASR final), rc=320ms ===",flush=True)
    for e,(med,p90) in summary.items():
        print(f"EOU {e}ms -> median {med:.3f}s  p90 {p90:.3f}s",flush=True)

asyncio.run(main())
