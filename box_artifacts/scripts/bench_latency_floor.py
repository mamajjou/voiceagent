"""True conversational latency floor.

Orchestrator model:
  - streams speech at realtime (20ms chunks)
  - TurnManager detects speech_end (energy VAD)
  - waits eou_ms of trailing silence, THEN commits
  - server finalizes

Measures speech_end -> commit -> ASR_final for each eou window. This is the
real latency a user experiences (sans Qwen).

Also reports: latency floor with commit-at-speech_end (eou=0).
"""
import asyncio, json, time, os
import numpy as np
import soundfile as sf
import websockets

WS='ws://127.0.0.1:8090/v1/realtime'
SR=16000; CHUNK_MS=20
CHUNK_SAMPLES=SR*CHUNK_MS//1000
CHUNK_BYTES=CHUNK_SAMPLES*2

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1)
    return (data*32767).astype(np.int16)

def ref_speech_end(pcm,thr=0.01):
    n=len(pcm); nch=(n+CHUNK_SAMPLES-1)//CHUNK_SAMPLES; last=0
    for i in range(nch):
        s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,n)
        c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*CHUNK_SAMPLES,n)

async def run_one(path,lang,eou_ms):
    pcm=load_pcm16(path)
    total=len(pcm); dur=total/SR
    se=ref_speech_end(pcm)/SR
    n_chunks=(total+CHUNK_SAMPLES-1)//CHUNK_SAMPLES
    silent=np.zeros(CHUNK_SAMPLES,dtype=np.int16)

    async with websockets.connect(WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{
            'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        t_ref=time.monotonic()
        # commit wall time = speech_end(in wall) + eou
        commit_wall=t_ref+se+eou_ms/1000.0
        async def sender():
            for i in range(n_chunks):
                s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,total)
                tgt=t_ref+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                # stall if we've passed commit point while still sending (shouldn't happen)
                await ws.send(pcm[s:e].tobytes())
            base=n_chunks*CHUNK_MS/1000.0
            # keep streaming silence just until commit_wall, then commit
            j=0
            while True:
                now=time.monotonic()
                if now>=commit_wall:
                    await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
                    break
                tgt=min(t_ref+base+(j*CHUNK_MS/1000.0), commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
                j+=1
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
    return dur, se, final['t'], final['text']

async def main():
    corpus=[('/tmp/de.wav','de'),('/tmp/de2.wav','de'),('/tmp/de3.wav','de'),
            ('/tmp/en1.wav','en'),('/tmp/en2.wav','en'),('/tmp/en3.wav','en'),('/tmp/en4.wav','en')]
    print("== True latency floor vs EOU window (rc=320ms) ==",flush=True)
    for eou in [0,350,650,1000]:
        vals=[]
        print(f"\n-- EOU={eou}ms --",flush=True)
        for path,lang in corpus:
            dur,se,f_t,text=await run_one(path,lang,eou)
            lat=f_t-se if f_t is not None else None
            vals.append(lat if lat is not None else 999)
            print(f"{os.path.basename(path)} se={se:.2f} f_t={f_t:.3f} lat={lat:.3f}",flush=True)
        g=sorted(v for v in vals if v<10)
        print(f"MEDIAN {g[len(g)//2]:.3f}s p90={g[int(len(g)*0.9)]:.3f}s",flush=True)

asyncio.run(main())
