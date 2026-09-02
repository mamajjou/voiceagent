"""CORRECT latency waterfall harness.

Fixes chunking: index the int16 sample array by SAMPLE offsets (CHUNK_SAMPLES),
not byte offsets. Paces at realtime. Uses reference speech-end from RMS.

Reports per clip: audio_dur, speech_end(RMS>=0.01), EOU window, final_t,
speech_end->final latency, partial stability (max LCP vs final).

Server is fixed at rnnt_right_context=3 (320ms) via launch flag.
"""
import asyncio, json, time, os, subprocess
import numpy as np
import soundfile as sf
import websockets

WS='ws://127.0.0.1:8090/v1/realtime'
SR=16000
CHUNK_MS=20
CHUNK_SAMPLES=SR*CHUNK_MS//1000      # 320 samples
CHUNK_BYTES=CHUNK_SAMPLES*2          # 640 bytes

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1)
    return (data*32767).astype(np.int16)

def ref_speech_end(pcm, thr=0.01):
    "last frame with RMS>thr; returns sample index of its end"
    n=len(pcm); nch=(n+CHUNK_SAMPLES-1)//CHUNK_SAMPLES; last=0
    for i in range(nch):
        s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,n)
        c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr:
            last=i+1
    return min(last*CHUNK_SAMPLES, n)

def lcp(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    return i

async def run_one(path, lang, eou_ms):
    pcm=load_pcm16(path)
    total_samples=len(pcm)
    audio_dur=total_samples/SR
    se_idx=ref_speech_end(pcm)
    se_s=se_idx/SR
    n_chunks=(total_samples+CHUNK_SAMPLES-1)//CHUNK_SAMPLES
    pad_chunks=max((eou_ms//CHUNK_MS)+30, 60)
    silent=np.zeros(CHUNK_SAMPLES,dtype=np.int16)

    async with websockets.connect(WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{
            'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        t_ref=time.monotonic()
        async def sender():
            for i in range(n_chunks):
                s=i*CHUNK_SAMPLES; e=min(s+CHUNK_SAMPLES,total_samples)
                tgt=t_ref+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            base=n_chunks*CHUNK_MS/1000.0
            for j in range(pad_chunks):
                tgt=t_ref+base+(j*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        final={'text':'','t':None}; partials=[]; accum=''
        async def receiver():
            nonlocal accum
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),10.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.delta':
                    dl=d.get('delta') or d.get('text') or ''
                    if dl:
                        accum+=dl
                        partials.append((time.monotonic()-t_ref,accum))
                elif t=='conversation.item.input_audio_transcription.completed':
                    final['text']=d.get('transcript') or d.get('text') or ''
                    final['t']=time.monotonic()-t_ref; break
                elif t=='error': break
        await asyncio.gather(sender(),receiver())
    final_lat=final['t']-se_s if final['t'] is not None else None
    maxlcp=max((lcp(p,final['text']) for _,p in partials), default=0)
    return {'dur':audio_dur,'se':se_s,'final_t':final['t'],'lat':final_lat,
            'text':final['text'],'n_part':len(partials),'maxlcp':maxlcp}

async def main():
    corpus=[
        ('/tmp/de.wav','de'),('/tmp/de2.wav','de'),('/tmp/de3.wav','de'),
        ('/tmp/de4.wav','de'),('/tmp/en1.wav','en'),('/tmp/en2.wav','en'),
        ('/tmp/en3.wav','en'),('/tmp/en4.wav','en'),
    ]
    lats={}
    for eou in [350,650,1000]:
        vals=[]
        print(f"\n===== EOU={eou}ms =====",flush=True)
        for path,lang in corpus:
            r=await run_one(path,lang,eou)
            vals.append(r['lat'] if r['lat'] is not None else 999)
            print(f"{os.path.basename(path)} dur={r['dur']:.2f} se={r['se']:.2f} f_t={r['final_t']:.3f} lat={r['lat']:.3f} part={r['n_part']} LCP={r['maxlcp']}/{len(r['text'])}",flush=True)
        good=sorted(v for v in vals if v<10)
        med=good[len(good)//2]
        lats[eou]=med
        print(f"  -> MEDIAN {med:.3f}s  p90={good[int(len(good)*0.9)]:.3f}s",flush=True)
    print("\n=== CLEAN SWEEP (speech_end->ASR_final, rc=320ms) ===",flush=True)
    for e,v in lats.items(): print(f"EOU {e}ms -> median {v:.3f}s",flush=True)

asyncio.run(main())
