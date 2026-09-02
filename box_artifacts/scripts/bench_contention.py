"""Measure GPU contention: does Qwen prefill-during-speech hurt ASR streaming?

Run ASR streaming on a clip while concurrently doing Qwen prefills. Compare
the ASR realtime factor and the ASR final latency vs. solo (no Qwen pressure).

This measures the DOWNSIDE of prefill-while-speaking on a single 3090.
"""
import asyncio, json, time, os
import numpy as np
import soundfile as sf
import websockets
import httpx

SR=16000; CHUNK_MS=20; cs=SR*CHUNK_MS//1000
ASR_WS='ws://127.0.0.1:8090/v1/realtime'
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1); return (data*32767).astype(np.int16)

def speech_end(pcm,thr=0.01):
    n=len(pcm); nch=(n+cs-1)//cs; last=0
    for i in range(nch):
        s=i*cs; e=min(s+cs,n); c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*cs,n)

async def asr_stream(path, lang, eou_ms, qwen_pressure: bool):
    pcm=load_pcm16(path); total=len(pcm)
    se=speech_end(pcm)/SR; speech_chunks=int((se*SR)//cs)
    silent=np.zeros(cs,dtype=np.int16)
    commit_wall=time.monotonic()+se+eou_ms/1000.0
    t0=time.monotonic(); final=''; last_partial_t=None
    async with websockets.connect(ASR_WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang}}))
        async def sender():
            for i in range(speech_chunks):
                s=i*cs; e=min(s+cs,total)
                tgt=t0+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            j=0
            while True:
                now=time.monotonic()
                if now>=commit_wall:
                    await ws.send(json.dumps({'type':'input_audio_buffer.commit'})); break
                tgt=min(t0+speech_chunks*CHUNK_MS/1000.0+(j*CHUNK_MS/1000.0),commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes()); j+=1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final, last_partial_t
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); tt=d.get('type','')
                if tt=='conversation.item.input_audio_transcription.delta':
                    last_partial_t=time.monotonic()-t0
                elif tt=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''; break
                elif tt=='error': break
        # Qwen pressure task if enabled
        async def qwen_loop():
            if not qwen_pressure: return
            # continuously prefill a growing-ish prompt to hold GPU
            async with httpx.AsyncClient(timeout=60.0) as client:
                for k in range(20):
                    try:
                        await client.post(f"{BASE}/completion", json={
                            "model":MODEL,"prompt":f"You are helpful.\nUser: What is the capital of France? {k}\nAssistant:",
                            "max_tokens":1,"cache_prompt":True,"temperature":0.7})
                    except: pass
                    await asyncio.sleep(0.02)
        await asyncio.gather(sender(), receiver(), qwen_loop())
    wall=time.monotonic()-t0
    audio_dur=total/SR
    return {'realtime_factor': audio_dur/wall, 'wall':wall, 'audio_dur':audio_dur,
            'asr_final': final, 'se':se, 'asr_final_after_se':(last_partial_t or 0)-se}

async def main():
    clips=[('/tmp/en3.wav','en-US'),('/tmp/de4.wav','de-DE')]
    print("=== GPU CONTENTION TEST: Qwen prefill during ASR streaming ===\n")
    for path,lang in clips:
        for pressure in [False, True]:
            r=await asr_stream(path,lang,650,pressure)
            label='solo  ' if not pressure else 'qwen+  '
            print(f"{os.path.basename(path):<12} {label} RT={r['realtime_factor']:.2f}x  wall={r['wall']:.2f}s  asr_after_se={r['asr_final_after_se']*1000:.0f}ms")

asyncio.run(main())
