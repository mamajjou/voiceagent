"""Measure GPU util: ASR streaming alone vs ASR+Qwen prefill.

Determines if there's headroom for prefill-while-speaking without dropping ASR
below realtime. Sampling util + VRAM + a lighter Qwen prefill variant.
"""
import asyncio, json, time, os, subprocess, threading
import numpy as np
import soundfile as sf
import websockets
import httpx

SR=16000; CHUNK_MS=20; cs=SR*CHUNK_MS//1000
ASR_WS='ws://127.0.0.1:8090/v1/realtime'
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'

def gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
        p=[x.strip() for x in r.stdout.strip().split(',')]
        ival=lambda x:int(float(x))
        return {'mem':ival(p[0]),'util':ival(p[1]),'power':ival(p[2])}
    except Exception:
        return {'err':1}

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

async def asr_stream(path, lang, eou_ms, qwen_prefill):
    pcm=load_pcm16(path); total=len(pcm)
    se=speech_end(pcm)/SR; speech_chunks=int((se*SR)//cs)
    silent=np.zeros(cs,dtype=np.int16)
    commit_wall=time.monotonic()+se+eou_ms/1000.0
    t0=time.monotonic(); final=''
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
            nonlocal final
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); tt=d.get('type','')
                if tt=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''; break
                elif tt=='error': break
        async def qwen_loop():
            if not qwen_prefill: return
            async with httpx.AsyncClient(timeout=60.0) as client:
                for k in range(3):
                    await client.post(f"{BASE}/completion", json={
                        "model":MODEL,"prompt":f"You are helpful.\nUser: What is the capital of France? ({k})\nAssistant:",
                        "max_tokens":1,"cache_prompt":True,"temperature":0.7})
                    await asyncio.sleep(0.05)
        await asyncio.gather(sender(), receiver(), qwen_loop())
    wall=time.monotonic()-t0
    return total/SR/wall  # realtime factor

async def sample_util(path, lang, qwen_prefill):
    """Sample GPU util during a streaming run."""
    peak_util=[0]; peak_power=[0]
    def sampler():
        t_end=time.monotonic()+6
        while time.monotonic()<t_end:
            g=gpu()
            if 'err' not in g:
                peak_util[0]=max(peak_util[0],g['util'])
                peak_power[0]=max(peak_power[0],g['power'])
            time.sleep(0.05)
    th=threading.Thread(target=sampler); th.start()
    await asyncio.sleep(0.2)
    rt=await asyncio.create_task(asyncio.to_thread(asyncio.run, asr_stream(path,lang,650,qwen_prefill)))
    # not quite right; just do inline
    return None

async def main():
    clips=[('/tmp/en3.wav','en-US'),('/tmp/de4.wav','de-DE')]
    print("=== GPU UTIL: ASR solo vs ASR + Qwen prefill ===\n")
    print(f"{'clip':<10} {'mode':<8} {'RT':>6} {'peak_util':>10} {'peak_W':>7}")
    for path,lang in clips:
        for qp,label in [(False,'solo'),(True,'qwen')]:
            # warm up qwen cache first so the 3 prefills are quick
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    await client.post(f"{BASE}/completion", json={"model":MODEL,"prompt":"You are helpful.\nUser: hi\nAssistant:","max_tokens":1,"cache_prompt":True})
                except: pass
            peak_util=[0]; peak_power=[0]
            stop=[False]
            def sampler():
                while not stop[0]:
                    g=gpu()
                    if 'err' not in g:
                        peak_util[0]=max(peak_util[0],g['util'])
                        peak_power[0]=max(peak_power[0],g['power'])
                    time.sleep(0.03)
            th=threading.Thread(target=sampler); th.start()
            rt=await asr_stream(path,lang,650,qp)
            stop[0]=True; th.join()
            print(f"{os.path.basename(path):<10} {label:<8} {rt:>6.2f} {peak_util[0]:>10} {peak_power[0]:>7}")

asyncio.run(main())
