"""Measure VRAM/GPU during concurrent streaming inference on the server."""
import asyncio, json, time, os, subprocess, threading
import numpy as np
import soundfile as sf
import websockets

WS = 'ws://127.0.0.1:8090/v1/realtime'
SR=16000; CHUNK_MS=20; CHUNK_SAMPLES=SR*CHUNK_MS//1000; CHUNK_BYTES=CHUNK_SAMPLES*2

def gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        p=[x.strip() for x in r.stdout.strip().split(',')]
        ival=lambda x: int(float(x))
        return {'mem_used':ival(p[0]),'mem_total':ival(p[1]),'util':ival(p[2]),'power':ival(p[3]),'temp':ival(p[4])}
    except Exception as e:
        return {'err':str(e)}

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1)
    return (data*32767).astype(np.int16)

async def stream_one(path, lang, samples):
    pcm=load_pcm16(path)
    n_chunks=(len(pcm)+CHUNK_BYTES-1)//CHUNK_BYTES
    silent=np.zeros(CHUNK_SAMPLES,dtype=np.int16)
    async with websockets.connect(WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang}}))
        t_ref=time.monotonic()
        async def sender():
            for i in range(n_chunks):
                s=i*CHUNK_BYTES; e=min(s+CHUNK_BYTES,len(pcm))
                tgt=t_ref+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            # pad silence + commit
            base=n_chunks*CHUNK_MS/1000.0
            for j in range(40):
                tgt=t_ref+base+(j*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        async def receiver():
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),10.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg)
                if d.get('type')=='conversation.item.input_audio_transcription.completed': break
                if d.get('type')=='error': break
        await asyncio.gather(sender(),receiver())

async def main():
    corpus=[('/tmp/de.wav','de'),('/tmp/en1.wav','en')]
    # baseline idle
    print("IDLE:", gpu(), flush=True)
    peak_mem=0; peak_util=0; peak_power=0; peak_temp=0
    stop=False
    def sampler():
        nonlocal peak_mem,peak_util,peak_power,peak_temp
        while not stop:
            g=gpu()
            if 'err' not in g:
                peak_mem=max(peak_mem,g['mem_used']); peak_util=max(peak_util,g['util'])
                peak_power=max(peak_power,g['power']); peak_temp=max(peak_temp,g['temp'])
            time.sleep(0.08)
    th=threading.Thread(target=sampler); th.start()
    for path,lang in corpus:
        for r in range(3):
            await stream_one(path,lang,None)
    stop=True; th.join()
    print(f"\nPEAK during 6 streams: mem={peak_mem}MiB util={peak_util}% power={peak_power}W temp={peak_temp}C", flush=True)

asyncio.run(main())
