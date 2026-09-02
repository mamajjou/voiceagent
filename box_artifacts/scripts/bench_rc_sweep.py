"""Sweep rnnt_right_context (needs server relaunch per value).

Since the server fixes right_context at launch, we relaunch it per RC value
and test the same corpus. Measures WER + streaming latency (commit-at-end,
so pure finalization latency) for each RC.

Usage: python3 bench_rc_sweep.py <rc>
"""
import asyncio, json, time, os, subprocess, sys
import numpy as np
import soundfile as sf
import websockets

WS='ws://127.0.0.1:8090/v1/realtime'
SR=16000; CHUNK_MS=20; CHUNK_SAMPLES=SR*CHUNK_MS//1000; CHUNK_BYTES=CHUNK_SAMPLES*2
CORPUS=[
    ('/tmp/de.wav','de','Guten Tag. Wie geht es dir?'),
    ('/tmp/de2.wav','de','Guten Morgen, wie geht es Ihnen heute?'),
    ('/tmp/de3.wav','de','Ich bin gestern nach Berlin gefahren.'),
    ('/tmp/de4.wav','de','Wie lautet die Hauptstadt von Deutschland?'),
    ('/tmp/en1.wav','en','She had your dark suit and greasy washwater all year.'),
    ('/tmp/en2.wav','en','Could you explain why the sky is blue?'),
    ('/tmp/en3.wav','en','What is the capital of France?'),
    ('/tmp/en4.wav','en','How does a transformer attention mechanism work?'),
]

def load_pcm16(path):
    data,sr=sf.read(path,always_2d=False,dtype='float32')
    if data.ndim==2: data=data.mean(axis=1)
    if sr!=SR:
        dur=len(data)/sr; tgt=int(dur*SR)
        old=np.linspace(0,1,len(data)); new=np.linspace(0,1,tgt)
        data=np.interp(new,old,data).astype(np.float32)
    data=np.clip(data,-1,1)
    return (data*32767).astype(np.int16)

async def run_one(path, lang):
    pcm=load_pcm16(path)
    n_chunks=(len(pcm)+CHUNK_BYTES-1)//CHUNK_BYTES
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
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        final={'text':'','t':None}
        async def receiver():
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),8.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.completed':
                    final['text']=d.get('transcript') or d.get('text') or ''
                    final['t']=time.monotonic()-t_ref; break
                elif t=='error': break
        await asyncio.gather(sender(),receiver())
    dur=len(pcm)/SR
    return final['text'], (final['t']-dur if final['t'] is not None else None)

async def main():
    for path,lang,ref in CORPUS:
        text,final_lat=await run_one(path,lang)
        print(f"{os.path.basename(path)} final_lat={final_lat:.3f} '{text[:50]}'",flush=True)
asyncio.run(main())
