"""Sweep EOU window across corpus, measure speech_end->ASR_final and
speech_end->Qwen_TTFT. Uses box-local ASR+Qwen. Reports median/p90/p95.
"""
import asyncio, json, time, os, sys, subprocess
import numpy as np
import soundfile as sf
import websockets
import httpx

SR=16000; CHUNK_MS=20; cs=SR*CHUNK_MS//1000
ASR_WS='ws://127.0.0.1:8090/v1/realtime'
QWEN_URL='http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM_PROMPT=("You are participating in a live spoken conversation. Respond naturally and directly. Prefer concise conversational answers.")
CORPUS=[
    ('/tmp/de.wav','de-DE'),('/tmp/de2.wav','de-DE'),('/tmp/de3.wav','de-DE'),
    ('/tmp/de4.wav','de-DE'),('/tmp/en1.wav','en-US'),('/tmp/en2.wav','en-US'),
    ('/tmp/en3.wav','en-US'),('/tmp/en4.wav','en-US'),
]

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

async def asr_turn(path,lang,eou_ms,t0,se):
    pcm=load_pcm16(path); total=len(pcm)
    n_chunks=(total+cs-1)//cs; eou_chunks=eou_ms//CHUNK_MS
    silent=np.zeros(cs,dtype=np.int16); E={}; st=time.monotonic()
    def tik(k): E[k]=time.monotonic()-t0
    async with websockets.connect(ASR_WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        final=''; speech_chunks=int((se*SR)//cs)
        commit_wall=t0+se+eou_ms/1000.0
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
                    tik('commit'); await ws.send(json.dumps({'type':'input_audio_buffer.commit'})); break
                tgt=min(t0+speech_chunks*CHUNK_MS/1000.0+(j*CHUNK_MS/1000.0), commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes()); j+=1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''; tik('asr_final'); break
                elif t=='error': break
        await asyncio.gather(sender(),receiver())
    return final, E

async def qwen_turn(final,lang,t0):
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":final}]
    payload={"model":QWEN_MODEL,"messages":messages,"temperature":0.7,"top_p":0.8,"top_k":20,
             "presence_penalty":1.5,"max_tokens":96,"stream":True,"cache_prompt":True,
             "chat_template_kwargs":{"enable_thinking":False}}
    first=None; full=[""]
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST",QWEN_URL,json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    j=json.loads(d); delta=j["choices"][0]["delta"].get("content") or ""
                    if delta:
                        if first is None: first=time.monotonic()-t0
                        full[0]+=delta
                except: continue
    return first, full[0]

async def main():
    print("EOU sweep (box-local): speech_end -> ASR_final -> Qwen_TTFT\n")
    print(f"{'EOU':>5} {'ASRf_med':>9} {'ASRf_p90':>9} {'TTFT_med':>9} {'TTFT_p90':>9}")
    for eou in [350,500,650,800,1000]:
        af=[]; tt=[]
        for path,lang in CORPUS:
            pcm=load_pcm16(path); se=speech_end(pcm)/SR
            t0=time.monotonic()
            final,E=await asr_turn(path,lang,eou,t0,se)
            afs=(E['asr_final']-se)*1000 if 'asr_final' in E else 9999
            af.append(afs)
            first,llm=await qwen_turn(final,lang,t0)
            tts=(first-se)*1000 if first else 9999
            tt.append(tts)
        af=sorted(af); tt=sorted(tt)
        print(f"{eou:>5} {af[len(af)//2]:>9.0f} {af[int(len(af)*0.9)]:>9.0f} {tt[len(tt)//2]:>9.0f} {tt[int(len(tt)*0.9)]:>9.0f}")

asyncio.run(main())
