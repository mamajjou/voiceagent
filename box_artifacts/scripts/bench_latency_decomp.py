"""Precise latency decomposition per turn with timing instrumentation.

Measures: audio_start, commit_wall (speech_end+eou), asr_final, llm_request,
llm_first_token, llm_done — all relative to t0, plus speech_end (0.01 RMS).
Reports the FULL waterfall including the gap between ASR-final and Qwen TTFT.
"""
import asyncio, json, time, os, sys
import numpy as np
import soundfile as sf
import websockets
import httpx

SR=16000; CHUNK_MS=20; cs=SR*CHUNK_MS//1000
ASR_WS='ws://127.0.0.1:8090/v1/realtime'
QWEN_URL='http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM_PROMPT=("You are participating in a live spoken conversation. Respond naturally and directly. Prefer concise conversational answers.")

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

async def run_one(path,lang,eou_ms):
    pcm=load_pcm16(path); total=len(pcm)
    se=speech_end(pcm)/SR
    n_chunks=(total+cs-1)//cs; eou_chunks=eou_ms//CHUNK_MS
    silent=np.zeros(cs,dtype=np.int16); E={}
    t0=time.monotonic()
    def tik(k): E[k]=time.monotonic()-t0
    import websockets
    async with websockets.connect(ASR_WS,max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(),2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        final=''
        speech_chunks=int((se*SR)//cs)  # only stream up to speech_end, drop file's trailing silence
        commit_wall=t0+se+eou_ms/1000.0
        async def sender():
            for i in range(speech_chunks):
                s=i*cs; e=min(s+cs,total)
                tgt=t0+(i*CHUNK_MS/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            # stream silence until commit_wall, then commit
            j=0
            while True:
                now=time.monotonic()
                if now>=commit_wall:
                    tik('commit')
                    await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
                    break
                tgt=min(t0+speech_chunks*CHUNK_MS/1000.0+(j*CHUNK_MS/1000.0), commit_wall)
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
                j+=1
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal final
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''
                    tik('asr_final'); break
                elif t=='error': break
        await asyncio.gather(sender(),receiver())
    eou_detected = E.get('asr_final', None)
    tik('llm_request')
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":final}]
    payload={"model":QWEN_MODEL,"messages":messages,"temperature":0.7,"top_p":0.8,"top_k":20,
             "presence_penalty":1.5,"max_tokens":96,"stream":True,"cache_prompt":True,
             "chat_template_kwargs":{"enable_thinking":False}}
    full=[""]; first=None
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
                        if first is None: first=time.monotonic()-t0; tik('llm_first')
                        full[0]+=delta
                except: continue
    tik('llm_done')
    return {'path':path,'se':se,**E,'llm_text':full[0][:120],
            'asr_final_after_se':(E['asr_final']-se)*1000,
            'ttft_after_se':(first-se)*1000 if first else None,
            'qwen_ttft_gap':(first-E['asr_final'])*1000 if first and 'asr_final' in E else None}

async def main():
    corpus=[('/tmp/de.wav','de-DE'),('/tmp/en3.wav','en-US')]
    allr=[]
    for eou in [650]:
        print(f"\n===== LATENCY DECOMPOSITION EOU={eou}ms =====")
        for path,lang in corpus:
            r=await run_one(path,lang,eou)
            allr.append(r)
            print(f"\n{os.path.basename(path)}:")
            print(f"  speech_end:        {r['se']*1000:>6.0f} ms")
            print(f"  commit (se+EOU):   {r.get('commit',0)*1000:>6.0f} ms")
            print(f"  ASR final:         {r.get('asr_final',0)*1000:>6.0f} ms  (after SE {r['asr_final_after_se']:.0f}ms)")
            print(f"  LLM request:       {r.get('llm_request',0)*1000:>6.0f} ms")
            print(f"  LLM first token:   {r.get('llm_first',0)*1000:>6.0f} ms  (after SE {r['ttft_after_se']:.0f}ms)")
            print(f"  LLM done:          {r.get('llm_done',0)*1000:>6.0f} ms")
            print(f"  -> GPU-free gap ASR_final->first: {r['qwen_ttft_gap']:.0f} ms")
            print(f"  assistant: {r['llm_text']}")

asyncio.run(main())
