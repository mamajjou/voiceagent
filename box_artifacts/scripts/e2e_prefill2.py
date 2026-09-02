"""End-to-end prefill-while-speaking (via /v1/chat/completions, thinking off).

While the user speaks, ASR emits partials. For each partial, fire a short chat
prefill that warms Qwen KV for (system + current partial user text). At EOU,
send the final text; llama.cpp reuses the cached growing user-prefix so the
full prompt is mostly prefilled -> fast first token.

Measures speech_end -> first-token WITH prefill-while-speaking vs baseline.
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
SYSTEM="You are participating in a live spoken conversation. Answer concisely."

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

async def chat_prefill(user_text):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":user_text}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            await client.post(f"{BASE}/v1/chat/completions", json={
                "model":MODEL,"messages":msgs,"max_tokens":1,"cache_prompt":True,
                "temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}})
        except Exception:
            pass

async def chat_generate(user_text, t0):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":user_text}]
    first=None; full=""
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{BASE}/v1/chat/completions", json={
            "model":MODEL,"messages":msgs,"max_tokens":64,"cache_prompt":True,
            "temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    # non-streamed single JSON blob (response too short to stream)
                    try:
                        j=json.loads(line)
                        c=j["choices"][0]["message"].get("content") or ""
                        if c:
                            if first is None: first=time.monotonic()-t0
                            full=c
                    except: pass
                    continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    j=json.loads(d)
                    delta=j["choices"][0]["delta"].get("content") or ""
                    if delta:
                        if first is None: first=time.monotonic()-t0
                        full+=delta
                except: pass
    return first, full

async def run(path, lang, eou_ms, use_prefill):
    pcm=load_pcm16(path); total=len(pcm)
    se=speech_end(pcm)/SR; speech_chunks=int((se*SR)//cs)
    silent=np.zeros(cs,dtype=np.int16)
    commit_wall=time.monotonic()+se+eou_ms/1000.0
    t0=time.monotonic(); final=''; partials_ts=[]
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
            accum=''
            recent_partials=[]  # (text) recent partials for stable-prefix LCP
            last_total=0; last_prefill_t=0.0  # track stable-prefix length checkpoints
            def stab_len():
                # longest common prefix of the most recent partials -> the STABLE prefix
                if not recent_partials: return len(accum)
                base=recent_partials[0]
                for p in recent_partials[1:]:
                    i=0
                    for a,b in zip(base,p):
                        if a==b: i+=1
                        else: break
                    base=base[:i]
                return len(base)
            while True:
                try: msg=await asyncio.wait_for(ws.recv(),12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg,bytes): continue
                d=json.loads(msg); tt=d.get('type','')
                if tt=='conversation.item.input_audio_transcription.delta':
                    dl=d.get('delta') or d.get('text') or ''
                    if dl:
                        accum+=dl; partials_ts.append(time.monotonic()-t0)
                        recent_partials.append(accum)
                        if len(recent_partials)>4: recent_partials.pop(0)
                        if use_prefill:
                            # THROTTLE + STABLE PREFIX: only prefill when the stable
                            # prefix has grown AND >=250ms since last prefill.
                            now=time.monotonic()
                            sl=stab_len()
                            if sl-last_total>=8 and (now-last_prefill_t)>=0.25:
                                stable_text=accum[:sl]
                                asyncio.create_task(chat_prefill(stable_text))
                                last_total=sl; last_prefill_t=now
                elif tt=='conversation.item.input_audio_transcription.completed':
                    final=d.get('transcript') or d.get('text') or ''; break
                elif tt=='error': break
        await asyncio.gather(sender(), receiver())
    if use_prefill: await asyncio.sleep(0.2)
    first, full = await chat_generate(final, t0)
    return {'se':se,'asr_final_after_se':(partials_ts[-1]-se) if partials_ts else None,
            'ttft_after_se':(first-se) if first else None,'text':final[:35],'llm':full[:35],'n_partials':len(partials_ts)}

async def main():
    clips=[('/tmp/en3.wav','en-US'),('/tmp/de4.wav','de-DE'),('/tmp/en2.wav','en-US')]
    print("=== E2E PREFILL-WHILE-SPEAKING vs BASELINE (chat, thinking off) ===\n")
    print(f"{'clip':<10} {'mode':<8} {'ASR_afterSE':>11} {'TTFT_afterSE':>13} {'saved':>7}  answer")
    for path,lang in clips:
        row={}
        for use,label in [(False,'baseline'),(True,'prefill ')]:
            r=await run(path,lang,650,use)
            row[label]=r
            af=r['asr_final_after_se']; tt=r['ttft_after_se']
            af_s=f"{af*1000:.0f}" if af is not None else "-"
            tt_s=f"{tt*1000:.0f}" if tt is not None else "-"
            print(f"{os.path.basename(path):<10} {label:<8} {af_s:>11} {tt_s:>13}  {r['llm']!r}")
        if 'baseline' in row and 'prefill ' in row and row['baseline']['ttft_after_se'] and row['prefill ']['ttft_after_se']:
            print(f"  -> SAVED {(row['baseline']['ttft_after_se']-row['prefill ']['ttft_after_se'])*1000:.0f} ms")
        print()

asyncio.run(main())
