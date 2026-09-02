"""Full E2E pipeline test on the box: FileReplay audio -> Nemotron ASR -> EOU -> Qwen.

Uses box-local services: ASR ws://127.0.0.1:8090/v1/realtime, Qwen http://127.0.0.1:8081.
Measures the latency waterfall: speech_end -> EOU -> ASR final -> Qwen first token.
"""
import asyncio, json, time, os, sys, subprocess
import numpy as np
import soundfile as sf
import httpx

sys.path.append('/workspace/voiceagent/src')
from voice_agent.audio import FileReplayAudioSource
from voice_agent.nemo_client import NemoClient, ASRConfig, ASRPartial

SR = 16000
ASR_WS = 'ws://127.0.0.1:8090/v1/realtime'
QWEN_URL = 'http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL = '/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM_PROMPT = (
    "You are participating in a live spoken conversation. "
    "Respond naturally and directly. Prefer concise conversational answers. "
    "Do not use elaborate formatting, section headings, tables, or long lists. "
    "The user's message was produced by automatic speech recognition. "
    "Resolve obvious minor transcription errors from context when possible. "
    "Do not mention the speech recognition system."
)

def load_pcm16(path):
    data, sr = sf.read(path, always_2d=False, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        dur = len(data)/sr; tgt = int(dur*SR)
        old = np.linspace(0,1,len(data)); new = np.linspace(0,1,tgt)
        data = np.interp(new, old, data).astype(np.float32)
    data = np.clip(data, -1, 1)
    return (data * 32767).astype(np.int16)

def ref_speech_end(pcm, thr=0.01):
    n = len(pcm); cs = SR*20//1000; nch=(n+cs-1)//cs; last=0
    for i in range(nch):
        s=i*cs; e=min(s+cs,n)
        c=pcm[s:e].astype(np.float32)/32768.0
        if c.size and np.sqrt(np.mean(c**2))>thr: last=i+1
    return min(last*cs, n)

async def qwen_stream(messages, on_token, on_first, on_request):
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "presence_penalty": 1.5, "max_tokens": 128, "stream": True,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first_ts = [None]
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", QWEN_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try:
                    j = json.loads(d)
                    delta = j["choices"][0]["delta"].get("content") or ""
                    if delta:
                        if first_ts[0] is None:
                            first_ts[0] = time.monotonic()
                            on_first(first_ts[0])
                        on_token(delta)
                except: continue

async def run_turn(path, lang, eou_ms):
    pcm = load_pcm16(path)
    total = len(pcm); full_dur = total/SR
    se = ref_speech_end(pcm)/SR
    n_chunks=(total+SR*20//1000-1)//(SR*20//1000)
    cs=SR*20//1000
    silent=np.zeros(cs, dtype=np.int16)
    eou_chunks=eou_ms//20

    t0 = time.monotonic()
    events = []
    def ev(name, **kw):
        events.append({"t": time.monotonic()-t0, "event": name, **kw})

    import websockets
    ev("audio_start")
    async with websockets.connect(ASR_WS, max_size=50*1024*1024) as ws:
        try: await asyncio.wait_for(ws.recv(), 2.0)
        except Exception: pass
        await ws.send(json.dumps({'type':'session.update','session':{'sample_rate':SR,'language':lang,'automatic_punctuation':True}}))
        final_text=''; partials=[]; accum=''
        async def sender():
            for i in range(n_chunks):
                s=i*cs; e=min(s+cs,total)
                tgt=t0+(i*20/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(pcm[s:e].tobytes())
            base=n_chunks*20/1000.0
            for j in range(eou_chunks):
                tgt=t0+base+(j*20/1000.0); now=time.monotonic()
                if now<tgt: await asyncio.sleep(tgt-now)
                await ws.send(silent.tobytes())
            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            await asyncio.sleep(0.1)
        async def receiver():
            nonlocal accum, final_text
            while True:
                try: msg=await asyncio.wait_for(ws.recv(), 12.0)
                except asyncio.TimeoutError: break
                if isinstance(msg, bytes): continue
                d=json.loads(msg); t=d.get('type','')
                if t=='conversation.item.input_audio_transcription.delta':
                    dl=d.get('delta') or d.get('text') or ''
                    if dl:
                        accum+=dl; partials.append((time.monotonic()-t0, accum))
                elif t=='conversation.item.input_audio_transcription.completed':
                    final_text=d.get('transcript') or d.get('text') or ''
                    ev('asr_final', text=final_text)
                    break
                elif t=='error': break
        await asyncio.gather(sender(), receiver())
    asr_final_t = time.monotonic()-t0

    # ~EOU happens at speech_end + eou; final arrive shortly after. log reference
    ev("reference_speech_end", se=se)

    # Now Qwen
    ev("llm_request")
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":final_text}]
    full=[""]; first_tok=[None]
    def on_tok(tok):
        full[0]+=tok
    def on_first(ts):
        first_tok[0]=ts
    await qwen_stream(messages, on_tok, on_first, None)
    full=full[0]; first_tok=first_tok[0]
    ev("llm_done", text=full)

    first_tok_lat = (first_tok - t0) if first_tok else None
    # waterfall relative to speech_end
    wf = {
        'path': path, 'lang': lang, 'dur': round(full_dur,2), 'se': round(se,2),
        'asr_final_abs': round(asr_final_t,3),
        'asr_final_after_se': round(asr_final_t - se, 3),
        'llm_ttft_after_se': round(first_tok_lat - se, 3) if first_tok_lat else None,
        'asr_text': final_text, 'llm_text': full[:120], 'n_partials': len(partials),
    }
    print(f"\n=== {os.path.basename(path)} ({lang}) ===")
    print(f"  duration={full_dur:.2f}s speech_end={se:.2f}s")
    print(f"  partials: {len(partials)}")
    for p in partials: print(f"    ~ {p[1][:70]}")
    print(f"  ✓ ASR final: {final_text}")
    print(f"  asr_final after speech_end: +{wf['asr_final_after_se']*1000:.0f} ms")
    print(f"  llm TTFT after speech_end:  +{wf['llm_ttft_after_se']*1000:.0f} ms" if wf['llm_ttft_after_se'] is not None else f"  llm TTFT: NONE")
    print(f"  assistant: {full[:150]}")
    return wf

async def main():
    corpus=[
        ('/tmp/de4.wav','de-DE'),
        ('/tmp/en3.wav','en-US'),
        ('/tmp/de2.wav','de-DE'),
        ('/tmp/en2.wav','en-US'),
    ]
    results=[]
    for path,lang in corpus:
        r=await run_turn(path,lang,650)
        results.append(r)
    print("\n=== E2E WATERFALL SUMMARY (speech_end -> ASR final -> Qwen TTFT, EOU=650) ===")
    for r in results:
        print(f"{os.path.basename(r['path'])}: ASR_final +{r['asr_final_after_se']*1000:.0f}ms | Qwen TTFT +{r['llm_ttft_after_se']*1000:.0f}ms" if r['llm_ttft_after_se'] is not None else f"{os.path.basename(r['path'])}: ASR_final +{r['asr_final_after_se']*1000:.0f}ms | TTFT NONE")

asyncio.run(main())
