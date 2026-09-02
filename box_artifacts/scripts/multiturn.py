"""Multi-turn Qwen conversation test: 3 chained turns with persistent history.
Turn 1 (DE): ask about capital -> Turn 2 (EN): follow-up referencing turn 1
-> Turn 3 (DE): reference both. Verifies cache_prompt + history persistence.
"""
import asyncio, json, time, os, sys
import httpx

QWEN_URL='http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM_PROMPT=("You are participating in a live spoken conversation. Respond naturally and directly. Prefer concise conversational answers.")

async def qwen_call(messages, t0):
    payload={"model":QWEN_MODEL,"messages":messages,"temperature":0.7,"top_p":0.8,"top_k":20,
             "presence_penalty":1.5,"max_tokens":96,"stream":False,"cache_prompt":True,
             "chat_template_kwargs":{"enable_thinking":False}}
    full=""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r=await client.post(QWEN_URL, json=payload)
        j=r.json()
        full=j["choices"][0]["message"]["content"]
        usage=j.get("usage",{})
        timings=j.get("timings",{})
        ttft=(time.monotonic()-t0)
    return full, usage, timings

async def main():
    history=[{"role":"system","content":SYSTEM_PROMPT}]
    turns=[
        ("de-DE","Wie lautet die Hauptstadt von Deutschland?"),
        ("en-US","And now tell me the population of the city you just mentioned."),
        ("de-DE","Wie gross ist der Rhein und in welche Stadt fliesst er?"),
    ]
    for i,(lang,text) in enumerate(turns,1):
        t0=time.monotonic()
        history.append({"role":"user","content":text})
        full,usage,timings=await qwen_call(history,t0)
        history.append({"role":"assistant","content":full})
        ttft_ms=(time.monotonic()-t0)*1000
        pb=timings.get("prompt_per_second"); gb=timings.get("predicted_per_second")
        cache=timings.get("cache_n")
        print(f"\nTurn {i} ({lang}):")
        print(f"  user:       {text}")
        print(f"  assistant:  {full[:120]}")
        print(f"  prompt_ps={pb}, gen_ps={gb}, cache_n={cache}, cached_tokens={usage.get('prompt_tokens_details')}, wall={ttft_ms:.0f}ms")
        print(f"  (prompt cached from prior turn: {cache if cache is not None else 'n/a'})")

asyncio.run(main())
