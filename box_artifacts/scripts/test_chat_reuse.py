"""Check whether /v1/chat/completions reuses growing user-prefix KV with thinking off.

We send a chat request with the same system + a growing user message, incrementally.
cache_prompt:true. Measure cache_n growth. If cache_n grows, the user-prefix IS reused
and we can use the chat endpoint (which properly disables thinking).
"""
import asyncio, json, time, httpx
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM="You are a helpful assistant. Answer concisely."
REPLY="\nAssistant:"

def payload(messages, n_pred=1):
    return {"model":MODEL,"messages":messages,"max_tokens":n_pred,
            "temperature":0.7,"cache_prompt":True,
            "chat_template_kwargs":{"enable_thinking":False}}

async def chat(prompt_user, n_pred=1):
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt_user}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        r=await client.post(f"{BASE}/v1/chat/completions", json=payload(messages,n_pred))
        j=r.json()
    content=j["choices"][0]["message"]["content"]
    t=j.get("timings",{})
    return content, t

async def main():
    evo=["What","What is","What is the","What is the capital","What is the capital of France?"]
    print("chat endpoint growing-prefix reuse (thinking off):")
    for p in evo:
        c,t=await chat(p)
        print(f"  '{p:<32}' cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms'):.1f} content={c!r}")
    # full commit
    c,t=await chat("What is the capital of France?")
    print(f"\n  FULL cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms'):.1f} content={c!r}")

asyncio.run(main())
