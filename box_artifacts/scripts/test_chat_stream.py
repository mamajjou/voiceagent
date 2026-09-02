import asyncio, json, time, httpx
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM="You are participating in a live spoken conversation. Answer concisely."

async def main():
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":"What is the capital of France?"}]
    t0=time.monotonic(); first=None; full=""; nlines=0
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{BASE}/v1/chat/completions", json={
            "model":MODEL,"messages":msgs,"max_tokens":32,"cache_prompt":True,
            "temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}}) as resp:
            print("status", resp.status_code)
            async for line in resp.aiter_lines():
                nlines+=1
                if not line.startswith("data:"): 
                    print("NONDATA:", repr(line[:60]))
                    continue
                d=line[5:].strip()
                if d=="[DONE]": break
                try:
                    j=json.loads(d)
                    c=j["choices"][0]
                    delta=c.get("delta",{})
                    tok=delta.get("content") or ""
                    if tok:
                        if first is None: first=time.monotonic()-t0
                        full+=tok
                except Exception as e:
                    print("parse err", e, repr(line[:60]))
    print(f"nlines={nlines} first={first and round(first*1000,1)}ms full='{full!r}'")

asyncio.run(main())
