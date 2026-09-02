import asyncio, json, time, httpx
BASE='http://127.0.0.1:8081'
MODEL='/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'

async def main():
    prompt="You are helpful.\nUser: What is the capital of France?\nAssistant:"
    t0=time.monotonic(); first=None; full=""; nlines=0
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{BASE}/completion", json={
            "model":MODEL,"prompt":prompt,"max_tokens":32,"cache_prompt":True,"temperature":0.7}) as resp:
            print("status", resp.status_code)
            async for line in resp.aiter_lines():
                nlines+=1
                if line.startswith("data:"):
                    d=line[5:].strip()
                    if d and d!="[DONE]":
                        try:
                            j=json.loads(d)
                            tok=j.get("content") or ""
                            if tok:
                                if first is None: first=time.monotonic()-t0
                                full+=tok
                        except Exception as e:
                            print("parse err", e, repr(line[:60]))
                elif line.strip():
                    print("non-data line:", repr(line[:60]))
    print(f"nlines={nlines} first={first and round(first*1000,1)}ms full='{full!r}'")

asyncio.run(main())
