import asyncio, json, httpx, time

QWEN_URL = 'http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL = '/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM_PROMPT = ("You are participating in a live spoken conversation. Respond naturally and directly. Prefer concise conversational answers.")

async def main():
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"What is the capital of France?"}]
    payload = {
        "model": QWEN_MODEL, "messages": messages,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "presence_penalty": 1.5, "max_tokens": 128, "stream": True,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0=time.monotonic(); got=0; full=""; first=None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", QWEN_URL, json=payload) as resp:
                print("status", resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"): continue
                    d=line[5:].strip()
                    if d=="[DONE]": break
                    try:
                        j=json.loads(d)
                        delta=j["choices"][0]["delta"].get("content") or ""
                        if delta:
                            if first is None: first=time.monotonic()-t0
                            full+=delta; got+=1
                    except Exception as ex:
                        print("parse err", ex, repr(line[:100]))
    except Exception as e:
        print("HTTPX ERR:", type(e).__name__, e)
    print(f"got={got} first_tok={first and round(first*1000,1)}ms full='{full[:100]}'")

asyncio.run(main())
