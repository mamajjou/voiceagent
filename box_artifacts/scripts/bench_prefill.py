"""Measure the prefill-while-speaking benefit.

Scenario A (baseline / cold): user finishes speaking -> send FULL user text to Qwen
    -> Qwen must prefill everything -> first token.

Scenario B (prefill-while-speaking): during speech, warm Qwen KV cache with the
    conversation prefix + the CURRENT stable partial user text. When EOU commits,
    only the small suffix is new -> much less prefill -> faster first token.

We simulate this by prefilling with a partial prefix, then extending with the
full text in a follow-up request. llama.cpp cache_prompt reuses the common prefix.
The saved time = (full prefill) - (suffix-only prefill).

We also measure the "wasted" work if the partial changes (prefix re-computed).
"""
import asyncio, json, time, httpx

QWEN_URL = 'http://127.0.0.1:8081/v1/chat/completions'
QWEN_MODEL = '/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM = "You are participating in a live spoken conversation. Respond naturally and directly."

def payload(messages, n_pred=1):
    return {"model": QWEN_MODEL, "messages": messages, "temperature": 0.7,
            "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
            "max_tokens": n_pred, "stream": False, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}

async def call(messages):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(QWEN_URL, json=payload(messages))
        j = r.json()
        cb = j["choices"][0]["message"]["content"]
        t = j.get("timings", {})
        return cb, t

async def main():
    full_texts = [
        "What is the capital of France?",
        "Could you explain why the sky is blue?",
        "Wie lautet die Hauptstadt von Deutschland?",
    ]
    # prefixes that are stable partials growing toward the full text
    prefixes = {
        "What is the capital of France?": ["What", "What is the capital"],
        "Could you explain why the sky is blue?": ["Could you", "Could you explain why the sky"],
        "Wie lautet die Hauptstadt von Deutschland?": ["Wie lautet", "Wie lautet die Hauptstadt von"],
    }
    print("=== PREFILL-WHILE-SPEAKING BENEFIT (cold vs warmed prefix) ===\n")
    for full in full_texts:
        # Cold: prefill full text fresh (zero-cache). Force by using a fresh unique system.
        msgs_full = [{"role":"system","content":SYSTEM},{"role":"user","content":full}]
        # Send once to populate nothing; measure TTFT as the generation time of a 1-token probe
        # TTFT for generation = prompt_ms (prefill) since first token comes after prefill.
        # Measure prompt_ms for full (cold) and for prefix-warmed then suffix.
        cb, t_full = await call(msgs_full)
        prompt_full = t_full.get("prompt_ms")
        print(f"{full!r}")
        print(f"  COLD full prefill: prompt_ms={prompt_full:.1f} (prompt_n={t_full.get('prompt_n')})")
        # Warmed: prefill with a prefix, then send full. cache reuses prefix.
        for pre in prefixes[full]:
            msgs_pre = [{"role":"system","content":SYSTEM},{"role":"user","content":pre}]
            cb, t_pre = await call(msgs_pre)
            # Now send the full text; should reuse the common prefix (system+pre)
            cb, t_full2 = await call(msgs_full)
            cached = t_full2.get("cache_n")
            now_prompt = t_full2.get("prompt_ms")
            print(f"  WARM prefix='{pre!r}': prefill_ms={t_pre.get('prompt_ms'):.1f} | full_cached={cached} full_prompt_ms={now_prompt:.1f}")
        print()

asyncio.run(main())
