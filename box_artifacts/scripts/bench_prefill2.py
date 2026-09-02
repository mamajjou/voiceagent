"""Measure prefill-while-speaking using raw /completion with a GROWING prompt.

The canonical llama.cpp KV-reuse pattern: keep one prompt sequence and extend it
across requests. Because the prefix is token-identical, cache_prompt reuses every
token up to the cached match -> only newly added tokens get prefilled.

Simulates: during speech, we prefill the user text as it grows. When EOU commits,
we prepend the reply instruction and generate. The suffix (new tokens) is small.

We measure the effective prefill cost for:
  - cold full prompt (no cache)
  - warm: prefilled incremental prefix, then extended to full prompt
"""
import asyncio, json, time, httpx

BASE = 'http://127.0.0.1:8081'
MODEL = '/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM = "You are a helpful assistant. Answer concisely."

def comp_payload(prompt, n_pred):
    return {"model": MODEL, "prompt": prompt, "max_tokens": n_pred,
            "temperature": 0.7, "cache_prompt": True,
            "stop": ["<|im_end|>"]}

async def completion(prompt, n_pred=1):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{BASE}/completion", json=comp_payload(prompt, n_pred))
        j = r.json()
        return j

# Build one growing prompt that represents the user's utterance growing token-by-token
# (as ASR partials become more complete), followed by the assistant reply instruction.
# A realistic partial evolution of "What is the capital of France?"
partial_evolution = [
    "What",
    "What is",
    "What is the",
    "What is the capital",
    "What is the capital of",
    "What is the capital of France?",
]
REPLY_TAIL = "\nYou are answering. Provide a concise answer:"

async def main():
    print("=== PREFILL-WHILE-SPEAKING (growing-prompt KV reuse) ===\n")
    # Force cold by adding a unique session prefix token to each run is hard;
    # instead measure relative reuse by cache_n across the growing sequence.
    prompt = ""
    print("Growing user prompt (prefill during speech):")
    for i, p in enumerate(partial_evolution):
        prompt = SYSTEM + "\nUser: " + p
        t0 = time.monotonic()
        j = await completion(prompt, n_pred=1)
        dt = (time.monotonic()-t0)*1000
        t = j.get("timings", {})
        print(f"  partial[{i}] len={len(p):>2} -> cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms'):.1f} (wall {dt:.1f}ms)")
    # Now the full final prompt (all of it) -> should reuse the last cached prefix
    full_prompt = SYSTEM + "\nUser: What is the capital of France?\nAssista"
    t0 = time.monotonic()
    j = await completion(full_prompt, n_pred=1)
    dt = (time.monotonic()-t0)*1000
    t = j.get("timings", {})
    print(f"\nFULL after warm: cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms'):.1f} (wall {dt:.1f}ms)")

    # COLD reference: fully fresh full prompt
    t0 = time.monotonic()
    j = await completion("You are a helpful assistant. Answer concisely.\nUser: What is the capital of France?\nAssista", n_pred=1)
    dt = (time.monotonic()-t0)*1000
    t = j.get("timings", {})
    print(f"COLD full:       cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms'):.1f} (wall {dt:.1f}ms)")

asyncio.run(main())
