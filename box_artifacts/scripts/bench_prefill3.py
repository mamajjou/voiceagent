"""Definitive prefill-while-speaking benefit.

Simulate the real flow on ONE persistent prompt sequence (byte-identical
system + growing user text). Use /completion cache_prompt to warm KV while the
user 'speaks'. Compare end-to-end first-token latency:

  BEFORE (baseline, cold): at EOU, send FULL prompt -> prefill everything -> first token.
  AFTER (prefill-while-speaking): warm incrementally during speech, at EOU send full
        -> only the new suffix needs prefill -> faster first token.

For a realistic utterance (~5 partials), and we append a fixed reply instruction.
"""
import asyncio, json, time, httpx

BASE = 'http://127.0.0.1:8081'
MODEL = '/workspace/models/Qwen3.8-27B-UD-Q4_K_M.gguf'
SYSTEM = "You are a helpful assistant. Answer concisely."
REPLY = "\nAssistant:"

def comp_payload(prompt, n_pred):
    return {"model": MODEL, "prompt": prompt, "max_tokens": n_pred,
            "temperature": 0.7, "cache_prompt": True}

async def completion(prompt, n_pred):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{BASE}/completion", json=comp_payload(prompt, n_pred))
        return r.json()

# realistic partial growth for a ~30-token utterance
def evolution(text):
    words = text.split()
    return [" ".join(words[:i+1]) for i in range(len(words))]

async def main():
    texts = [
        "What is the capital of France?",
        "Could you explain why the sky is blue?",
        "Wie lautet die Hauptstadt von Deutschland?",
        "How does a transformer attention mechanism work?",
    ]
    print("=== PREFILL-WHILE-SPEAKING: first-token latency after EOU ===\n")
    print(f"{'utterance':<42} {'COLD wall':>10} {'PREFILL wall':>12} {'saved':>8}")
    for text in texts:
        evo = evolution(text)
        # ---- BASELINE (cold): at EOU send full prompt, measure time to token ----
        t0 = time.monotonic()
        j = await completion(SYSTEM + "\nUser: " + text + REPLY, n_pred=1)
        cold_ms = (time.monotonic()-t0)*1000

        # ---- PREFILL-WHILE-SPEAKING: warm each partial, then commit ----
        # warm phase (simulates during speech)
        for p in evo:
            await completion(SYSTEM + "\nUser: " + p + REPLY, n_pred=1)
        # commit phase (at EOU): send full, only new suffix prefilled
        t0 = time.monotonic()
        j = await completion(SYSTEM + "\nUser: " + text + REPLY, n_pred=1)
        prefill_ms = (time.monotonic()-t0)*1000
        t = j.get("timings", {})
        print(f"{text[:42]:<42} {cold_ms:>10.1f} {prefill_ms:>12.1f} {cold_ms-prefill_ms:>8.1f}"
              + f"  (cache_n={t.get('cache_n')})")

asyncio.run(main())
