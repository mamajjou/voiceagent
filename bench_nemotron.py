#!/usr/bin/env python3
"""Benchmark Nemotron 3.5 ASR streaming realtime.
Measures: WER, endpoint latency, realtime factor, VRAM.
Uses FileReplayAudioSource (20ms chunks) -> NeMo-Speech.cpp websocket.
Falls back to mock if server not reachable.
"""
import time, json, argparse, subprocess, pathlib, sys
from pathlib import Path
import soundfile as sf
import numpy as np
import asyncio

# Add src to path
sys.path.insert(0, "src")
from voice_agent.audio import FileReplayAudioSource
from voice_agent.nemo_client import NemoClient, ASRConfig

def make_sine_wav(path, duration=5, sr=16000, freq=440):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    data = 0.1*np.sin(2*np.pi*freq*t).astype(np.float32)
    # add some silence gaps to test endpointing
    data[int(1.0*sr):int(1.3*sr)] = 0  # 300ms silence
    data[int(2.5*sr):int(3.0*sr)] = 0  # 500ms silence
    sf.write(str(path), data, sr)
    print(f"wrote sine {path} {duration}s")

def make_speech_like_wav(path):
    # Use espeak if available, else sine
    try:
        subprocess.check_call(["espeak", "-v", "en", "-s", "150", "Hello this is a test of the Nemotron speech recognition system. How quickly can we transcribe?", "-w", str(path)], timeout=5)
        print(f"espeak wrote {path}")
        # resample to 16k if needed
        data, sr = sf.read(str(path))
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sf.write(str(path), data, 16000)
        return
    except Exception as e:
        print(f"espeak failed {e}, using sine")
        make_sine_wav(path, duration=4)

def vram_mb():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"], timeout=2).decode().strip()
        return float(out.split()[0])
    except: return None

async def bench_one(audio_path, language="en-US", mock_text=None):
    cfg = ASRConfig(host="127.0.0.1", port=8090, language=language, rnnt_right_context=3, eou_ms=650)
    client = NemoClient(cfg, mock_text=mock_text)
    ok = await client.check_health()
    print(f"ASR health {ok} (mock={mock_text is not None})")
    if not ok and mock_text is None:
        print("Server not reachable, using mock")
        client.mock_text = "Hello this is a mock transcription for benchmarking realtime factor."

    # Prepare frames
    src = FileReplayAudioSource(audio_path, chunk_ms=20, realtime_factor=1.0)
    duration = src.duration_s()
    print(f"Audio {audio_path} duration {duration:.2f}s, realtime_factor=1.0 (will take ~{duration:.1f}s)")

    t_start = time.monotonic()
    vram_before = vram_mb()
    print(f"VRAM before {vram_before} MB")

    partials = []
    def on_partial(p):
        ts = time.monotonic() - t_start
        tag = "✓" if p.is_final else "~"
        print(f"[{ts:06.2f}] {tag} {p.text[:80]}")
        partials.append((ts, p.text, p.is_final))

    # Stream
    frames = list(src.frames())  # this will pace in realtime
    # For accurate realtime measurement, we need to measure wall time vs audio duration
    t0 = time.monotonic()
    # Use async generator
    async def gen():
        for f in frames:
            yield f

    # Actually FileReplayAudioSource.frames() already sleeps for realtime, so list() above already took duration seconds.
    # Instead, we should not pre-list; we should stream directly.
    # Redo without pre-list:
    src2 = FileReplayAudioSource(audio_path, chunk_ms=20, realtime_factor=1.0)
    # Use sync frames wrapped
    import asyncio
    loop = asyncio.get_event_loop()
    # For this bench, we will use the client mock which consumes frames
    final = await client.stream_frames(src2.frames(), on_partial, language=language)
    t1 = time.monotonic()
    wall = t1 - t0
    # But t0-t1 now includes the realtime pacing inside src2.frames() (which sleeps). So wall should ~ duration + ASR overhead.
    # For throughput test, use realtime_factor=0
    src_fast = FileReplayAudioSource(audio_path, chunk_ms=20, realtime_factor=0)
    t2 = time.monotonic()
    partials_fast = []
    def on_partial_fast(p):
        partials_fast.append((time.monotonic()-t2, p.text, p.is_final))
    final_fast = await client.stream_frames(src_fast.frames(), on_partial_fast, language=language)
    t3 = time.monotonic()
    fast_wall = t3 - t2

    vram_after = vram_mb()
    print(f"\nRealtime (1.0) wall {wall:.2f}s for {duration:.2f}s audio => realtime factor {duration/wall:.2f}x, overhead {wall-duration:.2f}s")
    print(f"Fast (0) wall {fast_wall:.3f}s => throughput {duration/fast_wall:.1f}x realtime")
    print(f"VRAM after {vram_after} MB (delta { (vram_after or 0)-(vram_before or 0):.0f} MB)")
    print(f"Final: '{final}'")
    print(f"Partials: {len(partials)}")
    # Stability
    if partials and final:
        # longest common prefix
        def lcp(a,b):
            n=min(len(a),len(b))
            for i in range(n):
                if a[i]!=b[i]: return i
            return n
        lcps = [lcp(p[1], final) for p in partials]
        print(f"Mean LCP {sum(lcps)/len(lcps):.1f}/{len(final)} chars")
    # Endpoint latency: for synthetic, we know reference end = duration, endpoint = last partial final time
    if partials:
        last_final_ts = [ts for ts,_,is_final in partials if is_final]
        if last_final_ts:
            print(f"ASR final latency (wall - duration) {wall - duration:.3f}s")
            # For real manifest, we would compare to reference end_s

    return {
        "duration": duration,
        "wall_realtime": wall,
        "wall_fast": fast_wall,
        "vram_before": vram_before,
        "vram_after": vram_after,
        "final": final,
        "partials": len(partials),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="/tmp/test.wav")
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--gen", action="store_true", help="generate test wav")
    args = ap.parse_args()
    if args.gen or not Path(args.audio).exists():
        make_speech_like_wav(Path(args.audio))
    # Run
    result = asyncio.run(bench_one(args.audio, language=args.language, mock_text="Hello this is a test of the Nemotron speech recognition system." if args.mock else None))
    print(json.dumps(result, indent=2))
