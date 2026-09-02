"""Prepare JSONL manifest from AMI / VoxPopuli subsets."""
import json
import argparse
from pathlib import Path
import soundfile as sf

def prepare_ami_subset(out_manifest: Path, audio_dir: Path, n_recordings: int = 3):
    """Download 2-5 AMI recordings via datasets and slice into turns."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed")
        return
    # Use diarizers-community/ami
    print("[prepare] loading AMI via datasets...")
    ds = load_dataset("diarizers-community/ami", "ihm", streaming=True, trust_remote_code=True)
    # ds is dict with splits
    # We'll take first n_recordings from test or train
    split = "test" if "test" in ds else list(ds.keys())[0]
    it = iter(ds[split])
    entries = []
    for i in range(n_recordings):
        try:
            ex = next(it)
        except StopIteration:
            break
        # ex has audio, timestamps_start, timestamps_end, speakers, transcript? field varies
        audio = ex["audio"]  # dict with array, path, sampling_rate
        speakers = ex.get("speakers") or ex.get("speaker") or []
        starts = ex.get("timestamps_start") or ex.get("begin_time") or []
        ends = ex.get("timestamps_end") or ex.get("end_time") or []
        # text field might be "text" or "transcript"
        text = ex.get("text") or ex.get("transcript") or ex.get("utterance") or ""
        # For AMI, we need to slice per speaker turn
        # Simplified: one entry per recording, but we will slice by speaker turns if available
        rec_id = ex.get("meeting_id") or ex.get("id") or f"ami_{i:03d}"
        # Save audio file
        arr = audio["array"] if isinstance(audio, dict) else audio
        sr = audio["sampling_rate"] if isinstance(audio, dict) else 16000
        out_wav = audio_dir / f"{rec_id}.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), arr, sr)
        if starts and ends and isinstance(speakers, list):
            for j, (s, e, spk) in enumerate(zip(starts, ends, speakers)):
                # need text alignment; fallback to full text
                entries.append({
                    "id": f"ami_{rec_id}_turn_{j:03d}",
                    "audio": str(out_wav),
                    "start_s": float(s),
                    "end_s": float(e),
                    "speaker": str(spk),
                    "language": "en-US",
                    "reference_text": text if isinstance(text, str) else "",
                    "source": "AMI"
                })
        else:
            entries.append({
                "id": f"ami_{rec_id}",
                "audio": str(out_wav),
                "start_s": 0.0,
                "end_s": len(arr)/sr,
                "speaker": "A",
                "language": "en-US",
                "reference_text": text if isinstance(text, str) else "",
                "source": "AMI"
            })
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {len(entries)} entries to {out_manifest}")

def prepare_voxpopuli_subset(out_manifest: Path, audio_dir: Path, n: int = 100, lang: str = "de"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed")
        return
    print(f"[prepare] loading VoxPopuli {lang}...")
    ds = load_dataset("facebook/voxpopuli", lang, split="test", streaming=True, trust_remote_code=True)
    it = iter(ds)
    entries = []
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        try:
            ex = next(it)
        except StopIteration:
            break
        audio = ex["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]
        text = ex.get("sentence") or ex.get("text") or ex.get("raw_text") or ""
        spk = ex.get("speaker_id") or ex.get("speaker") or "unknown"
        out_wav = audio_dir / f"voxpopuli_{lang}_{i:05d}.wav"
        sf.write(str(out_wav), arr, sr)
        entries.append({
            "id": f"voxpopuli_{lang}_{i:05d}",
            "audio": str(out_wav),
            "start_s": 0.0,
            "end_s": len(arr)/sr,
            "speaker": str(spk),
            "language": "de-DE" if lang=="de" else "en-US",
            "reference_text": text,
            "source": "VoxPopuli"
        })
    # append to manifest
    existing = []
    if out_manifest.exists():
        existing = [json.loads(l) for l in open(out_manifest) if l.strip()]
    with open(out_manifest, "w") as f:
        for e in existing + entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[prepare] added {len(entries)} VoxPopuli entries")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/manifest.jsonl")
    ap.add_argument("--audio-dir", default="eval/audio")
    ap.add_argument("--ami", type=int, default=3, help="num AMI recordings")
    ap.add_argument("--voxpopuli-de", type=int, default=50)
    ap.add_argument("--voxpopuli-en", type=int, default=0)
    args = ap.parse_args()
    prepare_ami_subset(Path(args.manifest), Path(args.audio_dir), n_recordings=args.ami)
    if args.voxpopuli_de > 0:
        prepare_voxpopuli_subset(Path(args.manifest), Path(args.audio_dir), n=args.voxpopuli_de, lang="de")
