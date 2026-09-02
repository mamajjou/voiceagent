"""Prepare JSONL manifest from AMI / VoxPopuli subsets."""
import json
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np

TARGET_SR = 16000

def resample_to_16k(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr == TARGET_SR:
        return arr
    duration = len(arr) / sr
    target_len = int(duration * TARGET_SR)
    old_idx = np.linspace(0, 1, len(arr))
    new_idx = np.linspace(0, 1, target_len)
    return np.interp(new_idx, old_idx, arr).astype(np.float32)

def prepare_ami_subset(out_manifest: Path, audio_dir: Path, n_recordings: int = 3):
    """Download 2-5 AMI recordings via datasets and slice into turns."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed")
        return
    print("[prepare] loading AMI via datasets...")
    ds = None
    last_err = None
    for repo in ["edinburghcstr/ami", "diarizers-community/ami"]:
        for cfg in ["ihm", None]:
            try:
                kwargs = {"streaming": True}
                if cfg:
                    ds = load_dataset(repo, cfg, split="test", **kwargs)
                else:
                    ds = load_dataset(repo, split="test", **kwargs)
                print(f"[prepare] loaded {repo} cfg={cfg}")
                break
            except Exception as e:
                last_err = e
                continue
        if ds is not None:
            break
    if ds is None:
        print(f"[prepare] AMI load failed: {last_err}")
        # create synthetic fallback so manifest exists
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for i in range(min(n_recordings, 2)):
            wav = audio_dir / f"ami_synth_{i:03d}.wav"
            arr = (0.1 * np.sin(2*np.pi*440*np.linspace(0, 3, 3*TARGET_SR))).astype(np.float32)
            sf.write(str(wav), arr, TARGET_SR)
            entries.append({
                "id": f"ami_synth_{i:03d}",
                "audio": str(wav),
                "start_s": 0.0,
                "end_s": 3.0,
                "speaker": "A",
                "language": "en-US",
                "reference_text": "hello this is a synthetic test utterance for ami fallback",
                "source": "AMI synthetic"
            })
        with open(out_manifest, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[prepare] wrote synthetic {len(entries)} AMI entries")
        return

    it = iter(ds)
    entries = []
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_recordings):
        try:
            ex = next(it)
        except StopIteration:
            break
        # audio field variations
        audio = ex.get("audio") or ex.get("waveform") or ex.get("speech")
        if audio is None:
            print(f"[prepare] no audio in record {i}, keys {list(ex.keys())}")
            continue
        if isinstance(audio, dict):
            arr = audio.get("array") or audio.get("data") or audio.get("waveform")
            sr = audio.get("sampling_rate") or audio.get("sample_rate") or 16000
        else:
            arr = audio
            sr = 16000
        if arr is None:
            continue
        arr = resample_to_16k(arr, sr)
        sr = TARGET_SR
        rec_id = ex.get("meeting_id") or ex.get("id") or ex.get("recording_id") or f"ami_{i:03d}"
        rec_id = str(rec_id).replace("/", "_")
        # full wav for debugging
        full_wav = audio_dir / f"{rec_id}.wav"
        sf.write(str(full_wav), arr, sr)
        speakers = ex.get("speakers") or ex.get("speaker") or ex.get("speakers_labels") or []
        starts = ex.get("timestamps_start") or ex.get("begin_time") or ex.get("start_times") or ex.get("segments_start") or []
        ends = ex.get("timestamps_end") or ex.get("end_time") or ex.get("end_times") or ex.get("segments_end") or []
        text = ex.get("text") or ex.get("transcript") or ex.get("utterance") or ex.get("sentence") or ""
        if isinstance(text, list):
            text = " ".join(text)
        # AMI often has speaker turns as parallel arrays
        if starts and ends and isinstance(speakers, (list, np.ndarray)) and len(starts)==len(ends):
            # slice each turn to separate wav for clean manifest
            for j, (s, e, spk) in enumerate(zip(starts, ends, speakers)):
                try:
                    s_f = float(s); e_f = float(e)
                except: continue
                if e_f <= s_f: continue
                s_idx = max(0, int(s_f * sr))
                e_idx = min(len(arr), int(e_f * sr))
                seg = arr[s_idx:e_idx]
                if len(seg) < sr*0.5:  # skip very short <0.5s
                    continue
                seg_wav = audio_dir / f"{rec_id}_turn_{j:03d}.wav"
                sf.write(str(seg_wav), seg, sr)
                # try to get per-turn text if available
                turn_text = text
                if isinstance(ex.get("transcripts"), list) and j < len(ex["transcripts"]):
                    turn_text = ex["transcripts"][j]
                elif isinstance(ex.get("utterances"), list) and j < len(ex["utterances"]):
                    turn_text = ex["utterances"][j]
                entries.append({
                    "id": f"ami_{rec_id}_turn_{j:03d}",
                    "audio": str(seg_wav),
                    "start_s": 0.0,
                    "end_s": float(len(seg)/sr),
                    "orig_start_s": float(s_f),
                    "orig_end_s": float(e_f),
                    "speaker": str(spk),
                    "language": "en-US",
                    "reference_text": turn_text if isinstance(turn_text, str) else str(turn_text),
                    "source": "AMI"
                })
                if len(entries) >= n_recordings*5:  # cap per-recording turns
                    break
        else:
            # single entry per recording
            txt = text if isinstance(text, str) else str(text)
            entries.append({
                "id": f"ami_{rec_id}",
                "audio": str(full_wav),
                "start_s": 0.0,
                "end_s": float(len(arr)/sr),
                "speaker": str(speakers[0]) if isinstance(speakers, list) and speakers else "A",
                "language": "en-US",
                "reference_text": txt,
                "source": "AMI"
            })
        print(f"[prepare] AMI {rec_id}: total entries {len(entries)}")
        if len(entries) >= n_recordings*5:
            break
    # if we got no sliced entries, keep full
    if not entries:
        print("[prepare] no AMI entries extracted, using fallback synthetic")
        for i in range(min(n_recordings, 2)):
            wav = audio_dir / f"ami_synth_{i:03d}.wav"
            arr = (0.1 * np.sin(2*np.pi*440*np.linspace(0, 3, 3*TARGET_SR))).astype(np.float32)
            sf.write(str(wav), arr, TARGET_SR)
            entries.append({"id": f"ami_synth_{i:03d}", "audio": str(wav), "start_s": 0.0, "end_s": 3.0, "speaker": "A", "language": "en-US", "reference_text": "hello this is a synthetic test utterance", "source": "AMI synthetic"})
    # write manifest (overwrite for AMI stage)
    with open(out_manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {len(entries)} AMI entries to {out_manifest}")

def prepare_voxpopuli_subset(out_manifest: Path, audio_dir: Path, n: int = 100, lang: str = "de"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed")
        return
    print(f"[prepare] loading VoxPopuli {lang}...")
    try:
        ds = load_dataset("facebook/voxpopuli", lang, split="validation", streaming=True)
    except Exception as e:
        print(f"[prepare] voxpopuli validation load failed {e}, trying test")
        try:
            ds = load_dataset("facebook/voxpopuli", lang, split="test", streaming=True)
        except Exception as e2:
            print(f"[prepare] voxpopuli load failed {e2}")
            # synthetic fallback
            audio_dir.mkdir(parents=True, exist_ok=True)
            entries = []
            for i in range(min(n, 5)):
                wav = audio_dir / f"voxpopuli_{lang}_synth_{i:05d}.wav"
                arr = (0.1 * np.sin(2*np.pi*440*np.linspace(0, 2, 2*TARGET_SR))).astype(np.float32)
                sf.write(str(wav), arr, TARGET_SR)
                entries.append({"id": f"voxpopuli_{lang}_synth_{i:05d}", "audio": str(wav), "start_s": 0.0, "end_s": 2.0, "speaker": "synth", "language": "de-DE" if lang=="de" else "en-US", "reference_text": "Hallo dies ist ein synthetischer Testsatz" if lang=="de" else "hello this is a synthetic test", "source": "VoxPopuli synthetic"})
            existing = []
            if out_manifest.exists():
                existing = [json.loads(l) for l in open(out_manifest) if l.strip()]
            with open(out_manifest, "w") as f:
                for e in existing + entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"[prepare] added {len(entries)} synthetic VoxPopuli entries")
            return
    it = iter(ds)
    entries = []
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        try:
            ex = next(it)
        except StopIteration:
            break
        audio = ex.get("audio")
        if audio is None:
            continue
        arr = audio.get("array") if isinstance(audio, dict) else audio
        sr = audio.get("sampling_rate") if isinstance(audio, dict) else 16000
        if arr is None:
            continue
        arr = resample_to_16k(arr, sr)
        text = ex.get("normalized_text") or ex.get("raw_text") or ex.get("sentence") or ex.get("text") or ""
        spk = ex.get("speaker_id") or ex.get("speaker") or ex.get("speaker_name") or "unknown"
        # VoxPopuli normalized_text is already lowercased, use raw_text if available for cased
        if not text:
            text = ex.get("raw_text") or ""
        out_wav = audio_dir / f"voxpopuli_{lang}_{i:05d}.wav"
        sf.write(str(out_wav), arr, TARGET_SR)
        entries.append({
            "id": f"voxpopuli_{lang}_{i:05d}",
            "audio": str(out_wav),
            "start_s": 0.0,
            "end_s": float(len(arr)/TARGET_SR),
            "speaker": str(spk),
            "language": "de-DE" if lang=="de" else "en-US",
            "reference_text": text,
            "source": "VoxPopuli"
        })
        if (i+1) % 10 == 0:
            print(f"[prepare] VoxPopuli {lang} {i+1}/{n}")
    existing = []
    if out_manifest.exists():
        existing = [json.loads(l) for l in open(out_manifest) if l.strip()]
    with open(out_manifest, "w") as f:
        for e in existing + entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[prepare] added {len(entries)} VoxPopuli {lang} entries (total {len(existing)+len(entries)})")

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
    if args.voxpopuli_en > 0:
        prepare_voxpopuli_subset(Path(args.manifest), Path(args.audio_dir), n=args.voxpopuli_en, lang="en")
