"""AudioSource abstraction: FileReplay and Microphone.

Every downstream component sees AudioFrame(pcm16, sample_rate, timestamp).
"""
from __future__ import annotations
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import soundfile as sf


@dataclass
class AudioFrame:
    pcm16: bytes  # little-endian int16 mono 16kHz
    sample_rate: int  # always 16000 for downstream
    timestamp: float  # monotonic seconds since stream start
    seq: int
    is_last: bool = False


class AudioSource:
    def frames(self) -> Generator[AudioFrame, None, None]:
        raise NotImplementedError


class FileReplayAudioSource(AudioSource):
    """Replay WAV/FLAC/etc as if from live mic.

    - resamples to mono 16k PCM16
    - emits 20ms chunks according to monotonic clock when realtime_factor=1.0
    - realtime_factor=0 => unpaced throughput
    """

    def __init__(
        self,
        path: str | Path,
        chunk_ms: int = 20,
        realtime_factor: float = 1.0,
        sample_rate: int = 16000,
    ):
        self.path = Path(path)
        self.chunk_ms = chunk_ms
        self.realtime_factor = realtime_factor
        self.target_sr = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

    def _load_mono_16k(self) -> np.ndarray:
        if not self.path.exists():
            raise FileNotFoundError(f"audio file not found: {self.path}")
        try:
            data, sr = sf.read(str(self.path), always_2d=False, dtype="float32")
        except Exception as e:
            raise RuntimeError(f"failed to read audio {self.path}: {e}") from e
        if data.size == 0:
            raise ValueError(f"empty audio file: {self.path}")
        # to mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        # resample if needed (simple linear interpolation)
        if sr != self.target_sr:
            # linear resample — adequate for 16k benchmark; keep deps minimal
            duration = len(data) / sr
            target_len = int(duration * self.target_sr)
            if target_len <= 0:
                raise ValueError(f"invalid target length {target_len} for {self.path}")
            old_idx = np.linspace(0, 1, len(data), dtype=np.float64)
            new_idx = np.linspace(0, 1, target_len, dtype=np.float64)
            data = np.interp(new_idx, old_idx, data).astype(np.float32)
        # soundfile returns float32 in [-1,1] when dtype=float32
        # clip and convert to int16
        data = np.clip(data, -1.0, 1.0)
        pcm = (data * 32767).astype(np.int16)
        return pcm

    def frames(self) -> Generator[AudioFrame, None, None]:
        pcm = self._load_mono_16k()
        total_samples = len(pcm)
        if total_samples == 0:
            return
        n_chunks = (total_samples + self.chunk_samples - 1) // self.chunk_samples
        start_wall = time.monotonic()
        for i in range(n_chunks):
            s = i * self.chunk_samples
            e = min(s + self.chunk_samples, total_samples)
            chunk = pcm[s:e]
            is_last = (i == n_chunks - 1)
            timestamp = i * self.chunk_ms / 1000.0
            # realtime pacing: wall time tracks timestamp / realtime_factor
            if self.realtime_factor > 0:
                target_wall = start_wall + timestamp / self.realtime_factor
                now = time.monotonic()
                sleep = target_wall - now
                if sleep > 0:
                    time.sleep(sleep)
            yield AudioFrame(
                pcm16=chunk.tobytes(),
                sample_rate=self.target_sr,
                timestamp=timestamp,
                seq=i,
                is_last=is_last,
            )

    def duration_s(self) -> float:
        pcm = self._load_mono_16k()
        return len(pcm) / self.target_sr


class MicrophoneAudioSource(AudioSource):
    """Live microphone via sounddevice."""

    def __init__(self, chunk_ms: int = 20, sample_rate: int = 16000, device: Optional[int | str] = None):
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.device = device
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._q: queue.Queue[AudioFrame | None] = queue.Queue()
        self._running = False
        self._seq = 0
        self._start_time: float = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}")
        # indata is float32 [-1,1], shape (frames, channels)
        try:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        except Exception:
            mono = indata
        pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
        ts = time.monotonic() - self._start_time
        frame = AudioFrame(
            pcm16=pcm.tobytes(),
            sample_rate=self.sample_rate,
            timestamp=ts,
            seq=self._seq,
            is_last=False,
        )
        self._seq += 1
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def frames(self) -> Generator[AudioFrame, None, None]:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError("sounddevice not installed; pip install sounddevice") from e

        self._running = True
        self._seq = 0
        self._start_time = time.monotonic()
        self._q = queue.Queue(maxsize=100)
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,
                device=self.device,
                callback=self._callback,
            )
        except Exception as e:
            raise RuntimeError(f"failed to open microphone (device={self.device!r}): {e}") from e
        with stream:
            print("[mic] Listening... speak naturally (Ctrl+C to stop)")
            while self._running:
                try:
                    frame = self._q.get(timeout=0.1)
                    if frame is None:
                        break
                    yield frame
                except queue.Empty:
                    continue

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except:
            pass


def load_audio_manifest_entry(entry: dict, chunk_ms: int = 20, realtime_factor: float = 1.0) -> FileReplayAudioSource:
    """Helper to create source from manifest entry."""
    return FileReplayAudioSource(entry["audio"], chunk_ms=chunk_ms, realtime_factor=realtime_factor)
