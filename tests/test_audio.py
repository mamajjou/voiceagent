import numpy as np, tempfile, soundfile as sf
from pathlib import Path
from voice_agent.audio import FileReplayAudioSource

def test_file_replay():
    sr=16000
    t=np.linspace(0,1,sr, endpoint=False)
    data=0.1*np.sin(2*np.pi*440*t)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr)
        src=FileReplayAudioSource(f.name, realtime_factor=0)
        frames=list(src.frames())
        assert len(frames) > 40  # 1s /20ms =50
        assert frames[0].sample_rate==16000
        assert frames[-1].is_last
        total=len(b"".join(fr.pcm16 for fr in frames))
        assert total>= len(data)*2 -320
        print("ok", len(frames))
        Path(f.name).unlink()
if __name__=="__main__":
    test_file_replay()
