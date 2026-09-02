import imageio_ffmpeg, subprocess, os, soundfile as sf
from gtts import gTTS

exe = imageio_ffmpeg.get_ffmpeg_exe()

def make(txt, lang, path):
    mp3 = path.replace('.wav', '.mp3')
    t = gTTS(txt, lang=lang)
    t.save(mp3)
    subprocess.run([exe, '-y', '-i', mp3, '-ar', '16000', '-ac', '1', path],
                   capture_output=True, check=True)
    d, s = sf.read(path)
    dur = len(d) / s
    print("OK", path, lang, round(dur, 2), flush=True)

de_sents = [
    ("Guten Morgen, wie geht es Ihnen heute?", "/tmp/de2.wav"),
    ("Ich bin gestern nach Berlin gefahren.", "/tmp/de3.wav"),
    ("Wie lautet die Hauptstadt von Deutschland?", "/tmp/de4.wav"),
    ("Koennen Sie erklaeren, warum der Himmel blau ist?", "/tmp/de5.wav"),
]
en_sents = [
    ("She had your dark suit and greasy washwater all year.", "/tmp/en1.wav"),
    ("Could you explain why the sky is blue?", "/tmp/en2.wav"),
    ("What is the capital of France?", "/tmp/en3.wav"),
    ("How does a transformer attention mechanism work?", "/tmp/en4.wav"),
]
for txt, path in de_sents:
    make(txt, 'de', path)
for txt, path in en_sents:
    make(txt, 'en', path)
print("DONE")
