import imageio_ffmpeg, subprocess, os, soundfile as sf
from gtts import gTTS

exe = imageio_ffmpeg.get_ffmpeg_exe()

def make(txt, lang, path):
    mp3 = path.replace('.wav', '.mp3')
    t = gTTS(txt, lang=lang)
    t.save(mp3)
    subprocess.run([exe, '-y', '-i', mp3, '-ar', '16000', '-ac', '1', path], check=True)
    d, s = sf.read(path)
    dur = len(d) / s
    print(f"{path}: '{txt}' -> {dur:.2f}s")
    return path

# Convert existing de_gtts.mp3
subprocess.run([exe, '-y', '-i', '/tmp/de_gtts.mp3', '-ar', '16000', '-ac', '1', '/tmp/de.wav'], check=True)
d, s = sf.read('/tmp/de.wav')
print('/tmp/de.wav', s, d.shape, d.dtype, d[:5])

de_sents = [
    ("Guten Morgen, wie geht es Ihnen heute?", "/tmp/de2.wav"),
    ("Ich bin gestern nach Berlin gefahren.", "/tmp/de3.wav"),
    ("Wie lautet die Hauptstadt von Deutschland?", "/tmp/de4.wav"),
    ("Koennen Sie erklaeren, warum der Himmel blau ist?", "/tmp/de5.wav"),
]
for txt, path in de_sents:
    make(txt, 'de', path)

en_sents = [
    ("She had your dark suit and greasy washwater all year.", "/tmp/en1.wav"),
    ("Could you explain why the sky is blue?", "/tmp/en2.wav"),
    ("What is the capital of France?", "/tmp/en3.wav"),
    ("How does a transformer attention mechanism work?", "/tmp/en4.wav"),
]
for txt, path in en_sents:
    make(txt, 'en', path)

print("DONE")
