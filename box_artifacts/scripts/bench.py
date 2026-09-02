import time, subprocess, json
model='/workspace/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf'
audio='/tmp/sample.wav'
start=time.time()
result=subprocess.run(['/workspace/nemo-prebuilt/nemo-speech-0.1.0-linux-x86_64-cuda/bin/nemo-speech','transcribe',audio,'--model',model,'--device','cuda:0','--format','json'], env={'LD_LIBRARY_PATH':'/workspace/nemo-prebuilt/nemo-speech-0.1.0-linux-x86_64-cuda/lib'}, capture_output=True, text=True)
elapsed=time.time()-start
print(f'elapsed {elapsed:.3f}s')
print('STDOUT', result.stdout[:1000])
print('STDERR', result.stderr[:1000])
print('returncode', result.returncode)
