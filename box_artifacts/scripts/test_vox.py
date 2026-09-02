from datasets import load_dataset
import time
print('loading de validation streaming...', flush=True)
ds = load_dataset('facebook/voxpopuli', 'de', split='validation', streaming=True, trust_remote_code=False)
print('iterating...', flush=True)
it = iter(ds)
for i in range(2):
    ex = next(it)
    print('ex', i, 'keys', list(ex.keys()), flush=True)
    a=ex.get('audio')
    print('audio type', type(a), flush=True)
    if isinstance(a, dict):
        print('audio keys', list(a.keys()), flush=True)
        arr=a.get('array')
        sr=a.get('sampling_rate')
        print('sr', sr, 'len', len(arr) if arr is not None else 0, flush=True)
    print('sentence', repr(ex.get('sentence','')[:120]), flush=True)
print('done', flush=True)
