import csv, json

def load(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

all_rows = []
for rc in [1,3,6]:
    rows = load(f"/tmp/rc_{rc}.csv")
    for r in rows:
        all_rows.append(r)
        r["rc"] = rc

# sort by rc then eou
all_rows.sort(key=lambda r: (int(r["rc"]), int(r["eou_ms"])))

# round for readability
def fnum(x):
    if x in ("None","",None): return None
    try: return round(float(x),1)
    except: return x
for r in all_rows:
    for k in ["asr_final_median_ms","asr_final_p90_ms","ttft_median_ms","ttft_p90_ms","wer_mean"]:
        r[k]=fnum(r.get(k))

with open("/tmp/sweep_15.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
with open("/tmp/sweep_15.json","w") as f:
    json.dump(all_rows, f, indent=2)

print(f"{'rc':>3} {'lkh':>6} {'eou':>5} {'ASRf_med':>9} {'TTFT_med':>9} {'WER':>5}")
for r in all_rows:
    print(f"{r['rc']:>3} {r['rc_ms']:>5}ms {r['eou_ms']:>5} {r['asr_final_median_ms']:>9} {r['ttft_median_ms']:>9} {r.get('wer_mean') or 0.0:>5}")
