#!/usr/bin/env python3
"""Download small eval subsets without huge downloads."""
import argparse
from eval.prepare_manifest import prepare_ami_subset, prepare_voxpopuli_subset
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami", type=int, default=2)
    ap.add_argument("--voxpopuli-de", type=int, default=50)
    args = ap.parse_args()
    prepare_ami_subset(Path("eval/manifest.jsonl"), Path("eval/audio"), n_recordings=args.ami)
    if args.voxpopuli_de:
        prepare_voxpopuli_subset(Path("eval/manifest.jsonl"), Path("eval/audio"), n=args.voxpopuli_de, lang="de")
