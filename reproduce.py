#!/usr/bin/env python3
"""Rebuild the paper's simulation figures and tables from supplied saved draws."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'output/cache'))


def verify(studies):
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if name.split('/')[1] in studies:
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Packaged input changed: {name}')
    for study in studies:
        cfg = json.loads((ROOT / 'configs' / f'{study}.json').read_text())
        for name, expected in cfg['engine_sources'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen scientific source changed: {name}')
    print('Packaged data and scientific-source hashes verified.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=('all', 'lasso', 'neural'), default='all')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    studies = ('lasso', 'neural') if args.only == 'all' else (args.only,)
    verify(studies)
    if args.verify_only:
        return
    for study in studies:
        subprocess.run([sys.executable, str(ROOT / 'analysis' / f'{study}.py')],
                       cwd=ROOT, check=True)
    print('Generated output/figures/*.pdf and output/{lasso,neural}/tables/*.tex.')


if __name__ == '__main__':
    main()
