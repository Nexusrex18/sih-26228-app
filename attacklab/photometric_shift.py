"""Seeded photometric drift fixture; separate from the production scanner."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate(out, n=80, seed=7, brightness=.2, noise=.01):
    if n < 2 or noise < 0 or not np.isfinite([brightness,noise]).all():
        raise ValueError('Invalid shift parameters')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Fixture output must be empty to avoid mixing datasets')
    rng = np.random.default_rng(seed)
    paths = []
    # Independent batches; do not present copies of the reference as independent draws.
    for group, shift in [('reference',0.),('incoming',brightness)]:
        directory = out/group
        directory.mkdir(parents=True,exist_ok=True)
        for i in range(n):
            pixels = rng.normal(.4,.08,(32,32,3)) + rng.normal(0,.025)
            if group == 'incoming':
                pixels += shift + rng.normal(0,noise,pixels.shape)
            path = directory/f'{i:04d}.png'
            Image.fromarray(np.uint8(np.clip(pixels,0,1)*255)).save(path)
            paths.append(path)
    manifest = {'seed':seed,'n':n,'brightness':brightness,'noise':noise,
                'files':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
    return manifest


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--brightness',type=float,default=.2)
    p.add_argument('--seed',type=int,default=7)
    a = p.parse_args()
    generate(a.out,seed=a.seed,brightness=a.brightness)
