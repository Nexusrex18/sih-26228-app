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




def inject(source, out, *, fraction=.5, brightness=0., contrast=1., noise=0.,
           jpeg_quality=None, seed=7):
    """Split an existing clean corpus, transform a seeded fraction of the incoming half.

    Reference and incoming have disjoint source images. This prevents the unchanged
    fraction from becoming a self-comparison. The caller declares the corpus clean.
    """
    import shutil

    from PIL import ImageOps

    from cva.loaders.drift import MAX_DRIFT_PIXELS, SUFFIXES
    from cva.loaders.safety import check_pixel_budget, safe_join

    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_dir() or source == out or source in out.parents:
        raise ValueError('Source must be a directory and output must be outside it')
    if (not np.isfinite([fraction, brightness, contrast, noise]).all() or
            not 0 <= fraction <= 1 or contrast < 0 or noise < 0 or
            (jpeg_quality is not None and (isinstance(jpeg_quality, bool) or
             not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 95))):
        raise ValueError('Invalid shift parameters')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Injector output must be empty to avoid mixing datasets')
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(paths) < 4:
        raise ValueError('Need at least four clean images for independent splits')
    for path in paths:
        safe_join(source, path.relative_to(source).as_posix())
        check_pixel_budget(path, MAX_DRIFT_PIXELS)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Clean source contains duplicate image content; deduplicate before splitting')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    middle = len(paths)//2
    split = {'reference': order[:middle], 'incoming': order[middle:]}
    selected = set(rng.choice(len(split['incoming']), round(fraction*len(split['incoming'])), replace=False).tolist())
    entries = []
    for group, indices in split.items():
        (out/group).mkdir(parents=True, exist_ok=True)
        for position, index in enumerate(indices):
            path = paths[index]
            changed = group == 'incoming' and position in selected
            suffix = ('.jpg' if jpeg_quality is not None else '.png') if changed else path.suffix.lower()
            dest = out/group/f'{position:05d}{suffix}'
            with Image.open(path) as original:
                original.load()
                image = ImageOps.exif_transpose(original).convert('RGB')
                exif = image.getexif().tobytes()
                before = np.asarray(image, dtype=float)/255.
            if changed:
                pixels = np.clip((before-.5)*contrast+.5+brightness+
                                 rng.normal(0, noise, before.shape), 0, 1)
                shifted = Image.fromarray(np.rint(pixels*255).astype('uint8'))
                options = {'quality': jpeg_quality, 'subsampling': 0} if jpeg_quality is not None else {}
                shifted.save(dest, exif=exif, **options)
            else:
                shutil.copyfile(path, dest)
            with Image.open(dest) as saved:
                after = np.asarray(ImageOps.exif_transpose(saved).convert('RGB'), dtype=float)/255.
            entries.append({'source': path.relative_to(source).as_posix(), 'source_sha256': hashes[index],
                            'output': dest.relative_to(out).as_posix(), 'transformed': changed,
                            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                            'brightness_before': float((before @ [.299,.587,.114]).mean()),
                            'brightness_after': float((after @ [.299,.587,.114]).mean())})
    manifest = {'seed': seed, 'source_n': len(paths), 'fraction': fraction,
                'transformed_n': len(selected), 'incoming_n': len(split['incoming']),
                'brightness': brightness, 'contrast': contrast, 'noise': noise,
                'jpeg_quality': jpeg_quality, 'files': entries,
                'assumptions': ['Source corpus is declared clean by caller.',
                                'Disjoint source split; brightness truth includes clipping and encoding.']}
    (out/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--source', help='Existing clean corpus; split before injecting into incoming')
    p.add_argument('--fraction', type=float, default=.5)
    p.add_argument('--brightness', type=float, default=.2)
    p.add_argument('--contrast', type=float, default=1.)
    p.add_argument('--noise', type=float, default=.01)
    p.add_argument('--jpeg-quality', type=int)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if a.source:
        inject(a.source, a.out, fraction=a.fraction, brightness=a.brightness,
               contrast=a.contrast, noise=a.noise, jpeg_quality=a.jpeg_quality, seed=a.seed)
    else:
        if a.fraction != .5 or a.contrast != 1. or a.jpeg_quality is not None:
            p.error('--fraction, --contrast and --jpeg-quality require --source')
        generate(a.out, seed=a.seed, brightness=a.brightness, noise=a.noise)
