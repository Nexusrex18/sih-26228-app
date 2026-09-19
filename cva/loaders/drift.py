"""Local image-folder adapter and optional cache adapter for Module D.

The shared COCO/YOLO loader is not present yet. This adapter reads images recursively;
it does not claim annotation, contributor or semantic coverage.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace, median_filter

from cva.core.drift import DriftBatch

SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def load_image_batch(root, embedding_file=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Image directory does not exist: {root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not files:
        raise ValueError(f'No supported images in {root}')
    ids, values, optional, issues = [], {}, {}, []
    digest = hashlib.sha256()
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Image symlink escapes dataset: {path}')
        sid = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest.update(sid.encode()+b'\0'+hashlib.sha256(raw).digest())
        try:
            with Image.open(path) as source:
                source.load()  # corrupt images must fail, never silently shrink the sample
                tables = getattr(source, 'quantization', None)
                exif = source.getexif()
                iso = exif.get(34855)
                image = ImageOps.exif_transpose(source).convert('RGB')
                # Fixed thumbnail budget limits feature extraction, not input decoding.
                image.thumbnail((256, 256))
                rgb = np.asarray(image, dtype=float)/255.
                grey = rgb @ np.array([.299,.587,.114])
                row = {'brightness': float(grey.mean()), 'rms_contrast': float(grey.std()),
                       'sharpness': float(laplace(grey, mode='reflect').var()),
                       'noise_residual': float(np.median(np.abs(grey-median_filter(grey,size=3))))}
                for c, channel in enumerate(('red','green','blue')):
                    counts = np.histogram(rgb[:,:,c], bins=8, range=(0,1))[0]
                    for i, fraction in enumerate(counts/counts.sum()):
                        row[f'{channel}_hist_{i}'] = float(fraction)
                extra = {}
                if tables:
                    extra['jpeg_quantization_mean'] = float(np.mean([v for t in tables.values() for v in t]))
                if isinstance(iso, (int,float)) and np.isfinite(iso) and iso > 0:
                    extra['exif_iso'] = float(iso)
        except (OSError, ValueError) as exc:
            raise ValueError(f'Unreadable image {sid}: {exc}') from exc
        ids.append(sid)
        for key, value in row.items():
            values.setdefault(key, []).append(value)
        for key in ('jpeg_quantization_mean', 'exif_iso'):
            optional.setdefault(key, []).append(extra.get(key))
    for key, data in optional.items():
        if all(v is not None for v in data):
            values[key] = data
        else:
            issues.append(f'{key} not assessed: {sum(v is None for v in data)}/{len(data)} images lack metadata.')
    issues += ['Image features use EXIF-oriented RGB thumbnails up to 256 pixels; noise and sharpness are proxies.',
               'JPEG quantization is a compression proxy, not an inferred encoder quality score.',
               'Sensor categories, contributor attribution and labels are not read by this folder adapter.']
    embeddings = extractor_id = extractor_version = None
    if embedding_file is not None:
        with np.load(embedding_file, allow_pickle=False) as cache:
            expected = {'sample_ids','embeddings','extractor_id','extractor_version'}
            if not expected.issubset(cache.files):
                raise ValueError(f'Embedding cache needs {sorted(expected)}')
            cached_ids = cache['sample_ids'].astype(str).tolist()
            if cached_ids != ids:
                raise ValueError('Embedding sample_ids must exactly match sorted relative image paths')
            embeddings = np.array(cache['embeddings'], dtype=float)
            extractor_id = str(cache['extractor_id'].item())
            extractor_version = str(cache['extractor_version'].item())
            digest.update(hashlib.sha256(Path(embedding_file).read_bytes()).digest())
    return DriftBatch(digest.hexdigest()[:20], tuple(ids),
                      {k:np.asarray(v) for k,v in values.items()}, embeddings,
                      extractor_id, extractor_version, tuple(issues))
