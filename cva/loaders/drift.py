"""Canonical Dataset adapter with shared S3/S6/S7 untrusted-image controls.

The existing COCO/YOLO loaders are also accepted by the drift plug-ins. Measurements
are an adapter cache, not a second public Dataset protocol.
"""
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace, median_filter

from cva.core.types import Dataset, Sample
from cva.loaders.datasets.base import InMemoryDataset
from cva.loaders.safety import (
    S3_SANDBOX_LIMITATION,
    UnsafeArtifact,
    check_pixel_budget,
    load_in_sandbox,
    safe_join,
    sha256_file,
)

SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
# Bound before full RGB decode, including images below Pillow's own limit.
MAX_DRIFT_PIXELS = 16_000_000


def dataset_id(dataset: Dataset) -> str:
    h = hashlib.sha256()
    for sample in dataset.samples:
        h.update(sample.sample_id.encode()+b'\0'+sample.content_sha256.encode()+b'\0')
    return h.hexdigest()[:20]


@dataclass
class FeatureTable:
    batch_id: str
    features: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


@dataclass
class MeasuredDataset(InMemoryDataset):
    measurements: FeatureTable | None = None


@dataclass(frozen=True)
class EmbeddingRows:
    """Private adapter for DriftTest's Any distributions; retains observations for KS."""
    sample_ids: tuple[str, ...]
    embeddings: np.ndarray
    extractor_id: str
    extractor_version: str

    def __post_init__(self):
        x = np.asarray(self.embeddings)
        if (x.ndim != 2 or x.shape[0] != len(self.sample_ids) or x.shape[1] == 0
                or not np.isfinite(x).all()):
            raise ValueError('Embeddings must be a finite N x D matrix aligned to sample IDs')
        if not self.extractor_id or not self.extractor_version:
            raise ValueError('Embeddings require extractor identity and version')


def load_embeddings(path, dataset):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as cache:
        expected = {'sample_ids','embeddings','extractor_id','extractor_version'}
        if not expected.issubset(cache.files):
            raise ValueError(f'Embedding cache needs {sorted(expected)}')
        ids = tuple(cache['sample_ids'].astype(str).tolist())
        if ids != tuple(s.sample_id for s in dataset.samples):
            raise ValueError('Embedding sample_ids must exactly match dataset order')
        return EmbeddingRows(ids,np.asarray(cache['embeddings'],float),
                             str(cache['extractor_id'].item()),str(cache['extractor_version'].item()))


def _measure_image(path):
    # Called ONLY inside S3. S7 executes before Image.load, not afterward.
    width,height = check_pixel_budget(path,MAX_DRIFT_PIXELS)
    digest = sha256_file(path)
    try:
        with Image.open(path) as source:
            source.load()
            tables = getattr(source,'quantization',None)
            iso = source.getexif().get(34855)
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((256,256))
            rgb = np.asarray(image,dtype=float)/255.
            grey = rgb @ np.array([.299,.587,.114])
            row = {'brightness':float(grey.mean()),'rms_contrast':float(grey.std()),
                   'sharpness':float(laplace(grey,mode='reflect').var()),
                   'noise_residual':float(np.median(np.abs(grey-median_filter(grey,size=3))))}
            for c,channel in enumerate(('red','green','blue')):
                counts = np.histogram(rgb[:,:,c],bins=8,range=(0,1))[0]
                for i,fraction in enumerate(counts/counts.sum()):
                    row[f'{channel}_hist_{i}'] = float(fraction)
            if tables:
                row['jpeg_quantization_mean'] = float(np.mean([v for t in tables.values() for v in t]))
            if isinstance(iso,(int,float)) and np.isfinite(iso) and iso > 0:
                row['exif_iso'] = float(iso)
    except (OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise UnsafeArtifact(path,'S7',f'Unreadable image: {exc}') from exc
    if digest != sha256_file(path):
        raise UnsafeArtifact(path,'S1','Image changed during decoding')
    return width,height,digest,row


def _table(dataset, rows):
    features,issues = {},[]
    for key in sorted(set().union(*(set(row) for row in rows)) | {'jpeg_quantization_mean','exif_iso'}):
        missing = sum(key not in row for row in rows)
        if not missing:
            features[key] = np.asarray([row[key] for row in rows])
        else:
            issues.append(f'{key} not assessed: {missing}/{len(rows)} images lack metadata.')
    issues += ['Image features use EXIF-oriented RGB thumbnails up to 256 pixels; noise and sharpness are proxies.',
               'JPEG quantization is a compression proxy, not encoder quality.',
               'Sensor categories are not assessed.',S3_SANDBOX_LIMITATION]
    return FeatureTable(dataset_id(dataset),features,tuple(issues))


def _load_folder(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not files:
        raise ValueError(f'No supported images in {root}')
    dataset = MeasuredDataset(root=root)
    rows = []
    for candidate in files:
        sid = candidate.relative_to(root).as_posix()
        path = safe_join(root,sid)
        w,h,digest,row = _measure_image(path)
        dataset.samples.append(Sample(sid,digest,path,w,h))
        rows.append(row)
    dataset.measurements = _table(dataset,rows)
    return dataset


def load_image_dataset(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Image directory does not exist: {root}')
    return load_in_sandbox('cva.loaders.drift:_load_folder',root)


def _measure_manifest(path):
    # Manifest is generated by the trusted adapter from canonical Sample objects.
    spec = json.loads(path.read_text())
    rows = []
    for sample in spec:
        image = safe_join(Path(sample['parent']),sample['name'])
        _,_,digest,row = _measure_image(image)
        if digest != sample['sha256']:
            raise UnsafeArtifact(image,'S1','Sample digest changed since dataset ingestion')
        rows.append(row)
    return rows


def measure_dataset(dataset):
    if isinstance(dataset,MeasuredDataset) and dataset.measurements is not None:
        return dataset.measurements
    if not dataset.samples:
        raise ValueError('Cannot measure an empty dataset')
    # S6 revalidates the loader's declared root when one exists.
    root = getattr(dataset,'root',None)
    spec = []
    for sample in dataset.samples:
        path = sample.path.absolute()
        if root is not None:
            path = safe_join(Path(root),str(path))
        spec.append({'parent':str(path.parent),'name':path.name,'sha256':sample.content_sha256})
    with tempfile.TemporaryDirectory(prefix='cva-drift-') as temp:
        manifest = Path(temp)/'images.json'
        manifest.write_text(json.dumps(spec))
        rows = load_in_sandbox('cva.loaders.drift:_measure_manifest',manifest)
    return _table(dataset,rows)
