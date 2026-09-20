import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def clean_photo_corpus(tmp_path):
    """Structured colour/edge/noise fixtures with declared ISO and JPEG quality."""
    root = tmp_path/'clean_photos'
    root.mkdir()
    rng = np.random.default_rng(91)
    yy, xx = np.mgrid[:32, :32]
    for i in range(120):
        luminance = .4 + .055*np.sin(xx/3) + .04*np.cos(yy/4) + rng.normal(0,.004)
        channels = np.stack([luminance+.04, luminance, luminance-.03], axis=-1)
        channels += rng.normal(0,.006,channels.shape)
        exif = Image.Exif()
        exif[34855] = 100 if i % 2 else 200
        Image.fromarray(np.rint(channels*255).astype('uint8')).save(
            root/f'{i:04d}.jpg', quality=95, subsampling=0, exif=exif)
    return root
