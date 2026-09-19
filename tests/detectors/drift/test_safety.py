import struct
import subprocess
import sys
import zlib

import pytest
from PIL import Image

from cva.loaders.drift import MAX_DRIFT_PIXELS, load_image_dataset
from cva.loaders.safety import UnsafeArtifact


def header_png(width,height):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(b'\0'))+chunk(b'IEND',b''))


@pytest.mark.parametrize('width,height',[(20_000,20_000),(5000,5000)])
def test_header_budget_rejects_before_decode(tmp_path,width,height):
    assert width*height > MAX_DRIFT_PIXELS
    (tmp_path/'bomb.png').write_bytes(header_png(width,height))
    with pytest.raises(UnsafeArtifact,match='S7'):
        load_image_dataset(tmp_path)


def test_cli_bomb_is_concise_error_not_traceback(tmp_path):
    (tmp_path/'bomb.png').write_bytes(header_png(20000,20000))
    result = subprocess.run([sys.executable,'-m','cva.cli','drift','--incoming',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 2
    assert 'S7' in result.stderr and 'Traceback' not in result.stderr


def test_symlink_escape(tmp_path):
    images = tmp_path/'images'
    images.mkdir()
    Image.new('RGB',(8,8)).save(tmp_path/'outside.png')
    (images/'escape.png').symlink_to(tmp_path/'outside.png')
    with pytest.raises(UnsafeArtifact,match='S6'):
        load_image_dataset(images)


def test_shared_cli_stays_light_for_drift():
    result = subprocess.run([sys.executable,'-c',
        "import sys; import cva.cli; assert 'torch' not in sys.modules; assert 'onnx' not in sys.modules"],
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
