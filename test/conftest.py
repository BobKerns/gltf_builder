'''
Test fixtures
'''
from pygltflib import BufferFormat, ImageFormat
import pytest

from pathlib import Path

import test

from gltf_builder import Builder
from gltf_builder.element import NameMode


@pytest.fixture
def outdir():
    dir = Path(__file__).parent / 'out'
    dir.mkdir(exist_ok=True)
    return dir

@pytest.fixture
def test_builder(outdir, request):
    builder = Builder(
        index_size=-1,
        name_mode=NameMode.UNIQUE,
        extras={
            'gltf_builder': {
                test: {
                    'test': request.node.name,
                }
            }
        }
    )
    yield builder
    result = builder.build()
    result.convert_buffers(BufferFormat.DATAURI)
    result.convert_images(ImageFormat.BUFFERVIEW)
    result.save_json(outdir / f'{request.node.name}.gltf')
    result.save_binary(outdir / f'{request.node.name}.glb')