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
def testing_output_dir():
    '''
    Provide a directory to save test results from all tests.
    '''
    dir = Path(__file__).parent / 'out'
    dir.mkdir(exist_ok=True)
    return dir


@pytest.fixture
def module_out(testing_output_dir, request) -> Path:
    '''
    Provide a directory to save test results from a test module
    (multiple tests).
    '''

    testdir = testing_output_dir / request.node.parent.name
    testdir = testdir.with_suffix('')
    testdir.mkdir(exist_ok=True)
    return testdir


def unlink_tree(path: Path):
    '''
    Recursively delete a directory and its contents.
    '''
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                unlink_tree(child)
            else:
                child.unlink()
        path.rmdir()
    else:
        path.unlink(missing_ok=True)


@pytest.fixture
def out_file(module_out, request) -> Path:
    '''
    Provide a path to save test results into a file.

    Use the `out_dir` fixture to get a directory to save multiple
    test results.
    '''
    name = request.node.name
    if name.startswith('test_geo_'):
        name = name[9:]
    elif name.startswith('test_'):
        name = name[5:]
    file: Path = module_out / name
    unlink_tree(file)
    return file


@pytest.fixture
def out_dir(out_file):
    '''
    Provide a directory to save test results.
    '''
    unlink_tree(out_file)
    out_file.mkdir(exist_ok=True)
    return out_file


@pytest.fixture
def save(out_dir):
    '''

    Save the result of a test to both a .gltf and .glb file.

    NOTE: Calling save(g) will modify the g object to store images
    and binary data in a data: URI format. This means the binary blob
    is no longer available. If you need the binary blob, call save(g)
    after you are done accessing the binary blob, or change the format
    back.
    '''
    def save(g):
        g.convert_images(ImageFormat.BUFFERVIEW)
        g.convert_buffers(BufferFormat.DATAURI)

        out = out_dir / out_dir.name
        g.save_json(out.with_suffix('.gltf'))
        g.save_binary(out.with_suffix('.glb'))
    return save


@pytest.fixture
def test_builder(request, save):
    builder = Builder(
        index_size=-1,
        name_mode=NameMode.UNIQUE,
        extras={
            'gltf_builder': {
                test: {
                    'module': request.node.parent.name,
                    'test': request.node.name,
                }
            }
        }
    )
    yield builder
    result = builder.build()
    save(result)
