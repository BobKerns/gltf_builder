'''
Test fixtures
'''
import re

from pygltflib import BufferFormat, ImageFormat
import pytest

from pathlib import Path

from gltf_builder import Builder, NameMode
from gltf_builder.elements import GLTF_LOG

LOG = GLTF_LOG.getChild(Path(__file__).stem)


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

RE_PARAAM = re.compile(r'\[([a-zA-Z0-9_+=@-]*)\]')

@pytest.fixture
def out_dir(module_out, request):
    '''
    Provide a directory to save test results.
    '''
    name = request.node.name
    
    if name.startswith('test_geo_'):
        name = name[9:]
    elif name.startswith('test_'):
        name = name[5:]
    name = RE_PARAAM.sub('', name)
    dir: Path = module_out / name
    dir.mkdir(exist_ok=True)
    return dir


def sanitize(name: str):
    '''
    Get the test name with any special characters removed.
    '''
    return RE_PARAAM.sub(r'_\1', name)


@pytest.fixture
def save(out_dir, request):
    '''

    Save the result of a test to both a .gltf and .glb file.

    NOTE: Calling save(g) will modify the g object to store images
    and binary data in a data: URI format. This means the binary blob
    is no longer available. If you need the binary blob, call save(g)
    after you are done accessing the binary blob, or change the format
    back.
    '''
    out = out_dir / sanitize(request.node.name)
    gltf = out.with_suffix('.gltf')
    glb = out.with_suffix('.glb')
    def save(g, **params):
        if isinstance(g, Builder):
            g = g.build(**params)
        g.convert_images(ImageFormat.BUFFERVIEW)
        g.convert_buffers(BufferFormat.DATAURI)

        LOG.info('Writing to %s{.gltf,.glb}', out.with_suffix(""))
        g.save_json(gltf)
        g.save_binary(glb)
        # Convert back for the tests.
        g.convert_buffers(BufferFormat.BINARYBLOB)
        return g
    glb.unlink(missing_ok=True)
    gltf.unlink(missing_ok=True)
    return save


@pytest.fixture
def test_builder(request, save):
    builder = Builder(
        index_size=-1,
        extras={
            'gltf_builder': {
                'test': {
                    'module': request.node.parent.name,
                    'test': request.node.name,
                }
            }
        }
    )
    yield builder
    result = builder.build()
    save(result)
