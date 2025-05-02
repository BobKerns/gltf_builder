'''
Test fixtures
'''
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
import re
from typing import Optional, Protocol
import warnings
import subprocess
import json
import shutil
import logging
from threading import Lock


from pygltflib import BufferFormat, ImageFormat
import pygltflib as gltf
import pytest

from pathlib import Path

from gltf_builder import Builder
from gltf_builder.compiler import ExtensionsData, ExtrasData
from gltf_builder.core_types import IndexSize, JsonObject, NamePolicy, PrimitiveMode
from gltf_builder.elements import GLTF_LOG, BMesh, BNode
from gltf_builder.extensions import load_extensions, _EXTENSION_PLUGINS
from gltf_builder.geometries import _CUBE, _CUBE_FACE1, _CUBE_FACE2, _CUBE_FACE3, _CUBE_FACE4, _CUBE_FACE5, _CUBE_FACE6

LOG = GLTF_LOG.getChild(Path(__file__).stem)

LOCK = Lock()
'''
A lock for actions that should not be run in parallel.
This is used to prevent multiple tests from running at the same time.

In case we ever do that. I have a strong aversion to tests that
"temporarily" modify global state.
'''

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

RE_PARAM = re.compile(r'\[([a-zA-Z0-9_+=@-]*)\]')

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
    name = RE_PARAM.sub('', name)
    dir: Path = module_out / name
    dir.mkdir(exist_ok=True)
    return dir

class Severity(IntEnum):
    ERROR = 0
    WARNING = 1
    INFORMATION = 2
    HINT = 3


class GltfFormat(StrEnum):
    GLTF = 'gltf'
    GLB = 'glb'


class SaveFn(Protocol):
    def __call__(self,
                gl: gltf.GLTF2, /, *,
                format: Optional[GltfFormat]=None,
                outfile: Optional[Path]=None,
                writeTimestamp: bool=True,
                maxIssues: int=100,
                ignoredIssues: Optional[list[str]]=None,
                severityOverrides: Optional[dict[str, Severity]]=None,
            ) -> gltf.GLTF2:
        '''
        A function to save the result of a test to both a .gltf and .glb file.
        '''
        ...


def sanitize(name: str):
    '''
    Get the test name with any special characters removed.
    '''
    return RE_PARAM.sub(r'_\1', name)

class ProxyBuilder(Builder):
    result: gltf.GLTF2|None = None
    _save: SaveFn|None = None
    validation: JsonObject|None = None
    '''
    A proxy builder that can be used to test the builder interface.
    '''
    def __init__(self, /, *,
                    index_size: Optional[IndexSize] = None,
                    name_policy: Optional[NamePolicy]=None,
                    extras: Optional[dict] = None,
                    extensions: Optional[dict] = None,
                    save: Optional[SaveFn] = None,
                    ) -> None:
        super().__init__(
            index_size=index_size,
            name_policy=name_policy,
            extras=extras,
            extensions=extensions,
        )
        self._save = save
        self.result = None
        self.validation = None

    def build(self, /,
              index_size: Optional[IndexSize]=None,
              ignoredIssues: Optional[list[str]]=None,
              severityOverrides: Optional[dict[str, Severity]]=None,
              maxIssues: int=100,
              format: Optional[GltfFormat]=None,
              outfile: Optional[Path]=None,
              writeTimestamp: bool=True,
              ) -> gltf.GLTF2:
        if self.result is not None:
            return self.result
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = super().build(index_size=index_size)
            self.result = result
            if self._save is not None:
                self._save(result,
                          ignoredIssues=ignoredIssues,
                          severityOverrides=severityOverrides,
                          maxIssues=maxIssues,
                          format=format,
                          outfile=outfile,
                          writeTimestamp=writeTimestamp,
                          )
            return result

    def __getitem__(self, name: str): # type: ignore[override]
        return (
            self.nodes.get(name)
            or self.meshes.get(name)
            or self.cameras.get(name)
            or self.materials.get(name)
            or self.textures.get(name)
            or self.samplers.get(name)
            or self.images.get(name)
            or self.scenes.get(name)
            #or self.animations.get(name)
            or self.skins.get(name)
            or self.accessors.get(name)
            or self.views.get(name)
            or self.buffers.get(name)
            or self.scenes.get(name)
            or self.extensions.get(name)
            or self.extras.get(name)
            or self[name]
        )


@pytest.fixture
def save(out_dir: Path,
         request: pytest.FixtureRequest) -> SaveFn:
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
    def save(b: 'Builder|gltf.GLTF2',
            ignoredIssues: Optional[list[str]]=None,
            severityOverrides: Optional[dict[str, Severity]]=None,
            maxIssues: int=100,
            format: Optional[GltfFormat]=None,
            outfile: Optional[Path]=None,
            writeTimestamp: bool=True,
            **params) -> 'gltf.GLTF2':

        if isinstance(b, Builder):
            g = b.build(**params)
        else:
            g = b
        extras = g.extras or {}
        if g.extras is None:
            g.extras = extras = {}
        if extras.get('gltf_builder') is None:
            extras['gltf_builder'] = {}
        if outfile is None:
            outfile = out.with_suffix('.json')

        validation = extras['gltf_builder'].get('validation')
        if validation is None:
            g.convert_images(ImageFormat.BUFFERVIEW)
            g.convert_buffers(BufferFormat.DATAURI)

            LOG.info('Writing to %s{.gltf,.glb,.json}', out.with_suffix(""))
            g.save_json(gltf)
            g.save_binary(glb)
            # Convert back for the tests.
            g.convert_buffers(BufferFormat.BINARYBLOB)
            validation = validate_gltf(gltf,
                                    outfile=outfile,
                                    format=format,
                                    maxIssues=maxIssues,
                                    writeTimestamp=writeTimestamp,
                                    severityOverrides=severityOverrides,
                                    ignoredIssues=ignoredIssues,
                                )
            extras['gltf_builder']['validation'] = validation
            if validation is None:
                raise ValueError("Validation failed")
            issues = validation['issues']
            if issues['numErrors'] + issues['numWarnings'] > 0:
                def fmt_issue(issue):
                    code = issue['code']
                    severity = issue['severity']
                    message = issue['message']
                    pointer = issue['pointer']
                    return f'{code}:{severity}: {message} ({pointer})'
                msg = '\n'.join(
                    fmt_issue(issue)
                    for issue in issues['messages']
                    if issue['severity'] in (Severity.ERROR, Severity.WARNING)
                )
                raise ValueError(f"Validation failed: {msg}")
        return g
    glb.unlink(missing_ok=True)
    gltf.unlink(missing_ok=True)
    return save


@pytest.fixture
def builder_extras(request):
    '''
    A dictionary of extras to add to the builder and the resulting glTF file.
    '''
    return {
            'gltf_builder': {
                'test': {
                    'module': request.node.parent.name,
                    'test': request.node.name,
                }
            }
        }

class BuilderContext(Protocol):
    @contextmanager
    @abstractmethod
    def __call__(self,
                 index_size: Optional[IndexSize]=None,
                 name_policy: Optional[NamePolicy]=None,
                 extras: Optional[dict]=None,
                 extensions: Optional[dict]=None,
                 save: Optional[SaveFn]=None,
            ) -> Generator[ProxyBuilder, None, None]:
        '''
        A context manager to create a test builder.
        '''
        ...

@pytest.fixture
def test_builder(request: pytest.FixtureRequest,
                 save: SaveFn,
                 builder_extras: ExtrasData,) -> BuilderContext:
    '''
    A fixture to create a test builder.
    '''
    outer_save = save
    @contextmanager
    def test_builder(
            index_size: Optional[IndexSize]=None,
            name_policy: Optional[NamePolicy]=None,
            extras: Optional[dict]=None,
            extensions: Optional[dict]=None,
            save: Optional[SaveFn]=None,
        ) -> Generator[ProxyBuilder, None, None]:
        if extras is None:
            extras = {}
        if extensions is None:
            extensions = {}
        if save is None:
            save = outer_save
        b = ProxyBuilder(
                index_size=index_size,
                name_policy=name_policy,
                save=save,
                extras=dict(builder_extras)
            )
        yield b
        if b.result is not None:
            result = b.build()
            save(result)
    return test_builder

VALIDATOR_JS = Path(__file__).parent / 'gltf-validator.js'
NODE = shutil.which('node')

IGNORED_ISSUES = [
    'UNUSED_OBJECT',
    'UNSUPPORTED_EXTENSION',
]

SEVERITY_OVERRIDES: dict[str, Severity] = {
}

def validate_gltf(file_path: Path,
                  ignoredIssues: Optional[list[str]]=None,
                  severityOverrides: Optional[dict[str, Severity]]=None,
                  maxIssues: int=100,
                  format: Optional[GltfFormat]=None,
                  outfile: Optional[Path]=None,
                  writeTimestamp: bool=True,
                  ) -> Optional[dict]:
    """
    Validates a glTF file using gltf-validator and returns the report as a dictionary.

    Parameters
    ----------
        file_path (Path): Path to the glTF file to validate.
        ignoredIssues (Optional[list[str]]): List of issues to ignore during validation.
        severityOverrides (Optional[dict[str, Severity]]): Dictionary of issue severities to override.
        maxIssues (int): Maximum number of issues to report.
        format (Optional[GltfFormat]): Format of the glTF file (GLTF or GLB).
        outfile (Optional[Path]): Path to save the validation report. If None, defaults to file_path with .json extension.

    Returns
    -------
        dict: The validation report as a dictionary.
    """
    outfile = outfile or file_path.with_suffix('.json')
    if ignoredIssues is None:
        ignoredIssues = IGNORED_ISSUES
    if severityOverrides is None:
        severityOverrides = SEVERITY_OVERRIDES
    if outfile.exists():
        outfile.unlink()
    if not NODE:
        raise FileNotFoundError("node not found. Ensure it is installed and in your PATH.")
    try:
        arg_ignoredIssues = ()
        if ignoredIssues:
            arg_ignoredIssues = '--ignoredIssues', ','.join(ignoredIssues)
        arg_severityOverrides = ()
        if severityOverrides:
            arg_severityOverrides = '--severityOverrides', ','.join(
                f"{issue}:{severity}"
                for issue, severity in severityOverrides.items()
            )
        arg_format = ()
        if format:
            arg_format = '--format', format.value
        arg_writeTimestamp = ()
        if not writeTimestamp:
            arg_writeTimestamp = '--writeTimestamp', 'false'

        cmd = [str(w) for w in (
            NODE, VALIDATOR_JS,
            '--out', outfile,
            *arg_ignoredIssues,
            '--maxIssues', maxIssues,
            *arg_severityOverrides,
            *arg_writeTimestamp,
            *arg_format,
            file_path,
        )]
        LOG.debug(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

        with open(outfile, 'r') as f:
            report = json.load(f)
        return report
    except subprocess.CalledProcessError as e:
         print(f"Error validating glTF: {e}")
         raise

TEST_EXTRAS: ExtrasData={"EXTRA": "DATA"}
TEST_EXTENSIONS: ExtensionsData={"TEST_extension": {"EXTRA": "DATA"}}

@pytest.fixture
def plugins():
    '''
    Clears, loads, and returns a map of available extension plugins.
    It then clears them again after the test.
    '''
    _EXTENSION_PLUGINS.clear()
    load_extensions()
    yield _EXTENSION_PLUGINS
    _EXTENSION_PLUGINS.clear()


@pytest.fixture
def no_plugins():
    '''
    Clears the extension plugins.
    '''
    _EXTENSION_PLUGINS.clear()
    yield _EXTENSION_PLUGINS
    _EXTENSION_PLUGINS.clear()

@pytest.fixture(params=[
    (IndexSize.NONE, 0, 0),
    (IndexSize.AUTO, 1, 1),
    (IndexSize.UNSIGNED_BYTE, 1, 1),
    (IndexSize.UNSIGNED_SHORT, 2,1 ),
    (IndexSize.UNSIGNED_INT, 4, 1),
])
def index_sizes(
        request
    ) -> tuple[IndexSize, int, int]:
    '''
    Provide a list of index sizes to test.
    '''
    return request.param


@dataclass
class GeometryData:
    builder: Builder
    meshes: dict[str, BMesh] = field(default_factory=dict)
    nodes: dict[str, BNode] = field(default_factory=dict)
    save: Callable[[gltf.GLTF2], gltf.GLTF2] = lambda g, **kwargs: g
    def build(self, **kwargs):
        return self.save(self.builder.build(**kwargs))
    def __getitem__(self, name):
        return (
            self.nodes.get(name)
            or self.meshes.get(name)
            or self.builder[name]
        )
    @property
    def index_size(self):
        return self.builder.index_size
    @index_size.setter
    def index_size(self, size):
        self.builder.index_size = size


@pytest.fixture(scope='function')
def cube(save):
    b = Builder()
    m = b.create_mesh('CUBE_MESH')
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE1))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE2))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE3))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE4))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE5))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE6))
    top = b.create_node('TOP')
    top.create_node('CUBE', mesh=m)
    yield GeometryData(builder=b,
                   meshes={'CUBE_MESH': m},
                   nodes={'TOP': top},
                   save=save,
                )


@pytest.fixture()
def DEBUG(request):
    '''
    A fixture to enable `DEBUG` logging.
    '''
    from gltf_builder.log import GLTF_LOG
    with LOCK:
        old = GLTF_LOG.level
        GLTF_LOG.setLevel(logging.DEBUG)
        yield GLTF_LOG.getChild(Path(request.node.name).stem)
        GLTF_LOG.setLevel(old)
