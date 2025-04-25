'''
Test fixtures
'''
from enum import IntEnum, StrEnum
import re
from typing import Optional, Protocol
import warnings
import subprocess
import json
import shutil

from pygltflib import BufferFormat, ImageFormat
import pygltflib as gltf
import pytest

from pathlib import Path

from gltf_builder import Builder
from gltf_builder.core_types import JsonObject, NamePolicy
from gltf_builder.elements import GLTF_LOG
from gltf_builder.protocols import _BuilderProtocol

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
                gl: gltf.GLTF2, /,
                format: Optional[GltfFormat]=None,
                outfile: Optional[Path]=None,
                writeTimestamp: bool=True,
                maxIssues: int=100,
                ignoredIssues: Optional[list[str]]=None,
                severityOverrides: Optional[dict[str, Severity]]=None,
            ) -> None:
        '''
        A function to save the result of a test to both a .gltf and .glb file.
        '''
        ...


def sanitize(name: str):
    '''
    Get the test name with any special characters removed.
    '''
    return RE_PARAAM.sub(r'_\1', name)

class ProxyBuilder(Builder):
    result: gltf.GLTF2|None = None
    save: SaveFn|None = None
    validation: JsonObject|None = None
    '''
    A proxy builder that can be used to test the builder interface.
    '''
    def __init__(self, /, *,
                    index_size: int = -1,
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
        self.save = save
        self.result = None
        self.validation = None

    def build(self, /,
              index_size: Optional[int]=None,
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
            if self.save is not None:
                self.save(result,
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
            or self._accessors.get(name)
            or self._views.get(name)
            or self._buffers.get(name)
            or self.scenes.get(name)
            or self.extensions.get(name)
            or self.extras.get(name)
            or self[name]
        )


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
    def save(b: '_BuilderProtocol|gltf.GLTF2',
            ignoredIssues: Optional[list[str]]=None,
            severityOverrides: Optional[dict[str, Severity]]=None,
            maxIssues: int=100,
            format: Optional[GltfFormat]=None,
            outfile: Optional[Path]=None,
            writeTimestamp: bool=True,
             **params):
        if isinstance(b, _BuilderProtocol):
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
                raise ValueError(f"Validation failed")
            issues = validation['issues']
            if issues['numErrors'] + issues['numWarnings'] > 0:
                msg = '\n'.join(
                    f'{issue["code"]}:{issue["severity"]}: {issue["message"]}'
                    for issue in issues['messages']
                    if issue['severity'] in (Severity.ERROR, Severity.WARNING)
                )
                raise ValueError(f"Validation failed: {msg}")
        return g
    glb.unlink(missing_ok=True)
    gltf.unlink(missing_ok=True)
    return save


@pytest.fixture
def test_builder(request, save):
    builder = ProxyBuilder(
        index_size=-1,
        save=save,
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
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)

        with open(outfile, 'r') as f:
            report = json.load(f)
        return report
    except subprocess.CalledProcessError as e:
         print(f"Error validating glTF: {e}")
         raise
