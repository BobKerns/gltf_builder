'''
Wrapper for the glTF asset object.
'''

from typing import Optional, TYPE_CHECKING
from importlib.metadata import version

import pygltflib as gltf

from gltf_builder.compiler import _CompileState, Phase
from gltf_builder.core_types import ExtensionsData, ExtrasData
from gltf_builder.entities import BAsset
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


__version__  = version('gltf-builder')

GENERATOR = f'gltf-builder@v{__version__}/pygltflib@v{gltf.__version__}'
'''
The default value for the `generator` field in the `Asset`.
'''


class _AssetState(_CompileState[gltf.Asset, '_AssetState', '_Asset']):
    '''
    State for the compilation of an asset.
    '''
    pass

class _Asset(BAsset):
    '''
    Wrapper for the glTF `Asset`` object.
    '''
    @classmethod
    def state_type(cls):
        return _AssetState


    def __init__(self, name: str='', /,
                 generator: Optional[str]=GENERATOR,
                 copyright: Optional[str]=None,
                 version: str='2.0',
                 minVersion: Optional[str]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 ):
        if extras is None:
            extras = {}
        if generator is None:
            generator = GENERATOR
        if name:
            extras = {
                **(extras or {}),
                'glt_builder': {
                    'name': name
                },
            }
        super().__init__(name,
                         extras=extras,
                         extensions=extensions,
                         )
        self.generator = generator
        self.copyright = copyright
        self.version = version
        self.minVersion = minVersion

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _AssetState,
                    /):
        match phase:
            case Phase.BUILD:
                return gltf.Asset(
                    generator=self.generator or GENERATOR,
                    version=self.version or '2.0',
                    minVersion=self.minVersion,
                    copyright=self.copyright,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                pass

def asset(
        name: str='',
        /,
        generator: Optional[str]=GENERATOR,
        version: str='2.0',
        minVersion: Optional[str]=None,
        copyright: Optional[str]=None,
        extras: Optional[ExtrasData]=None,
        extensions: Optional[ExtensionsData]=None,
        ):
    '''
    Create a new `BAsset` object.

    Parameters
    ----------
    name : str, optional
        The name of the asset.
    generator : Optional[str], optional
        The generator of the asset.
    version : str, optional
        The version of the asset.
    minVersion : Optional[str], optional
        The minimum version of the asset.

    Returns
    -------
    BAsset
        The created asset object.
    '''
    return _Asset(
        name,
        generator=generator,
        version=version,
        minVersion=minVersion,
        copyright=copyright,
        extras=extras,
        extensions=extensions,
    )
