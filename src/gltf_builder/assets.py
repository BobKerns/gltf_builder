'''
Wrapper for the glTF asset object.
'''

from typing import Optional, Any
from importlib.metadata import version

import pygltflib as gltf

from gltf_builder.compiler import _CompileState

__version__  = version('gltf-builder')

GENERATOR = f'gltf-builder@v{__version__}/pygltflib@v{gltf.__version__}'
'''
The default value for the `generator` field in the `Asset`.
'''


class _AssetState(_CompileState[gltf.Asset, '_AssetState']):
    '''
    State for the compilation of an asset.
    '''
    pass

class BAsset(gltf.Asset):
    '''
    Wrapper for the glTF `Asset`` object.
    '''

    @classmethod
    def state_type(cls):
        return _AssetState
    

    def __init__(self,
                 generator: Optional[str]=GENERATOR,
                 copyright: Optional[str]=None,
                 version: str='2.0',
                 minVersion: Optional[str]=None,
                 extras: Optional[dict[str, Any]]=None,
                 extensions: Optional[dict[str, Any]]=None,
                 ):
        super().__init__(generator=generator,
                         version=version,
                         copyright=copyright,
                         minVersion=minVersion,
                         extras=extras,
                         extensions=extensions,
                         )