'''
Wrapper for the glTF asset object.
'''

from typing import Optional, Any
from importlib.metadata import version

import pygltflib as gltf

__version__  = version('gltf-builder')

GENERATOR = f'gltf-builder@v{__version__}/pygltflib@v{gltf.__version__}'

class BAsset(gltf.Asset):
    '''
    Wrapper for the glTF `Asset`` object.
    '''
    def __init__(self,
                 generator: Optional[str]=GENERATOR,
                 copyright: Optional[str]=None,
                 version: str='2.0',
                 minVersion: Optional[str]=None,
                 extras: Optional[dict[str, Any]]=None,
                 extensions: Optional[dict[str, Any]]=None,
                 **kwargs):
        super().__init__(generator=generator,
                         version=version,
                         copyright=copyright,
                         minVersion=minVersion,
                         extras=extras,
                         extensions=extensions,
                         **kwargs)