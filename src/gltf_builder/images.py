'''
Image data for textures in glTF format.
'''


from typing import Any, Optional, cast
from pathlib import Path

import pygltflib as gltf
import numpy as np

from gltf_builder.compiler import _Scope, _DoCompileReturn, _CompileState
from gltf_builder.core_types import (
    BufferViewTarget, ImageType, JsonObject, Phase, ScopeName
)
from gltf_builder.elements import BImage
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.utils import std_repr


class _ImageState(_CompileState[gltf.Image, '_ImageState']):
    '''
    State for the compilation of an image.
    '''
    __memory: memoryview|None = None
    @property
    def memory(self) -> memoryview:
        if self.__memory is None:
            raise ValueError('Image memory not available')
        return self.__memory

class _Image(BImage):
    '''
    Implementation class for `BImage`.
    '''
    
    @classmethod
    def state_type(cls):
        return _ImageState
    

    __memory: memoryview|None = None
    _scope_name = ScopeName.IMAGE
    
    @property
    def mimeType(self) -> str:
        '''
        The MIME type for the image data.
        '''
        match self.imageType:
            case ImageType.JPEG:
                return 'image/jpeg'
            case ImageType.PNG:
                return 'image/png'

    def __init__(self,
                 /,
                 name: str='',
                 blob: Optional[bytes|np.ndarray[tuple[int], np.dtype[np.uint8]]]=None,
                 uri: str|Path|None=None,
                 imageType: ImageType=ImageType.PNG,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions,
        )
        match blob:
            case np.ndarray():
                self.blob = blob.tobytes()
            case bytes():
                self.blob = blob
        self.uri = uri
        self.imageType = imageType

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            blob=self.blob,
            uri=self.uri,
            imageType=self.imageType,
        )

    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState[gltf.Image, _ImageState],
                    /) -> _DoCompileReturn[gltf.Image]:
        match phase:
            case Phase.COLLECT:
                builder.images.add(self)
                if self.blob is not None:
                    name=builder._gen_name(self, scope=ScopeName.BUFFER_VIEW)
                    self.view = scope._get_view(builder.buffer,
                                      BufferViewTarget.ARRAY_BUFFER,
                                      name=name,
                    )
                    return [self.view.compile(builder, scope, phase,)]
            case Phase.SIZES:
                return len(self.blob) if self.blob is not None else 0
            case Phase.OFFSETS:
                if self.view is not None:
                    assert self.blob is not None
                    self.__memory = self.view.memoryview(0, len(self.blob))
                return 0
            case Phase.BUILD:
                if self.view is not None:
                    assert self.blob is not None
                    assert self.__memory is not None
                    self.__memory[:] = self.blob
                    self.view.compile(builder, scope, Phase.BUILD)
                img = gltf.Image(
                        name=self.name,
                        #pygltflib is sloppy about types
                        uri=cast(str, self.uri),
                        mimeType=self.mimeType,
                        extras=self.extras,
                        extensions=self.extensions,
                    )
                return img
            case _: pass
    
    def __repr__(self):
        return std_repr(self, (
            'name',
            'uri',
            'imageType',
        ))
    
def image(
    name: str='', /,
    blob: Optional[bytes|np.ndarray[tuple[int], np.dtype[np.uint8]]]=None,
    uri: str|Path|None=None,
    imageType: ImageType=ImageType.PNG,
    extras: Optional[JsonObject]=None,
    extensions: Optional[JsonObject]=None,
) -> _Image:
    '''
    Create an image for a texture.
    
    Parameters
    ----------
    name : str, optional
        The name of the image.
    blob : bytes or numpy.ndarray, optional
        The image data as a byte string or a numpy array.
    uri : str or Path, optional
        The URI of the image.
    imageType : ImageType, optional
        The type of the image (JPEG or PNG).
    extras : dict, optional
        Extra data to be stored with the image.
    extensions : dict, optional
        Extensions to be stored with the image.
    
    Returns
    -------
    _Image
        An instance of _Image containing the provided data.
    '''
    return _Image(
        name=name,
        blob=blob,
        uri=uri,
        imageType=imageType,
        extras=extras,
        extensions=extensions,
    )
