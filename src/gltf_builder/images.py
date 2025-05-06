'''
Image data for textures in glTF format.
'''


from typing import Optional, cast, TYPE_CHECKING
from pathlib import Path

import pygltflib as gltf
import numpy as np

from gltf_builder.compiler import _DoCompileReturn, _GlobalCompileState
from gltf_builder.core_types import (
    BufferViewTarget, ExtensionsData, ExtrasData, ImageType, Phase, EntityType
)
from gltf_builder.entities import BBufferView, BImage
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


class _ImageState(_GlobalCompileState[gltf.Image, '_ImageState', '_Image']):
    '''
    State for the compilation of an image.
    '''
    view: Optional[BBufferView] = None
    memory: memoryview|None = None

    __blob: bytes|None = None
    @property
    def blob(self) -> bytes:
        if self.__blob is None:
            if self.memory is None:
                raise ValueError('Image memory not available')
            self.__blob = self.memory.tobytes()
            return self.__blob
        return self.__blob

class _Image(BImage):
    '''
    Implementation class for `BImage`.
    '''

    @classmethod
    def state_type(cls):
        return _ImageState

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
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
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

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _ImageState,
                    /) -> _DoCompileReturn[gltf.Image]:
        match phase:
            case Phase.COLLECT:
                globl.add(self)
                if self.blob is not None:
                    name=globl._gen_name(self, entity_type=EntityType.BUFFER_VIEW)
                    state.view = globl.get_view(globl.buffer,
                                      BufferViewTarget.ARRAY_BUFFER,
                                      name=name,
                    )
                    return [state.view.compile(globl, phase,)]
            case Phase.SIZES:
                return len(self.blob) if self.blob is not None else 0
            case Phase.OFFSETS:
                if state.view is not None:
                    assert state.blob is not None
                    v_state = globl.state(state.view)
                    state.memory = v_state.memory[0:len(state.blob)]
                return 0
            case Phase.BUILD:
                if self.view is not None:
                    assert state.blob is not None
                    memory = state.memory
                    blob = state.blob
                    assert memory is not None
                    assert blob is not None
                    memory[:] = blob
                    self.view.compile(globl, Phase.BUILD)
                img = gltf.Image(
                        name=self.name,
                        #pygltflib is sloppy about types
                        # uri can be str or None
                        uri=cast(str, str(self.uri) if self.uri else None),
                        mimeType=self.mimeType,
                        extras=self.extras,
                        extensions=self.extensions,
                    )
                return img
            case _: pass

    def __repr__(self): # pragma: no cover
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
    extras: Optional[ExtrasData]=None,
    extensions: Optional[ExtensionsData]=None,
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
