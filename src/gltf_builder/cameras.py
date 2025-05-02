'''
Functions for creating cameras in glTF format.
'''

from typing import Any, Literal, Optional, TypeVar, overload

import pygltflib as gltf

from gltf_builder.core_types import CameraType, Phase
from gltf_builder.elements import BCamera, BOrthographicCamera, BPerspectiveCamera
from gltf_builder.utils import std_repr
from gltf_builder.compiler import _CompileState, ExtensionsData, ExtrasData
from gltf_builder.global_state import GlobalState


_CAMERA = TypeVar('_CAMERA', bound='_Camera')

class _CameraState(_CompileState[gltf.Camera, '_CameraState', _CAMERA]):
    '''
    State for the compilation of a camera.
    '''
    pass

class _Camera(BCamera):
    '''
    Implementation class for `BCamera`.
    '''

    @classmethod
    def state_type(cls):
        return _CameraState

    def __init__(self,
                 name: str='',
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 type_extras: Optional[ExtrasData]=None,
                 type_extensions: Optional[ExtensionsData]=None,
                ):
        super().__init__(
            name,
            extras=extras,
            extensions=extensions
        )
        self.type_extras: dict[str, Any] = dict(type_extras or ())
        self.type_extensions: dict[str, Any] = dict(type_extensions or ())


class _PerspectiveCamera(_Camera, BPerspectiveCamera, BCamera):
    '''
    Builder representation of a glTF Perspective Camera
    '''

    def __init__(self,
                 name: str='',
                 /,
                 yfov: float=1.0,
                 znear: float=0.1,
                 zfar: float=100.0,
                 aspectRatio: Optional[float]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 type_extras: Optional[ExtrasData]=None,
                 type_extensions: Optional[ExtensionsData]=None,
                 ):
        super().__init__(
            name,
            extras=extras,
            extensions=extensions,
            type_extras=type_extras,
            type_extensions=type_extensions,
        )
        self.yfov=yfov
        self.znear=znear
        self.zfar=zfar
        self.aspectRatio=aspectRatio

    def _do_compile(self,
                    builder: GlobalState,
                    phase: Phase,
                    state: _CompileState[gltf.Camera, _CameraState, '_PerspectiveCamera'],
                    /
                ):
        match phase:
            case Phase.BUILD:
                return gltf.Camera(
                    type=self.type,
                    name=self.name,
                    perspective=self.perspective,
                    extras=self.extras,
                    extensions=self.extensions,
                )

    def __repr__(self):
        return std_repr(self, (
            'yfov',
            'znear',
            'zfar',
            'aspectRatio',
        ))


class _OrthographicCamera(_Camera, BOrthographicCamera):
    '''
    Builder representation of a glTF Orthographic Camera
    '''

    def __init__(self,
                name: str='',
                /,
                xmag: float=1.0,
                ymag: float=1.0,
                znear: float=0.1,
                zfar: float=100.0,
                 aspectRatio: Optional[float]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 type_extras: Optional[ExtrasData]=None,
                 type_extensions: Optional[ExtensionsData]=None,
                ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions,
            type_extras=type_extras,
            type_extensions=type_extensions,
        )
        self.xmag=xmag
        self.ymag=ymag
        self.znear=znear
        self.zfar=zfar

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            xmag=self.xmag,
            ymag=self.ymag,
            znear=self.znear,
            zfar=self.zfar,
            type_extras=dict(self.type_extras),
            type_extensions=dict(self.type_extensions),
        )

    def _do_compile(self,
                    builder: GlobalState,
                    phase: Phase,
                    state: _CompileState[gltf.Camera, _CameraState, '_OrthographicCamera'],
                    ):
        match phase:
            case Phase.BUILD:
                return gltf.Camera(
                    orthographic=self.orthographic,
                    type=self.type,
                    name=self.name,
                    extras=self.extras,
                    extensions=self.extensions,
                )

    def __repr__(self):
        return std_repr(self, (
            'xmag',
            'ymag',
            'znear',
            'zfar',
        ))


@overload
def camera(type: Literal[CameraType.PERSPECTIVE],
            name: str='',
            /, *,
            yfov: float=1.0,
            znear: float=0.1,
            zfar: float=100.0,
            aspectRatio: float|None=None,
            extras: dict|None=None,
            extensions: dict|None=None,
            ) -> BPerspectiveCamera: ...
@overload
def camera(type: Literal[CameraType.ORTHOGRAPHIC],
            name: str='',
            /, *,
            xmag: float=1.0,
            ymag: float=1.0,
            znear: float=0.1,
            zfar: float=100.0,
            extras: dict|None=None,
            extensions: dict|None=None,
           ) -> BOrthographicCamera: ...
def camera(type: CameraType,
            name: str='',
            /,
            xmag: float=1.0,
            ymag: float=1.0,
            yfov: float=1.0,
            znear: float=0.1,
            zfar: float=100.0,
            aspectRatio: float|None=None,
            extras: dict|None=None,
            extensions: dict|None=None,
            ) -> BCamera:
    '''
    Create a new camera.

    Parameters
    ----------
    type : CameraType
        The type of the camera.
    name : str, optional
        The name of the camera.
    xmag : float, optional
        The x magnification of the camera. [For orthographic cameras]
    ymag : float, optional
        The y magnification of the camera. [For orthographic cameras]
    yfov : float, optional
        The y field of view of the camera. [For perspective cameras]
    znear : float, optional
        The near clipping plane of the camera.
    zfar : float, optional
        The far clipping plane of the camera.
    aspectRatio : float, optional
        The aspect ratio of the camera. [For perspective cameras]
    extras : dict, optional
        Extra data to be stored with the camera.
    extensions : dict, optional
        Extensions to be stored with the camera.
    Returns
    -------
    BCamera
        The camera object.
    '''
    match type:
        case CameraType.PERSPECTIVE:
            return _PerspectiveCamera(
                name,
                yfov=yfov,
                znear=znear,
                zfar=zfar,
                aspectRatio=aspectRatio,
                extras=extras,
                extensions=extensions
            )
        case CameraType.ORTHOGRAPHIC:
            return _OrthographicCamera(
                name,
                xmag=xmag,
                ymag=ymag,
                znear=znear,
                zfar=zfar,
                extras=extras,
                extensions=extensions
            )
        case _:  # pragma: no cover
            raise ValueError(f'Unknown camera type: {type}')