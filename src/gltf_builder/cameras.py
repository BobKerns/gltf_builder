'''
Functions for creating cameras in glTF format.
'''

from typing import Any, Literal, overload

import pygltflib as gltf

from gltf_builder.core_types import CameraType, JsonObject, Phase
from gltf_builder.elements import BCamera, BOrthographicCamera, BPerspectiveCamera
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.utils import std_repr
from gltf_builder.compile import _CompileStates, _Scope


class _PerspectiveCanera(BPerspectiveCamera):
    '''
    Builder representation of a glTF Perspective Camera
    '''

    def __init__(self,
                 name: str='',
                 yfov: float=1.0,
                 znear: float=0.1,
                 zfar: float=100.0,
                 aspectRatio: float|None=None,
                 extras: JsonObject|None=None,
                 extensions: JsonObject|None=None,
                 type_extras: JsonObject|None=None,
                 type_extensions: JsonObject|None=None,
                 ):
        super().__init__(
            name,
            extras=extras,
            extensions=extensions
        )
        self.yfov=yfov
        self.znear=znear
        self.zfar=zfar
        self.aspectRatio=aspectRatio
        self.type_extras=dict(type_extras or ())
        self.type_extensions=dict(type_extensions or ())
                 
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    states: _CompileStates,
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
    

class _OrthographicCampera(BOrthographicCamera):
    '''
    Builder representation of a glTF Orthographic Camera
    '''

    def __init__(self,
                name: str='',
                xmag: float=1.0,
                ymag: float=1.0,
                znear: float=0.1,
                zfar: float=100.0,
                 aspectRatio: float|None=None,
                 extras: JsonObject|None=None,
                 extensions: JsonObject|None=None,
                 type_extras: JsonObject|None=None,
                 type_extensions: JsonObject|None=None,
                ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions
        )
        self.type_extras=dict(type_extras or ())
        self.type_extensions=dict(type_extensions or ())
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
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    states: _CompileStates,
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
    '''
    match type:
        case CameraType.PERSPECTIVE:
            return _PerspectiveCanera(
                name=name,
                yfov=yfov,
                znear=znear,
                zfar=zfar,
                aspectRatio=aspectRatio,
                extras=extras,
                extensions=extensions
            )
        case CameraType.ORTHOGRAPHIC:
            return _OrthographicCampera(
                name=name,
                xmag=xmag,
                ymag=ymag,
                znear=znear,
                zfar=zfar,
                extras=extras,
                extensions=extensions
            )
        case _:
            raise ValueError(f'Unknown camera type: {type}')