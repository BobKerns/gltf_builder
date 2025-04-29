'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from pathlib import Path
from typing import (
    Generic, Protocol, Optional, Any, TypeVar, overload, runtime_checkable,
    TYPE_CHECKING,
)
from abc import abstractmethod
from collections.abc import Iterable, Sequence

import pygltflib as gltf

from gltf_builder.quaternions import Quaternion
from gltf_builder.core_types import (
    AlphaMode, CameraType, ComponentType, ExtensionData,
    ExtensionsData, ExtrasData, ImageType,
    MagFilter, MinFilter, PrimitiveMode,
    BufferViewTarget, ElementType, NPTypes, ScopeName, WrapMode
)
from gltf_builder.attribute_types import (
    AttributeDataIterable, AttributeDataList, BTYPE_co,
    ColorSpec, Point, Scale, TangentSpec, UvSpec,
    Vector3, Vector3Spec, PointSpec,
    vector3,
)
from gltf_builder.matrix import Matrix4
from gltf_builder.compiler import (
    _STATE, _Compilable, _GLTF,
    _Scope
)
from gltf_builder.protocols import _BNodeContainerProtocol
from gltf_builder.log import GLTF_LOG
from gltf_builder.utils import std_repr
from gltf_builder.vertices import Vertex
if TYPE_CHECKING:
    from gltf_builder.accessors import _AccessorState  # noqa: F401
    #from gltf_builder.animations import _AnimationState
    from gltf_builder.assets import _AssetState  # noqa: F401
    from gltf_builder.buffers import _BufferState  # noqa: F401
    from gltf_builder.cameras import _CameraState  # noqa: F401
    from gltf_builder.extensions import ExtensionPlugin  # noqa: F401
    from gltf_builder.images import _ImageState  # noqa: F401
    from gltf_builder.materials import _MaterialState  # noqa: F401
    from gltf_builder.meshes import _MeshState  # noqa: F401
    from gltf_builder.primitives import _PrimitiveState  # noqa: F401
    from gltf_builder.scenes import _SceneState  # noqa: F401
    from gltf_builder.skins import _SkinState  # noqa: F401
    from gltf_builder.textures import _TextureState  # noqa: F401
    from gltf_builder.nodes import _NodeState  # noqa: F401
    from gltf_builder.samplers import _SamplerState  # noqa: F401
    from gltf_builder.views import _BufferViewState  # noqa: F401


LOG = GLTF_LOG.getChild(Path(__file__).stem)
@runtime_checkable
class Element(_Compilable[_GLTF, _STATE], Protocol[_GLTF, _STATE]):
    '''
    A fundamental element of a glTF model.
    '''

    def __init__(self,
                 name: str='',
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                index: int=-1,
            ):
        super().__init__(
            name,
            extras=extras,
            extensions=extensions,
            extension_objects=None,
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other: Any):
        return self is other

    def __repr__(self):
        return std_repr(self, (
            'name',
        ), id=id(self))

    def __str__(self):
        typ = type(self).__name__.lstrip('_')
        return f'{typ}-{self.name or "?"}'


class BBuffer(Element[gltf.Buffer, '_BufferState'], Protocol):
    '''
    Buffer interface.
    '''
    _scope_name = ScopeName.BUFFER


class BBufferView(Element[gltf.BufferView, '_BufferViewState'], Protocol):
    _scope_name = ScopeName.BUFFER_VIEW
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int

NP = TypeVar('NP', bound=NPTypes)
NUM = TypeVar('NUM', bound=float|int, covariant=True)

@runtime_checkable
class BAccessor(Element[gltf.Accessor, '_AccessorState'], Protocol, Generic[NP, BTYPE_co]):
    _scope_name = ScopeName.ACCESSOR
    count: int
    elementType: ElementType
    componentType: ComponentType
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    componentCount: int = 0
    '''The number of components per element.'''
    componentSize: int = 0
    '''The number of bytes per component.'''
    byteStride: int = 0
    '''The total number of bytes per element.'''
    dtype: type[NP]
    '''The numpy dtype for the data.'''
    bufferType: str = 'f'
    '''The buffer type char for `memoryview.cast()`.'''

@runtime_checkable
class BPrimitive(Element[gltf.Primitive, '_PrimitiveState'], Protocol):
    '''
    Base class for primitives
    '''
    _scope_name = ScopeName.PRIMITIVE
    mode: PrimitiveMode
    points: list[Point]
    attribs: dict[str, AttributeDataList]
    indices: Sequence[int]
    mesh: Optional['BMesh']


@runtime_checkable
class BMesh(Element[gltf.Mesh, '_MeshState'], _Scope, Protocol):
    _scope_name = ScopeName.MESH
    primitives: list[BPrimitive]
    weights: list[float]

    @overload
    def add_primitive(self, primitive: BPrimitive, /, *,
                      extras: Optional[ExtrasData]=None,
                      extensions: Optional[ExtensionsData]=None,) -> BPrimitive: ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras:  Optional[ExtrasData]=None,
                      extensions:  Optional[ExtensionsData]=None,
                      **attribs: AttributeDataIterable,
                    ) -> BPrimitive:
        ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *vertices: Vertex,
                      extras:  Optional[ExtrasData]=None,
                      extensions:  Optional[ExtensionsData]=None,
                    ) -> BPrimitive:
        ...
    @abstractmethod
    def add_primitive(self, mode: PrimitiveMode|BPrimitive, /,
                      *points: PointSpec|Vertex,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras:  Optional[ExtrasData]=None,
                      extensions:  Optional[ExtensionsData]=None,
                      **attribs: AttributeDataIterable|None,
                    ) -> BPrimitive:
        ...

@runtime_checkable
class BCamera(Element[gltf.Camera, '_CameraState'], Protocol):
    '''
    Camera for glTF.
    '''
    _scope_name = ScopeName.CAMERA
    @property
    @abstractmethod
    def type(self) -> CameraType: ...

    type_extras: ExtrasData
    type_extensions: ExtensionsData


@runtime_checkable
class BOrthographicCamera(BCamera, Protocol):
    '''
    Orthographic camera for glTF.
    '''
    xmag: float
    ymag: float
    zfar: float
    znear: float

    @property
    def type(self) -> CameraType:
        return CameraType.ORTHOGRAPHIC
    @property
    def orthographic(self) -> gltf.Orthographic:
        return gltf.Orthographic(
            xmag=self.xmag,
            ymag=self.ymag,
            zfar=self.zfar,
            znear=self.znear,
        )
    @property
    def perspective(self) -> Optional[gltf.Perspective]:
        return None

    def _init__(self,
                name: str='',
                /, *,
                xmag: float=1.0,
                ymag: float=1.0,
                znear: float=0.1,
                zfar: float=100.0,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
            ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions
        )
        self.xmag = xmag
        self.ymag = ymag
        self.znear = znear
        self.zfar = zfar


@runtime_checkable
class BPerspectiveCamera(BCamera, Protocol):
    '''
    Perspective camera for glTF.
    '''
    aspectRatio: Optional[float]
    yfov: float
    zfar: Optional[float]
    znear: float
    @property
    def type(self) -> CameraType:
        return CameraType.PERSPECTIVE
    @property
    def orthographic(self) -> Optional[gltf.Orthographic]:
        return None
    @property
    def perspective(self) -> gltf.Perspective:
        return gltf.Perspective(
            aspectRatio=self.aspectRatio,
            yfov=self.yfov,
            zfar=self.zfar,
            znear=self.znear,
        )

    def __init__(self,
                 name: str='',
                 /, *,
                 yfov: float=1.0,
                 znear: float=0.1,
                 zfar: float=100.0,
                 aspectRatio: float|None=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions
        )
        self.yfov = yfov
        self.znear = znear
        self.zfar = zfar
        self.aspectRatio = aspectRatio


@runtime_checkable
class BNode(Element[gltf.Node, '_NodeState'], _BNodeContainerProtocol, _Scope, Protocol):
    _scope_name = ScopeName.NODE
    mesh: BMesh|None
    root: bool
    __translation: Optional[Vector3]
    @property
    def translation(self) -> Optional[Vector3]:
        return self.__translation
    @translation.setter
    def translation(self, value: Vector3Spec|None):
        if value is not None:
            self.__translation = vector3(value)
        else:
            self.__translation = None
    rotation: Optional[Quaternion]
    scale: Optional[Scale]
    matrix: Optional[Matrix4]
    camera: Optional[BCamera]

    @abstractmethod
    def create_mesh(self,
                name: str='',
                /, *,
                primitives: Optional[Iterable['BPrimitive']]=None,
                weights: Optional[Iterable[float]]=None,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
            ) -> 'BMesh':
        '''
        Create a `BMesh` and add it to this `BNode`.

        Parameters
        ----------
        name : str, optional
            The name of the mesh.
        primitives : Optional[Iterable[BPrimitive]], optional
            The primitives of the mesh.
        weights : Optional[Iterable[float]], optional
            The weights of the mesh.
        extras : Optional[JsonObject], optional
            The extras of the mesh.
        extensions : Optional[JsonObject], optional
            The extensions of the mesh.
        Returns
        -------
        BMesh
            The created mesh.
        '''
        ...


@runtime_checkable
class BImage(Element[gltf.Image, '_ImageState'], Protocol):
    '''
    Image for glTF.
    '''
    _scope_name = ScopeName.IMAGE
    imageType: ImageType
    blob: Optional[bytes] = None
    uri: Optional[str|Path] = None
    view: Optional[BBufferView] = None

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

@runtime_checkable
class BSampler(Element[gltf.Sampler, '_SamplerState'], Protocol):
    '''
    Texture samplers for glTF.
    '''
    _scope_name = ScopeName.SAMPLER
    magFilter: Optional[MagFilter]
    minFilter: Optional[MinFilter]
    wrapS: Optional[WrapMode]
    wrapT: Optional[WrapMode]

@runtime_checkable
class BTexture(Element[gltf.Texture, '_TextureState'], Protocol):
    '''
    Texture for glTF.
    '''
    _scope_name = ScopeName.TEXTURE
    sampler: BSampler
    source: BImage

@runtime_checkable
class BMaterial(Element[gltf.Material, '_MaterialState'], Protocol):
    '''
    Material for glTF.
    '''
    _scope_name = ScopeName.MATERIAL
    baseColorFactor: Optional[tuple[float, float, float, float]]
    baseColorTexture: Optional[BTexture]
    metallicFactor: Optional[float]
    roughnessFactor: Optional[float]
    metallicRoughnessTexture: Optional[BTexture]
    normalTexture: Optional[BTexture]
    occlusionTexture: Optional[BTexture]
    emissiveFactor: Optional[tuple[float, float, float]]
    emissiveTexture: Optional[BTexture]
    alphaMode: AlphaMode
    alphaCutoff: Optional[float]
    doubleSided: bool

class BScene(Element[gltf.Scene, '_SceneState'], Protocol):
    '''
    Scene for glTF.
    '''
    _scope_name = ScopeName.SCENE
    nodes: list[BNode]

class BSkin(Element[gltf.Skin, '_SkinState'], Protocol):
    '''
    Skin for a glTF model.
    '''
    _scope_name = ScopeName.SKIN
    inverseBindMatrices: Optional[Matrix4]
    skeleton: BNode
    joints: list[BNode]

class BAsset(Element[gltf.Asset, '_AssetState'], Protocol):
    _scope_name = ScopeName.ASSET
    generator: Optional[str] = None
    version: str = '2.0'
    minVersion: Optional[str] = None

