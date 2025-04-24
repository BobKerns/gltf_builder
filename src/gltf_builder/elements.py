'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from pathlib import Path
from typing import (
    Generic, Protocol, Optional, Any, TypeVar, overload, runtime_checkable,
)
from abc import abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
import pygltflib as gltf

from gltf_builder.holders import _Holder
from gltf_builder.quaternions import Quaternion, QuaternionSpec
from gltf_builder.core_types import (
    AlphaMode, CameraType, ComponentType, ImageType, JsonObject, MagFilter, MinFilter, PrimitiveMode,
    BufferViewTarget, ElementType, NPTypes, ScopeName, WrapMode
)
from gltf_builder.attribute_types import (
    BTYPE, AttributeData, AttributeDataIterable, AttributeDataList,
    ColorSpec, Point, Scale, TangentSpec, UvSpec,
    Vector3, Vector3Spec, PointSpec, Vector4,
    vector3,
)
from gltf_builder.matrix import Matrix4
from gltf_builder.compile import (
    _Compileable, T,
    _Scope
)
from gltf_builder.protocols import _BNodeContainerProtocol
from gltf_builder.log import GLTF_LOG
from gltf_builder.vertices import Vertex


LOG = GLTF_LOG.getChild(Path(__file__).stem)
@runtime_checkable
class Element(_Compileable[T], Protocol):
    def __init__(self,
                 name: str='',
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                index: int=-1,
            ):
        super().__init__(
            name,
            extras=extras,
            extensions=extensions,
            index=index,
        )
    
    def __hash__(self):
        return id(self)
        
    def __eq__(self, other: Any):
        return self is other
    
    def _repr_additional(self) -> str:
        return ''
    
    def __repr__(self):
        typ = type(self).__name__.lstrip('_')
        idx = f'[{self._index}]' if self._index != -1 else ''
        name = self.name or id(self)
        more = self._repr_additional()
        if more:
            return f'<{typ} {name}{idx} {more}>'
        return f'<{typ} {name}{idx}>'
    
    def __str__(self):
        typ = type(self).__name__.lstrip('_')
        if self._index == -1:
            idx = ''
        else:
            idx=f'[{self._index}]'
        return f'{typ}-{self.name or "?"}{idx}'


class BBuffer(Element[gltf.Buffer], _Scope, Protocol):
    _scope_name = ScopeName.BUFFER
    @property
    @abstractmethod
    def blob(self) -> bytes:
        ...
    views: _Holder['BBufferView']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def bytearray(self) -> bytearray: ...

    @abstractmethod
    def create_view(self,
                  target: BufferViewTarget,
                  /, *,
                  name: str='',
                  byteStride: int=0,
                  extras:  Optional[JsonObject]=None,
                  extensions: Optional[JsonObject]=None,
                ) -> 'BBufferView':
        '''
        Create a `BBufferView` for this `BBuffer`.
        '''
        ...


class BBufferView(Element[gltf.BufferView], Protocol):
    _scope_name = ScopeName.BUFFER_VIEW
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int
    accessors: _Holder['BAccessor[NPTypes, AttributeData]']

    @property
    @abstractmethod
    def blob(self) -> bytes: ...

    @abstractmethod
    def memoryview(self, offset: int, size: int) -> memoryview: ...

    @abstractmethod
    def _add_accessor(self, acc: 'BAccessor[NPTypes, AttributeData]') -> None: ...

NP = TypeVar('NP', bound=NPTypes)
NUM = TypeVar('NUM', bound=float|int, covariant=True)

@runtime_checkable
class BAccessor(Element[gltf.Accessor], Protocol, Generic[NP, BTYPE]):
    _scope_name = ScopeName.ACCESSOR
    view: BBufferView
    data: list[BTYPE]
    __array: np.ndarray[tuple[int], np.dtype[NP]]|None = None
    @property
    def array(self) -> np.ndarray[tuple[int], np.dtype[NP]]:
        if self.__array is None:
            self.__array = np.array(self.data, dtype=self.dtype)
        return self.__array
    count: int
    elt_type: ElementType
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

    @abstractmethod
    def _add_data(self, data: Sequence[BTYPE]) -> None:
        ...
    '''
    Add a Sequence of data to the accessor.
    '''
    
    @abstractmethod
    def _add_data_item(self, data: BTYPE) -> None:
        ...

@runtime_checkable
class BPrimitive(_Compileable[gltf.Primitive], Protocol):
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
class BMesh(Element[gltf.Mesh], _Scope, Protocol):
    _scope_name = ScopeName.MESH
    primitives: list[BPrimitive]
    weights: list[float]

    @property
    @abstractmethod
    def detached(self) -> bool:
        '''
        A detached mesh is not added to the builder, but is returned
        to be used as the root of an instanceable object, or to be added
        to multiple nodes and thus to the builder later.
        '''
        ...


    @overload
    def add_primitive(self, primitive: BPrimitive, /, *,
                      extras: Optional[JsonObject]=None,
                      extensions: Optional[JsonObject]=None,) -> BPrimitive: ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *points: PointSpec,
                      NORMAL: Optional[Iterable[Vector3Spec]]=None,
                      TANGENT: Optional[Iterable[TangentSpec]]=None,
                      TEXCOORD_0: Optional[Iterable[UvSpec]]=None,
                      TEXCOORD_1: Optional[Iterable[UvSpec]]=None,
                      COLOR_0: Optional[Iterable[ColorSpec]]=None,
                      extras:  Optional[JsonObject]=None,
                      extensions:  Optional[JsonObject]=None,
                      **attribs: AttributeDataIterable,
                    ) -> BPrimitive:
        ...
    @overload
    def add_primitive(self, mode: PrimitiveMode, /,
                      *vertices: Vertex,
                      extras:  Optional[JsonObject]=None,
                      extensions:  Optional[JsonObject]=None,
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
                      extras:  Optional[JsonObject]=None,
                      extensions:  Optional[JsonObject]=None,
                      **attribs: AttributeDataIterable|None,
                    ) -> BPrimitive:
        ...


@runtime_checkable
class BCamera(Element[gltf.Camera], Protocol):
    '''
    Camera for glTF.
    '''
    _scope_name = ScopeName.CAMERA
    @property
    @abstractmethod
    def type(self) -> CameraType: ...

    type_extras: JsonObject
    type_extensions: JsonObject

    

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
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
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
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
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
class BNode(Element[gltf.Node], _BNodeContainerProtocol, _Scope, Protocol):
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

    @property
    @abstractmethod
    def detached(self) -> bool:
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instancable object.
        '''
        ...

    @abstractmethod
    def detach(self):
        '''
        Detath this node and its children from the builder.
        '''
        ...

    @abstractmethod
    def create_mesh(self,
                name: str='',
                /, *,
                primitives: Optional[Iterable['BPrimitive']]=None,
                weights: Optional[Iterable[float]]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                detached: bool=False,
            ) -> 'BMesh':
        '''
        Create a `BMesh` for this `BNode`, or if `detached` is `True`,
        just create a `BMesh` and return it for later use.
        '''
        ...


@runtime_checkable
class BImage(Element[gltf.Image], Protocol):
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
class BSampler(Element[gltf.Sampler], Protocol):
    '''
    Texture samplers for glTF.
    '''
    _scope_name = ScopeName.SAMPLER
    magFilter: Optional[MagFilter]
    minFilter: Optional[MinFilter]
    wrapS: Optional[WrapMode]
    wrapT: Optional[WrapMode]

@runtime_checkable
class BTexture(Element[gltf.Texture], Protocol):
    '''
    Texture for glTF.
    '''
    _scope_name = ScopeName.TEXTURE
    sampler: BSampler
    source: BImage

@runtime_checkable
class BMaterial(Element[gltf.Material], Protocol):
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

class BScene(Element[gltf.Scene], Protocol):
    '''
    Scene for glTF.
    '''
    _scope_name = ScopeName.SCENE
    nodes: list[BNode]

class BSkin(Element[gltf.Skin], Protocol):
    '''
    Skin for a glTF model.
    '''
    inverseBindMatrices: Optional[Matrix4]
    skeleton: BNode
    joints: list[BNode]

