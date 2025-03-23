'''
Base class for objects which will be referred to by their index
in the glTF. This also holds the name, defaulting it by the index.
'''

from pathlib import Path
from types import MappingProxyType
from typing import (
    NamedTuple, TypeAlias, Protocol, TypeVar, Generic, Optional, Any, overload, 
    runtime_checkable,
)
from abc import abstractmethod
from enum import IntEnum, StrEnum
from collections.abc import Mapping, Iterable, Sequence
import math
from logging import getLogger

import numpy as np
import pygltflib as gltf

from gltf_builder.holder import Holder
from gltf_builder.quaternion import Quaternion
import gltf_builder.quaternion as Q
    
GLTF_LOG = getLogger('gltf_builder')
LOG = GLTF_LOG.getChild(Path(__file__).stem)
LOG.setLevel('DEBUG')

class Phase(StrEnum):
    '''
    Enum for the phases of the compile process. Not all are implemented.
    '''
    PRIMITIVES = 'primitives'
    '''
    Process the data for the primitives for the glTF file.
    '''
    COLLECT = 'collect'
    '''
    Create the accessors and views for the glTF file, and collect all 
    subordinate objects.
    '''
    ENUMERATE = 'enumerate'
    '''
    Assign index values to each object
    '''
    VERTICES = 'vertices'
    '''
    Optimize the vertices for the glTF file.
    '''
    SIZES = 'sizes'
    '''
    Calculate sizes for the accessors and views for the glTF file.
    '''
    OFFSETS = 'offsets'
    '''
    Calculate offsets for the accessors and views for the glTF file.
    '''
    BUFFERS = 'buffers'
    '''
    Initialize buffers to receive data
    '''
    VIEWS = 'views'
    '''
    Initialize buffer views to receive data
    '''
    BUILD = 'build'
    '''
    Construct the binary data for the glTF file.
    '''
    

class PrimitiveMode(IntEnum):
    POINTS = gltf.POINTS
    LINES = gltf.LINES
    LINE_LOOP = gltf.LINE_LOOP
    LINE_STRIP = gltf.LINE_STRIP
    TRIANGLES = gltf.TRIANGLES
    TRIANGLE_STRIP = gltf.TRIANGLE_STRIP
    TRIANGLE_FAN = gltf.TRIANGLE_FAN
    
class BufferViewTarget(IntEnum):
    ARRAY_BUFFER = gltf.ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = gltf.ELEMENT_ARRAY_BUFFER
    
class ElementType(StrEnum):
    SCALAR = "SCALAR"
    VEC2 = "VEC2"
    VEC3 = "VEC3"
    VEC4 = "VEC4"
    MAT2 = "MAT2"
    MAT3 = "MAT3"
    MAT4 = "MAT4"
    
class ComponentType(IntEnum):
    BYTE = gltf.BYTE
    UNSIGNED_BYTE = gltf.UNSIGNED_BYTE
    SHORT = gltf.SHORT
    UNSIGNED_SHORT = gltf.UNSIGNED_SHORT
    UNSIGNED_INT = gltf.UNSIGNED_INT
    FLOAT = gltf.FLOAT

Vector2: TypeAlias = tuple[float, float]
Vector3: TypeAlias = tuple[float, float, float]
Vector4: TypeAlias = tuple[float, float, float, float]
Matrix2: TypeAlias = tuple[
    float, float,
    float, float,
]
Matrix3: TypeAlias = tuple[
    float, float, float,
    float, float, float,
    float, float, float
]
Matrix4: TypeAlias = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]

Scalar: TypeAlias = float
Point: TypeAlias = Vector3
Tangent: TypeAlias = Vector4
Normal: TypeAlias = Vector3


EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})


class NameMode(StrEnum):
    '''
    Enum for how to handle or generate names for objects.
    '''
    
    AUTO = 'auto'
    '''
    Automatically generate names for objects which do not have one.
    '''
    MANUAL = 'manual'
    '''
    Use the name provided.
    '''
    UNIQUE = 'unique'
    '''
    Ensure the name is unique.
    '''
    NONE = 'none'
    '''
    Do not use names.
    '''


class BNodeContainerProtocol(Protocol):
    children: Holder['BNode']
    @property
    def nodes(self):
        return self.children
    @nodes.setter
    def nodes(self, nodes: Holder['BNode']):
        self.children = nodes
    
    @abstractmethod
    def create_node(self,
                name: str='',
                children: Iterable['BNode']=(),
                mesh: Optional['BMesh']=None,
                translation: Optional[Vector3]=None,
                rotation: Optional[Quaternion]=None,
                scale: Optional[Vector3]=None,
                matrix: Optional[Matrix4]=None,
                extras: Mapping[str, Any]=EMPTY_MAP,
                extensions: Mapping[str, Any]=EMPTY_MAP,
                detached: bool=False,
                **attrs: tuple[float|int,...]
                ) -> 'BNode':
        ...
    
    @abstractmethod
    def instantiate(self, node_or_mesh: 'BNode|BMesh', /,
                    name: str='',
                    translation: Optional[Vector3]=None,
                    rotation: Optional[Quaternion]=None,
                    scale: Optional[Vector3]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Mapping[str, Any]=EMPTY_MAP,
                    extensions: Mapping[str, Any]=EMPTY_MAP,
                ) -> 'BNode':
        ...

    def print_hierarchy(self, indent=0):
        """Prints the node hierarchy in a readable format."""
        pre = '| ' * indent
        print(f'{pre}Node: {self}')
        if isinstance(self, BNode):
            if self.mesh:
                print(f'{pre}⇒ Mesh: {self.mesh}')
            if self.scale:
                x, y, z = self.scale
                if x == y and y == z:
                    print(f'{pre}⇒ Scale: {x:.2f}')
                else:
                    print(f'{pre}⇒ Scale: <{x:.2f}, {y:.2f}, {z:.2f}>')
            if self.rotation:
                (x, y, z), r = Q.to_axis_angle(self.rotation)
                print(f'{pre}⇒ Rotation: <{x:.2f}, {y:.2f}, {z:.2f}> @ {r*180/math.pi:.2f}°')
            if self.translation:  
                x, y, z = self.translation  
                print(f'{pre}⇒ Translation: <{x:.2f}, {y:.2f}, {z:.2f}>') 
            if self.matrix:
                print(f'{pre}⇒ Matrix: {self.matrix}')
        for child in self.children:
            child.print_hierarchy(indent + 1)

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> 'BNode':
        ...

    @abstractmethod
    def __setitem__(self, name: str, node: 'BNode'):
        ...

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        ...

    @abstractmethod
    def __iter__(self):
        ...

@runtime_checkable
class BuilderProtocol(BNodeContainerProtocol, Protocol):
    asset: gltf.Asset
    '''
    The asset information for the glTF file.
    '''
    meshes: Holder['BMesh']
    '''
    The meshes in the glTF file.
    '''
    _buffers: Holder['BBuffer']
    '''
    The buffers in the glTF file.'''
    _views: Holder['BBufferView']
    '''
    The buffer views in the glTF file.
    '''
    _accessors: Holder['BAccessor']
    '''
    The accessors in the glTF file.
    '''
    extras: dict[str, Any]
    '''
    The extras for the glTF file.
    '''
    extensions: dict[str, Any]
    '''
    The extensions for the glTF file.
    '''
    index_size: int = 32
    '''
    Number of bits to use for indices. Default is 32.
    
    This is used to determine the component type of the indices.
    8 bits will use UNSIGNED_BYTE, 16 bits will use UNSIGNED_SHORT,
    and 32 bits will use UNSIGNED_INT.

    A value of 0 will use the smallest possible size for a particular
    mesh.

    A value of -1 will disaable indexed goemetry.

    This is only used when creating the indices buffer view.

    '''
    attr_type_map: dict[str, tuple[ElementType, ComponentType]]
    '''
    The mapping of attribute names to their types.
    '''
    name_mode: NameMode
    '''
    The mode for handling names.

    AUTO: Automatically generate names for objects that do not have one.
    MANUAL: Use the name provided.
    UNIQUE: Ensure the name is unique.
    NONE: Do not use names.
    '''
    
    @abstractmethod
    def create_mesh(self,
                 name: str='',
                 primitives: Iterable['BPrimitive']=(),
                 detatched: bool=False,
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
                 detached: bool=False,
            ) -> 'BMesh':
        
        ...
    
    @abstractmethod
    def _add_buffer(self,
                   name: str='') -> 'BBuffer':
        ...
    
    @abstractmethod
    def _add_view(self,
                 name: str='',
                 buffer: Optional['BBuffer']=None,
                 target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
            ) -> 'BBufferView':
        ...
    
    @abstractmethod
    def build(self) -> gltf.GLTF2:
        ...

    @abstractmethod
    def define_attrib(self, name: str, type: ElementType, componentType: ComponentType):
        ...

    @abstractmethod
    def get_attrib_info(self, name: str) -> tuple[ElementType, ComponentType]:
        ...

    @abstractmethod
    def _get_index_size(self, max_value: int) -> int:
        ...

    def _gen_name(self, name: str|None, gen_prefix: str|object) -> str|None:
        '''
        Generate a name for an object according to the current `NameMode` policy.

        PARAMETERS
        ----------
        object: Element
            The object to generate a name for.
        '''
        ...

Collected: TypeAlias = tuple['Compileable', list['Collected']]

T = TypeVar('T')
class Compileable(Generic[T], Protocol):
    __phases: list[Phase]
    __compiled: T|None = None
    _len: int = -1
    __byte_offset: int = -1
    @property
    def byteOffset(self) -> int:
        return self.__byte_offset
    @byteOffset.setter
    def byteOffset(self, offset: int):
        if self.__byte_offset == -1:
            self.__byte_offset = offset
        elif self.__byte_offset != offset:
            raise ValueError(f'Byte offset already set, old={self.__byte_offset}, new={offset}')
    
    def _relocate(self, new_offset: int):
        '''
        Relocate an item to a new byte offset.
        '''
        self.__byte_offset = new_offset
    
    extensions: dict[str, Any]
    extras: dict[str, Any]
    name: str = ''
    index: int = -1

    def __init__(self,
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                ):
        self.__phases = []
        self.__collected: Collected
        self.extras = extras
        self.extensions = extensions
    
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.COLLECT
                ) -> list['Collected']: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.SIZES) -> int|None: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.OFFSETS) -> int: ...
    @overload
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.BUILD) -> T: ...
    def compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> T|int|None:
        if phase in self.__phases:
            match phase:
                case Phase.COLLECT:
                    return self.__collected
                case Phase.SIZES:
                    if hasattr(self, '_len'):
                        return self._len
                    return 0
                case Phase.OFFSETS:
                    if hasattr(self, 'byteOffset'):
                        return self.byteOffset
                    return -1
                case Phase.BUILD:
                    return self.__compiled
        else:
            #print(f'Compiling {self} in phase {phase}')
            self.__phases.append(phase)
            match phase:
                case Phase.COLLECT:
                    self.name = builder._gen_name(self)
                    self.__collected = self, self._do_compile(builder, scope, phase) or []
                    return self.__collected
                case Phase.SIZES:
                    bytelen = self._do_compile(builder, scope, phase)
                    if bytelen is not None:
                        self._len = bytelen
                        print(f'{self} has length {self._len}')
                        return bytelen
                    return 0
                case Phase.OFFSETS:
                    self._do_compile(builder, scope, phase)
                    if self.byteOffset >= 0:
                        print(f'{self} has offset {self.byteOffset}')
                        return self.byteOffset + self._len
                    return -1
                case Phase.BUILD:
                    if self.__compiled is None:
                        self.__compiled = self._do_compile(builder, scope, phase)
                    return self.__compiled
                case _:
                    return self._do_compile(builder, scope, phase)

    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.COLLECT
                    ) -> list['Compileable']: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.SIZES) -> int: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase.BUILD) -> T: ...
    @overload
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> None: ...
    @abstractmethod
    def _do_compile(self, builder: BuilderProtocol, scope: '_Scope', phase: Phase) -> T|int|None:
        ...

    def __len__(self) -> int:
        return self._len
    
    def __bool__(self) -> bool:
        if self._len < 0:
            return False
        return bool(self._len)

class AccessorKey(NamedTuple):
    type: ElementType
    componentType: ComponentType
    normalized: bool
    name: str = ''
    
    def __hash__(self):
        return hash((self.type, self.componentType, self.normalized, self.name))
    
    def __eq__(self, other):
        return (self.type is other.type
                and self.componentType == other.componentType
                and self.elementType == other.elementType
                and self.name == other.name)


class _Scope(Protocol):
    __parent: Optional['_Scope'] = None
    __views: dict[BufferViewTarget, 'BBufferView']
    __accessors: dict['BAccessor']
    __is_accessor_scope: bool = False
    __is_view_scope: bool = False
    '''
    Scope for allocating `BBufferView` and `BAccessor` objects.

    Scopes include meshes, nodes, and buffers, with meshes at the top
    and buffersat the bottom. Between, the nodes have their own hierarchy,
    so a parent node can have a view scope, but a child node can have an
    accessor scope. This allows control over sharing of verticies, accessors,
    and views. Vertex sharing follows the same scope as accessors, while
    views can follow the same scope as the accessors they contain, or any
    scope above them.
    '''
    def __init__(self,
                parent: Optional['_Scope']=None,
                is_accessor_scope: bool=False,
                is_view_scope: bool=False,
            ):
        self.__parent = parent
        self.__views = {}
        self.__accessors = Holder()
        self.__is_accessor_scope = is_accessor_scope
        self.__is_view_scope = is_view_scope

    def _get_accessor(self, 
                    eltType: ElementType,
                    componenType: ComponentType,
                    name: str = '',
                    normalized: bool=False,
                    BufferViewTarget: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> 'BAccessor':
        key = AccessorKey(eltType, componenType, normalized, name)
        accessor = self.__accessors.get(key, None)
        if accessor is None:
            view = self._get_view(BufferViewTarget)
            accessor = BAccessor(view, name,)
            accessor = view.add_accessor(eltType, componenType, normalized=normalized)



    def _get_view(self, target: BufferViewTarget, for_object: Compileable) -> 'BBufferView':
        ...
    

class Element(Compileable[T], Protocol):
    __index: int = -1 # -1 means not set
    @property
    def index(self) -> int:
        return self.__index
    @index.setter
    def index(self, index: int):
        if self.__index != -1 and self.__index != index:
            raise ValueError(f'Index already set old={self.__index}, new={index}')
        self.__index = index
    
    def __init__(self,
                 name: str='',
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
            ):
        super().__init__(extras, extensions)
        self.name = name
        self.extensions = dict(extras) if extras else None
        self.extras = dict(extensions) if extensions else None

    def __index__(self):
        return self.__index
    
    def __hash__(self):
        return id(self)
        
    def __eq__(self, other):
        return self is other
    
    def __repr__(self):
        return f'<{type(self).__name__} {self!s}>'
    
    def __str__(self):
        if self.name == '':
            if self.index == -1:
                return f'{type(self).__name__}-?'
            else:
                return f'{type(self).__name__}-{self.index}'
        else:
            return f'{type(self).__name__}-{self.name}'


class BBuffer(Element[gltf.Buffer], _Scope, Protocol):
    @property
    @abstractmethod
    def blob(self) -> bytes:
        ...
    views: Holder['BBufferView']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def _get_view(self,
                target: BufferViewTarget,
                name: str='',
                normalized: bool=False,
                byteStride: Optional[int]=None,
            ) -> 'BBufferView':
        ...


class BBufferView(Element[gltf.BufferView], Protocol):
    buffer: BBuffer
    target: BufferViewTarget
    byteStride: int
    accessors: list['BAccessor']

    @property
    @abstractmethod
    def blob(self):
        ...

    @abstractmethod
    def _memory(self, offset: int, size: int) -> memoryview:
        ...


AttributeDataItem: TypeAlias = int|float|tuple[int, ...]|tuple[float, ...]|np.ndarray[tuple[int, ...], Any] 

AttributeDataList: TypeAlias = (
    list[int]
    |list[float]
    |list[tuple[int, ...]]
    |list[tuple[float, ...]]
    |list[np.ndarray[tuple[int, ...], Any]]
)
'''
List of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''

AttributeDataSequence: TypeAlias = (
    Sequence[int]
    |Sequence[float]
    |Sequence[tuple[int, ...]]
    |Sequence[tuple[float, ...]]
    |Sequence[np.ndarray[tuple[int, ...], Any]]
)
'''
Sequence of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''


class BAccessor(Element[gltf.Accessor], Protocol):
    _view: BBufferView
    data: AttributeDataList
    count: int
    type: ElementType
    byteOffset: int
    componentType: int
    normalized: bool
    max: Optional[list[float]]
    min: Optional[list[float]]
    componentCount: int = 0
    '''The number of components per element.'''
    componentSize: int = 0
    '''The number of bytes per component.'''
    byteStride: int = 0
    '''The total number of bytes per element.'''
    dtype: np.dtype = np.float32
    '''The numpy dtype for the data.'''
    bufferType: str = 'f'
    '''The buffer type char for `memoryview.cast()`.'''

    @abstractmethod
    def _add_data(self, data: AttributeDataSequence) -> None:
        ...
    '''
    Add a Sequence of data to the accessor.
    '''
    
    @abstractmethod
    def _add_data_item(self, data: AttributeDataItem) -> None:
        ...
        
class BPrimitive(Compileable[gltf.Primitive], Protocol):
    '''
    Base class for primitives
    '''
    mode: PrimitiveMode
    points: list[Point]
    attribs: dict[str, list[tuple[int|float,...]]]
    indicies: list[int]
    mesh: Optional['BMesh']
    

@runtime_checkable
class BMesh(Element[gltf.Mesh], _Scope, Protocol):
    primitives: list[BPrimitive]
    weights: list[float]

    @property
    @abstractmethod
    def detached(self):
        '''
        A detached mesh is not added to the builder, but is returned
        to be used as the root of an instancable object, or to be added
        to multiple nodes and thus to the builder later.
        '''
        ...


    @abstractmethod
    def add_primitive(self, mode: PrimitiveMode,
                      *points: Point,
                      NORMAL: Optional[Iterable[Vector3]]=None,
                      TANGENT: Optional[Iterable[Vector4]]=None,
                      TEXCOORD_0: Optional[Iterable[Vector3]]=None,
                      TEXCOORD_1: Optional[Iterable[Vector3]]=None,
                      COLOR_0: Optional[Iterable[Vector4]]=None,
                      JOINTS_0: Optional[Iterable[Vector4]]=None,
                      WEIGHTS_0: Optional[Iterable[Vector4]]=None,
                      extras: Mapping[str, Any]|None=EMPTY_MAP,
                      extensions: Mapping[str, Any]|None=EMPTY_MAP,
                      **attribs: Iterable[tuple[int|float,...]]
                    ) -> BPrimitive:
        ...

    __views: Holder[BBufferView] 

    def _get_view(self, target: BufferViewTarget, for_object: Compileable) -> BBufferView:
        ...
    

@runtime_checkable
class BNode(Element[gltf.Node], _Scope, Protocol):
    mesh: BMesh
    root: bool
    translation: Optional[Vector3]
    rotation: Optional[Quaternion]
    scale: Optional[Vector3]
    matrix: Optional[Matrix4]

    @property
    @abstractmethod
    def detached(self):
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instancable object.
        '''
        ...

    @abstractmethod
    def create_mesh(self,
                name: str='',
                primitives: Iterable['BPrimitive']=(),
                weights: Iterable[float]|None=(),
                extras: Mapping[str, Any]|None=EMPTY_MAP,
                extensions: Mapping[str, Any]|None=EMPTY_MAP,
                detached: bool=False,
            ) -> 'BMesh':
        '''
        Create a `BMesh` for this `BNode`, or if `detached` is `True`,
        just create a `BMesh` and return it for later use.
        '''
        ...