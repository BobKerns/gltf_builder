'''
Protocol classes to avoid circular imports.
'''

from abc import abstractmethod
from pathlib import Path
from typing import (
    Literal, NamedTuple, Protocol, TypeAlias, runtime_checkable,
    Optional, TYPE_CHECKING, Any
)
from collections.abc import Callable, Iterable
import math

import pygltflib as gltf

from gltf_builder.compiler import _GLTF, _Compileable, _Scope, _CompileState
from gltf_builder.holders import _Holder
from gltf_builder.core_types import (
    BufferViewTarget, ElementType, ComponentType, ImageType, JsonObject,
    NPTypes, NameMode, ScopeName,
)
from gltf_builder.attribute_types import (
    AttributeData, Vector3Spec, BTYPE
)
from gltf_builder.matrix import Matrix4
from gltf_builder.quaternions import QuaternionSpec, Quaternion as Q
if TYPE_CHECKING:
    from gltf_builder.elements import(
        BNode, BMesh, BPrimitive, BBuffer, BBufferView, BAccessor, BImage,
        BSampler, BTexture, BScene, BSkin, BMaterial, BCamera,
    )

class _BufferViewKey(NamedTuple):
    buffer: 'BBuffer'
    target: BufferViewTarget
    byteStride: int
    name: str


_AttributeParser: TypeAlias = Callable[..., BTYPE]
'''
Parse the given data into an attribute data item.
'''

class AttributeType(NamedTuple):
    name: str
    elementType: ElementType
    componentType: ComponentType
    type: type
    parser: _AttributeParser[AttributeData]|None = None

class _BNodeContainerProtocol(Protocol):
    _parent: Optional['BNode'] = None
    children: _Holder['BNode']
    descendants: dict[str, 'BNode']   
    @property
    def nodes(self):
        return self.children
    
    @abstractmethod
    def create_node(self,
                name: str='',
                /, *,
                children: Iterable['BNode']=(),
                mesh: Optional['BMesh']=None,
                translation: Optional[Vector3Spec]=None,
                rotation: Optional[QuaternionSpec]=None,
                scale: Optional[Vector3Spec]=None,
                matrix: Optional[Matrix4]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                ) -> 'BNode':
        ...

    @abstractmethod
    def instantiate(self, node_or_mesh: 'BNode|BMesh', /,
                    name: str='',
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject]=None,
                ) -> 'BNode':
        ...

    def print_hierarchy(self, indent:int=0):
        """Prints the node hierarchy in a readable format."""
        from gltf_builder.elements import BNode
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
    def __len__(self) -> int:
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
    def __iter__(self) -> Iterable['BNode']:
        ...

@runtime_checkable
class _BuilderProtocol(_BNodeContainerProtocol, _Scope, Protocol):
    '''
    Abstract class for a Builder.  This exists to avoid circular dependencies.
    '''
    asset: gltf.Asset
    '''
    The asset information for the glTF file.
    '''
    meshes: _Holder['BMesh']
    '''
    The meshes in the glTF file.
    '''
    cameras: _Holder['BCamera']
    '''
    The cameras in the glTF file.
    '''
    _buffers: _Holder['BBuffer']
    '''
    The buffers in the glTF file.'''
    _views: _Holder['BBufferView']
    '''
    The buffer views in the glTF file.
    '''
    _accessors: _Holder['BAccessor[NPTypes, AttributeData]']
    '''
    The accessors in the glTF file.
    '''
    images: _Holder['BImage']
    '''
    The images in the glTF file.
    '''
    materials: _Holder['BMaterial']
    '''
    The materials in the glTF file.
    '''
    samplers: _Holder['BSampler']
    '''
    The samplers in the glTF file.
    '''
    textures: _Holder['BTexture']
    '''
    The textures in the glTF file.
    '''
    scenes: _Holder['BScene']
    '''
    The scenes in the glTF file.
    '''
    skins: _Holder['BSkin']
    '''
    The skins in the glTF File
    '''
    extras: dict[str, Any]
    '''
    The extras for the glTF file.
    '''
    extensions: dict[str, Any]
    '''
    The extensions for the glTF file.
    '''
    scene: Optional['BScene']
    '''
    The initial scene.
    '''
    extensionsUsed: list[str]
    '''
    The extensions used in this file
    '''
    extensionsRequired: list[str]
    '''
    The extensions required to load this file.
    '''
    index_size: int = -1
    '''
    Number of bits to use for indices. Default is -, meaning no index.
    
    This is used to determine the component type of the indices.
    8 bits will use UNSIGNED_BYTE, 16 bits will use UNSIGNED_SHORT,
    and 32 bits will use UNSIGNED_INT.

    A value of 0 will use the smallest possible size for a particular
    mesh.

    A value of -1 will disaable indexed goemetry.

    This is only used when creating the indices buffer view.

    '''
    attr_type_map: dict[str, AttributeType]
    '''
    The mapping of attribute names to their types.
    '''
    name_policy: dict[ScopeName, NameMode]
    '''
    The mode for handling names, for each `ScopeName`

    AUTO: Automatically generate names for objects that do not have one.
    MANUAL: Use the name provided.
    UNIQUE: Ensure the name is unique.
    NONE: Do not use names.
    '''
    @property
    @abstractmethod
    def buffer(self) -> 'BBuffer': ...
    '''
    The primary `BBuffer` for the glTF file.
    '''
    _states: dict[int, _CompileState]
    
    @abstractmethod
    def create_mesh(self,
                 name: str='',
                 primitives: Optional[Iterable['BPrimitive']]=None,
                 weights: Optional[Iterable[float]]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
            ) -> 'BMesh':
        
        ...
    
    @abstractmethod
    def build(self) -> gltf.GLTF2:
        ...

    @abstractmethod
    def define_attrib(self, name: str,
                      elementType: ElementType,
                      componentType: ComponentType,
                      type: type[BTYPE],
                      parser: _AttributeParser[BTYPE]|None=None):
        ...

    @abstractmethod
    def get_attribute_type(self, name: str) -> AttributeType:
        ...

    @abstractmethod
    def _get_index_size(self, max_value: int) -> ComponentType|Literal[-1]:
        ...

    def _gen_name(self,
                  obj: _Compileable[_GLTF], /, *,
                  prefix: str='',
                  scope: ScopeName|None=None,
                  index: Optional[int]=None,
                  suffix: str=''
                  ) -> str:
        '''
        Generate a name for an object according to the current `NameMode` policy.

        PARAMETERS
        ----------
        obj: Element
            The object to generate a name for.
        gen_prefix: str|object
            The prefix to use for the generated name.
            If the prefix is an object, its `__class__.__name__` will be used.
        '''
        ...

    def _create_accessor(self,
                elementType: ElementType,
                componentType: ComponentType,
                btype: type[BTYPE],
                name: str='',
                normalized: bool=False,
                buffer: Optional['BBuffer']=None,
                count: int=0,
                target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> 'BAccessor[NPTypes, BTYPE]':
        '''
        Create a `BAccessor` for the given element type and component type.
        PARAMETERS
        ----------
        elementType: ElementType
            The element type for the accessor.
        componentType: ComponentType
            The component type for the accessor.
        btype: type[BTYPE]
            The type of the accessor data.
        name: str
            The name of the accessor.
        normalized: bool
            Whether the accessor data is normalized.
        target: BufferViewTarget
            The target for the buffer view.
        RETURNS
        -------
        BAccessor[NPTypes, BTYPE]
            The created accessor.
        '''
        ...

    def create_image(self,
                imageType: ImageType,
                name: str='',
                /, *,
                blob: Optional[bytes]=None,
                uri: Optional[str|Path]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
            ) -> 'BImage':
        '''
        Create a `BImage` for the given image type.
        PARAMETERS
        ----------
        imageType: ImageType
            The image type for the image.
        name: str
            The name of the image.
        blob: bytes
            The image data as a byte array.
        uri: str|Path
            The URI for the image.
        RETURNS
        -------
        BImage
            The created image.
        '''
        ...

    @abstractmethod
    def instantiate(self, node_or_mesh: 'BNode|BMesh', /,
                    name: str='',
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject]=None,
                ) -> 'BNode':
        '''
        Instantiate a node or mesh with the given parameters.
        PARAMETERS
        ----------
        node_or_mesh: BNode|BMesh
            The node or mesh to instantiate.
        name: str
            The name of the node.
        translation: Vector3Spec
            The translation of the node.
        rotation: QuaternionSpec
            The rotation of the node.
        scale: Vector3Spec
            The scale of the node.
        matrix: Matrix4
            The transformation matrix of the node.
        extras: JsonObject
            Extra data for the node.
        extensions: JsonObject
            Extensions for the node.
        RETURNS
        -------
        BNode
            The instantiated node.
        '''
        ...
