'''
Protocol classes to avoid circular imports.
'''

from typing import (
    Protocol, runtime_checkable, Optional, TYPE_CHECKING, abstractmethod, Any
)
from collections.abc import Iterable, Mapping
import math

import pygltflib as gltf

from gltf_builder.holder import Holder
from gltf_builder.core_types import (
    ElementType, ComponentType, NameMode, EMPTY_MAP,
    Vector3, Matrix4
)
from gltf_builder.quaternion import Quaternion
import gltf_builder.quaternion as Q
if TYPE_CHECKING:
    from gltf_builder.element import(
        BNode, BMesh, BPrimitive, BBuffer, BBufferView, BAccessor,
    )

class BNodeContainerProtocol(Protocol):
    _parent: Optional['BNode'] = None
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
        from gltf_builder.element import BNode
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