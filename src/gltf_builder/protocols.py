'''
Protocol classes to avoid circular imports.
'''

from abc import abstractmethod
from typing import (
    NamedTuple, Protocol, TypeAlias,
    Optional, TYPE_CHECKING,
)
from collections.abc import Callable, Iterable
import math

from gltf_builder.holders import _Holder
from gltf_builder.core_types import (
    BufferViewTarget, ElementType, ComponentType,
    ExtensionsData, ExtrasData,
)
from gltf_builder.attribute_types import (
    AttributeData, Vector3Spec, BTYPE
)
from gltf_builder.matrix import Matrix4
from gltf_builder.quaternions import QuaternionSpec, Quaternion as Q
if TYPE_CHECKING:
    from gltf_builder.elements import(
        BNode, BMesh, BBuffer,
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
    @property
    @abstractmethod
    def parent(self) -> Optional['BNode']:
        '''
        Return the parent of this node.
        '''
        ...
    nodes: _Holder['BNode']
    descendants: dict[str, 'BNode']
    @property
    def children(self):
        return self.nodes

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
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
                ) -> 'BNode':
        ...

    @abstractmethod
    def instantiate(self, node_or_mesh: 'BNode|BMesh', /,
                    name: str='',
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Optional[ExtrasData]=None,
                    extensions: Optional[ExtensionsData]=None,
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
