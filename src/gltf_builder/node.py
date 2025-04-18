'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from collections.abc import Iterable
from typing import Optional, cast

import pygltflib as gltf

from gltf_builder.compile import Collected
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.attribute_types import Vector3Spec, vector3, scale as to_scale
from gltf_builder.matrix import Matrix4Spec, matrix as to_matrix
from gltf_builder.element import (
    BBuffer, BBufferView, Element, BNode, BMesh, BPrimitive,
    Scope_,
)
from gltf_builder.quaternions import QuaternionSpec, quaternion
from gltf_builder.holder import Holder_
from gltf_builder.protocols import (
    BNodeContainerProtocol, BuilderProtocol,
)


class BNodeContainer(BNodeContainerProtocol):
    children: Holder_[BNode]
    @property
    def nodes(self):
        return self.children
    @nodes.setter
    def nodes(self, nodes: Holder_['BNode']):
        self.children = nodes
    
    def __init__(self, /,
                builder: BuilderProtocol,
                buffer: BBuffer,
                children: Iterable[BNode]=(),
            ):
        self.builder = builder
        self.buffer = buffer
        self._local_views = {}
        self.children = Holder_(BNode, *children)
        for c in children:
            if isinstance(self, BNode):
                if c._parent is not None and c._parent is not self:
                    raise ValueError(f'Node {c.name} already has a parent')
                c._parent = self
            else:
                if c._parent is not None:
                    raise ValueError(f'Node {c.name} already has a parent')
                c._parent = None
            c.root = False
        self.descendants: dict[str, BNode] = {}
    
    def create_node(self,
                name: str='',
                /, *,
                children: Iterable[BNode]=(),
                mesh: Optional[BMesh]=None,
                translation: Optional[Vector3Spec]=None,
                rotation: Optional[QuaternionSpec]=None,
                scale: Optional[Vector3Spec]=None,
                matrix: Optional[Matrix4Spec]=None,
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                detached: bool=False,
                ) -> 'Node_':
        '''
        Add a node to the builder or as a child of another node.
        if _detached_ is True, the node will not be added to the builder,
        but will be returned to serve as the root of an instancable object.
        '''
        root = isinstance(self, BuilderProtocol) and not detached
        node = Node_(name=name,
                    root=root,
                    children=children,
                    mesh=mesh,
                    translation=translation,
                    rotation=rotation,
                    scale=scale,
                    matrix=matrix,
                    extras=extras,
                    extensions=extensions,
                    builder=self.builder,
                    detached=detached,
                )
        if not detached:
            self.children.add(node)
            if name:
                n = self
                while n is not None:
                    if name not in n.descendants:
                        n.descendants[name] = node
                    n = n._parent
        return node

    
    def instantiate(self, node_or_mesh: BNode|BMesh, /,
                    name: str='',
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4Spec]=None,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject]=None,
                ) -> BNode:
        if isinstance(node_or_mesh, BMesh):
            return self.create_node(
                name,
                mesh=node_or_mesh,
                translation=translation,
                rotation=rotation,
                scale=scale,
                matrix=matrix,
                extras=extras,
                extensions=extensions,
            )
        def clone(node: BNode) -> BNode:
            return Node_(
                name=node.name,
                children=[clone(child) for child in node.children],
                mesh=node.mesh,
                translation=node.translation,
                rotation=node.rotation,
                scale=node.scale,
                matrix=node.matrix,
                extras=node.extras,
                extensions=node.extensions,
                builder=self.builder,
            )
        return self.create_node(
            name,
            translation=translation,
            rotation=rotation,
            scale=scale,
            matrix=matrix,
            extras=extras,
            extensions=extensions,
            children=[clone(node_or_mesh)],
            detached=False,
        )

    def __getitem__(self, name: str) -> BNode:
        return self.descendants[name]
    
    def __setitem__(self, name: str, node: 'BNode'):
        self.descendants[name] = node

    def __contains__(self, name: str) -> bool:
        return name in self.descendants

    def __iter__(self):
        return iter(self.children)
    
    def __len__(self) -> int:
        return len(self.children)

class Node_(BNodeContainer, BNode):
    __detached: bool
    @property
    def detached(self) -> bool:
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instancable object.
        '''
        return self.__detached
    
    def __init__(self,
                 builder: BuilderProtocol,
                 name: str ='',
                 children: Iterable[BNode]=(),
                 mesh: Optional[BMesh]=None,
                 root: Optional[bool]=None,
                 translation: Optional[Vector3Spec]=None,
                 rotation: Optional[QuaternionSpec]=None,
                 scale: Optional[Vector3Spec]=None,
                 matrix: Optional[Matrix4Spec]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 buffer: Optional[BBuffer]=None,
                 index: int=-1,
                 detached: bool=False,
                 ):
        super(Element, self).__init__(
                         name=name,
                         extras=extras,
                         extensions=extensions,
                         index=index,
                        )
        BNodeContainer.__init__(self,
                                buffer=buffer or builder.buffer,
                                builder=builder,
                                children=children,
                            )
        self.__detached = detached
        self.root = root or False
        self.mesh = mesh
        self.translation = vector3(translation) if translation else None
        self.rotation = quaternion(rotation) if rotation else None
        self.scale = to_scale(scale) if scale else None
        self.matrix = to_matrix(matrix) if matrix else None
        self._local_views = Holder_(BBufferView)
        
    def _do_compile(self, builder: BuilderProtocol, scope: Scope_, phase: Phase):
        match phase:
            case Phase.COLLECT:
                self.builder.nodes.add(self)
                if self.mesh:
                    return [self.mesh.compile(builder, scope, phase)]
                return cast(list[Collected], [])
            case Phase.SIZES:
                size = sum(
                    n.compile(builder, scope, phase)
                    for n in self.children
                )
                if self.mesh is not None:
                    size += self.mesh.compile(builder, scope, phase)
                return size
            case Phase.BUILD:
                if self.mesh is not None:
                    self.mesh.compile(builder, scope, phase)
                return gltf.Node(
                    name=self.name,
                    mesh=self.mesh.index if self.mesh else None,
                    children=[child.index for child in self.children],
                    translation=list(self.translation) if self.translation else None,
                    rotation=list(self.rotation) if self.rotation else None,
                    scale=list(self.scale) if self.scale else None,
                    matrix=list(float(v) for v in self.matrix) if self.matrix else None,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                if self.mesh is not None:
                    self.mesh.compile(builder, scope, phase)
                for child in self.children:
                    child.compile(builder, scope, phase)

    def create_mesh(self,
                 name: str='',
                  /, *,
                 primitives: Iterable['BPrimitive']=(),
                 weights: Iterable[float]|None=(),
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 detached: bool=False,
            ) -> 'BMesh':
        mesh = self.builder.create_mesh(name=name,
                                    primitives=primitives,
                                    extras=extras,
                                    extensions=extensions,
                                    detached=detached or self.detached,
                                )
        if detached:
            return mesh
        self.mesh = mesh
        return mesh