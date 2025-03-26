'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from collections.abc import Iterable, Mapping
from typing import Optional, Any

import pygltflib as gltf

from gltf_builder.core_types import Phase, EMPTY_MAP
from gltf_builder.attribute_types import (
    Matrix4, Vector3,
)
from gltf_builder.element import (
    Element, _Scope, BNode, BMesh, BPrimitive,
)
from gltf_builder.quaternion import Quaternion
from gltf_builder.mesh import _Mesh 
from gltf_builder.holder import Holder
from gltf_builder.protocols import (
    BNodeContainerProtocol, BuilderProtocol,
)


class BNodeContainer(BNodeContainerProtocol):
    builder: BuilderProtocol
    children: Holder['_Node']
    descendants: dict[str, '_Node']   
    @property
    def nodes(self):
        return self.children
    @nodes.setter
    def nodes(self, nodes: Holder['_Node']):
        self.children = nodes
    
    def __init__(self, /,
                builder: BuilderProtocol,
                children: Iterable['_Node']=(),
                **_
            ):
        self.builder = builder
        self.children = Holder(*children)
        for c in children:
            if c._parent is not None and c._parent is not self:
                raise ValueError(f'Node {c.name} already has a parent')
            c._parent = self
            c.root = False
        self.descendants = {}
    
    def create_node(self,
                name: str='',
                children: Iterable[BNode]=(),
                mesh: Optional[BMesh]=None,
                translation: Optional[Vector3]=None,
                rotation: Optional[Quaternion]=None,
                scale: Optional[Vector3]=None,
                matrix: Optional[Matrix4]=None,
                extras: Mapping[str, Any]=EMPTY_MAP,
                extensions: Mapping[str, Any]=EMPTY_MAP,
                detached: bool=False,
                **attrs: tuple[float|int,...]
                ) -> '_Node':
        '''
        Add a node to the builder or as a child of another node.
        if _detached_ is True, the node will not be added to the builder,
        but will be returned to serve as the root of an instancable object.
        '''
        root = isinstance(self, BuilderProtocol) and not detached
        node = _Node(name=name,
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
                    **attrs,
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
                    translation: Optional[Vector3]=None,
                    rotation: Optional[Quaternion]=None,
                    scale: Optional[Vector3]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Mapping[str, Any]=EMPTY_MAP,
                    extensions: Mapping[str, Any]=EMPTY_MAP,
                ) -> BNode:
        if isinstance(node_or_mesh, BMesh):
            return self.create_node(
                name=name,
                mesh=node_or_mesh,
                translation=translation,
                rotation=rotation,
                scale=scale,
                matrix=matrix,
                extras=extras,
                extensions=extensions,
            )
        def clone(node: BNode):
            return _Node(
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
            name=name,
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

class _Node(BNodeContainer, BNode):
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
                 children: Iterable['_Node']=(),
                 mesh: Optional[_Mesh]=None,
                 root: Optional[bool]=None,
                 translation: Optional[Vector3]=None,
                 rotation: Optional[Quaternion]=None,
                 scale: Optional[Vector3]=None,
                 matrix: Optional[Matrix4]=None,
                 extras: Mapping[str, Any]=EMPTY_MAP,
                 extensions: Mapping[str, Any]=EMPTY_MAP,
                 detached: bool=False,
                 ):
        Element.__init__(self, name, extras, extensions)
        BNodeContainer.__init__(self,
                                builder=builder,
                                children=children,
                            )
        self.__detached = detached
        self.root = root
        self.mesh = mesh
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.matrix = matrix
        
    def _do_compile(self, builder: BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                self.builder.nodes.add(self)
                if self.mesh:
                    self.mesh.compile(builder, scope, phase)
                    return [self.mesh]
                return []
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
                    translation=self.translation,
                    rotation=self.rotation,
                    scale=self.scale,
                    matrix=self.matrix,
                )
            case _:
                if self.mesh is not None:
                    self.mesh.compile(builder, scope, phase)
                for child in self.children:
                    child.compile(builder, scope, phase)

    def create_mesh(self,
                 name: str='',
                 primitives: Iterable['BPrimitive']=(),
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
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