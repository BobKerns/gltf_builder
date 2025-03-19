'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from collections.abc import Iterable, Mapping
from typing import Optional, Any

import pygltflib as gltf

from gltf_builder.element import (
    Element, EMPTY_MAP, Matrix4, Vector3,
    BNodeContainerProtocol, BuilderProtocol,
    BNode, BMesh, BPrimitive,
)
from gltf_builder.quaternion import Quaternion
from gltf_builder.mesh import _Mesh 
from gltf_builder.holder import Holder


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

    _parent: Optional[BNodeContainerProtocol]
    
    def __init__(self, /,
                builder: BuilderProtocol,
                children: Iterable['_Node']=(),
                _parent: Optional[BNodeContainerProtocol]=None,
                **_
            ):
        self.builder = builder
        self.children = Holder(*children)
        self._parent = _parent
        self.descendants = {}
    
    def add_node(self,
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
                    _parent=self,
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

    
    def instantiate(self, node: BNode, /,
                    name: str='',
                    translation: Optional[Vector3]=None,
                    rotation: Optional[Quaternion]=None,
                    scale: Optional[Vector3]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Mapping[str, Any]=EMPTY_MAP,
                    extensions: Mapping[str, Any]=EMPTY_MAP,
                ) -> BNode:
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
        return self.add_node(
            name=name,
            translation=translation,
            rotation=rotation,
            scale=scale,
            matrix=matrix,
            extras=extras,
            extensions=extensions,
            children=[clone(node)],
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
    detached: bool
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
                 _parent: Optional[BNodeContainerProtocol]=None,
                 ):
        Element.__init__(self, name, extras, extensions)
        BNodeContainer.__init__(self,
                                builder=builder,
                                children=children,
                                _parent=_parent,
                            )
        self.detached = detached
        self.root = root
        self.mesh = mesh
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.matrix = matrix
        
    def do_compile(self, builder: BuilderProtocol):
        if self.mesh:
            builder.meshes.add(self.mesh)
            self.mesh.compile(builder)
        for child in self.children:
            child.compile(builder)
        self.builder.nodes.add(self)
        return gltf.Node(
            name=self.name,
            mesh=self.mesh.index if self.mesh else None,
            children=[child.index for child in self.children],
            translation=self.translation,
            rotation=self.rotation,
            scale=self.scale,
            matrix=self.matrix,
        )

    def add_mesh(self,
                 name: str='',
                 primitives: Iterable['BPrimitive']=(),
                 extras: Mapping[str, Any]|None=EMPTY_MAP,
                 extensions: Mapping[str, Any]|None=EMPTY_MAP,
                 detached: bool=False,
            ) -> 'BMesh':
        return self.builder.add_mesh(name=name,
                                    primitives=primitives,
                                    extras=extras,
                                    extensions=extensions,
                                    detached=detached or self.detached,
                                )