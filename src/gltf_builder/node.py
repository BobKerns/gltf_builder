'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from collections.abc import Iterable, Mapping
from typing import Optional, Any

import pygltflib as gltf

from gltf_builder.element import (
    Element, EMPTY_SET, Matrix4, Quaternion, Vector3,
    BNodeContainerProtocol, BNode, BuilderProtocol, BMesh
)
from gltf_builder.mesh import _Mesh 
from gltf_builder.holder import Holder


class BNodeContainer(BNodeContainerProtocol):
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
                 children: Iterable['_Node']=(),
                 _parent: Optional[BNodeContainerProtocol]=None,
                 **_
                ):
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
                extras: Mapping[str, Any]=EMPTY_SET,
                extensions: Mapping[str, Any]=EMPTY_SET,
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
                    extras: Mapping[str, Any]=EMPTY_SET,
                    extensions: Mapping[str, Any]=EMPTY_SET,
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
    def __init__(self,
                 name: str ='',
                 children: Iterable['_Node']=(),
                 mesh: Optional[_Mesh]=None,
                 root: Optional[bool]=None,
                 translation: Optional[Vector3]=None,
                 rotation: Optional[Quaternion]=None,
                 scale: Optional[Vector3]=None,
                 matrix: Optional[Matrix4]=None,
                 extras: Mapping[str, Any]=EMPTY_SET,
                 extensions: Mapping[str, Any]=EMPTY_SET,
                 _parent: Optional[BNodeContainerProtocol]=None,
                 ):
        Element.__init__(self, name, extras, extensions)
        BNodeContainer.__init__(self,
                                children=children,
                                _parent=_parent,
                            )
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
        return gltf.Node(
            name=self.name,
            mesh=self.mesh.index if self.mesh else None,
            children=[child.index for child in self.children],
            translation=self.translation,
            rotation=self.rotation,
            scale=self.scale,
            matrix=self.matrix,
        )
