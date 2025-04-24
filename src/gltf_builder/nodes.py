'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, cast

import pygltflib as gltf

from gltf_builder.compile import _CompileStates
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.attribute_types import Vector3Spec, vector3, scale as to_scale
from gltf_builder.matrix import Matrix4Spec, matrix as to_matrix
from gltf_builder.elements import (
    BBuffer, BBufferView, BCamera, Element, BNode, BMesh, BPrimitive,
    _Scope,
)
from gltf_builder.meshes import mesh
from gltf_builder.quaternions import QuaternionSpec, quaternion
from gltf_builder.holders import _Holder
from gltf_builder.protocols import (
    _BNodeContainerProtocol, _BuilderProtocol,
)
from gltf_builder.utils import std_repr


class _BNodeContainer(_BNodeContainerProtocol):
    children: _Holder[BNode]
    @property
    def nodes(self):
        return self.children
    
    def __init__(self, /,
                children: Iterable[BNode]=(),
            ):
        self._local_views = {}
        self.children = _Holder(BNode, *children)
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
                ) -> 'BNode':
        '''
        Add a node to the builder or as a child of another node.

        Parameters
        ----------
        name : str
            Name of the node.
        children : Iterable[BNode]
            Children of the node.
        mesh : Optional[BMesh]
            Mesh of the node.
        translation : Optional[Vector3Spec]
            Translation of the node.
        rotation : Optional[QuaternionSpec]
            Rotation of the node.
        scale : Optional[Vector3Spec]
            Scale of the node.
        matrix : Optional[Matrix4Spec]
            Matrix of the node.
        extras : Optional[JsonObject]
            Extra data to be added to the node.
        extensions : Optional[JsonObject]
            Extensions to be added to the node.
        Returns
        -------
        BNode
        '''
        root = isinstance(self, _BuilderProtocol)
        node = _Node(name,
                    root=root,
                    children=children,
                    mesh=mesh,
                    translation=translation,
                    rotation=rotation,
                    scale=scale,
                    matrix=matrix,
                    extras=extras,
                    extensions=extensions,
                )
        self.children.add(node)
        if name:
            n = self
            while n is not None:
                if name not in n.descendants:
                    n.descendants[name] = node
                n = n._parent
        return node

    def instantiate(self, node_or_mesh: BNode|BMesh,
                    name: str='', /,
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
                mesh=node_or_mesh.clone(),
                translation=translation,
                rotation=rotation,
                scale=scale,
                matrix=matrix,
                extras=extras,
                extensions=extensions,
            )
        return self.create_node(
            name,
            children=[node_or_mesh.clone()],
            translation=translation,
            rotation=rotation,
            scale=scale,
            matrix=matrix,
            extras=extras,
            extensions=extensions,
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

class _Node(_BNodeContainer, BNode):
    '''
    Implementation class for `BNode`.
    '''    
    def __init__(self,
                 name: str ='', /,
                 children: Iterable[BNode]=(),
                 mesh: Optional[BMesh]=None,
                 camera: Optional[BCamera]=None,
                 root: Optional[bool]=None,
                 translation: Optional[Vector3Spec]=None,
                 rotation: Optional[QuaternionSpec]=None,
                 scale: Optional[Vector3Spec]=None,
                 matrix: Optional[Matrix4Spec]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 index: int=-1,
                 ):
        super(Element, self).__init__(
                         name,
                         extras=extras,
                         extensions=extensions,
                         index=index,
                        )
        _BNodeContainer.__init__(self,
                                children=children,
                            )
        self.root = root or False
        self.mesh = mesh
        self.camera = camera
        self.translation = vector3(translation) if translation else None
        self.rotation = quaternion(rotation) if rotation else None
        self.scale = to_scale(scale) if scale else None
        self.matrix = to_matrix(matrix) if matrix else None
        self._local_views = _Holder(BBufferView)

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            children=[c.clone() for c in self.children],
            mesh=self.mesh,
            camera=self.camera.clone() if self.camera else None,
            translation=self.translation,
            rotation=self.rotation,
            scale=self.scale,
            matrix=self.matrix,
            root=self.root,
        )
        
    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    states: _CompileStates,
                    /):
        match phase:
            case Phase.COLLECT:
                builder.nodes.add(self)
                return (
                    c.compile(builder, scope, phase, states)
                    for c in (self.mesh, self.camera, *self.children)
                    if c is not None
                )
            case Phase.SIZES:
                size = sum(
                    n.compile(builder, scope, phase, states)
                    for n in self.children
                )
                if self.mesh is not None:
                    size += self.mesh.compile(builder, scope, phase, states)
                return size
            case Phase.BUILD:
                if self.mesh is not None:
                    self.mesh.compile(builder, scope, phase, states)
                return gltf.Node(
                    name=self.name,
                    mesh=self.mesh._index if self.mesh else None,
                    children=[child._index for child in self.children],
                    translation=list(self.translation) if self.translation else None,
                    rotation=list(self.rotation) if self.rotation else None,
                    scale=list(self.scale) if self.scale else None,
                    matrix=list(float(v) for v in self.matrix) if self.matrix else None,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                if self.mesh is not None:
                    self.mesh.compile(builder, scope, phase, states)
                for child in self.children:
                    child.compile(builder, scope, phase, states)

    def create_mesh(self,
                 name: str='',
                  /, *,
                 primitives: Optional[Iterable['BPrimitive']]=None,
                 weights: Optional[Iterable[float]]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
            ) -> 'BMesh':
        m = mesh(name=name,
                    primitives=primitives,
                    weights=weights,
                    extras=extras,
                    extensions=extensions,
                )
        self.mesh = m
        return m
    
    def __repr__(self):
        return std_repr(self, (
            'name',
            'mesh',
            'camera',
            'translation',
            'rotation',
            'scale',
            'matrix',
        ))

def node(
    name: str='',
    children: Iterable[BNode]=(),
    mesh: Optional[BMesh]=None,
    camera: Optional[BCamera]=None,
    translation: Optional[Vector3Spec]=None,
    rotation: Optional[QuaternionSpec]=None,
    scale: Optional[Vector3Spec]=None,
    matrix: Optional[Matrix4Spec]=None,
    extras: Optional[JsonObject]=None,
    extensions: Optional[JsonObject]=None,
) -> _Node:
    '''
    Create a detached node with the given attributes.
    '''
    return _Node(
        name,
        children=[c.clone() for c in children],
        mesh=mesh.clone() if mesh is not None else None,
        camera=camera.clone() if camera else None,
        translation=translation,
        rotation=rotation,
        scale=scale,
        matrix=matrix,
        extras=extras,
        extensions=extensions,
    )