'''
Builder representation of a gltr node. This will be compiled down during
the build phase.
'''


from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, cast

import pygltflib as gltf

from gltf_builder.compile import _Collected
from gltf_builder.core_types import JsonObject, Phase
from gltf_builder.attribute_types import Vector3Spec, vector3, scale as to_scale
from gltf_builder.matrix import Matrix4Spec, matrix as to_matrix
from gltf_builder.elements import (
    BBuffer, BBufferView, BCamera, Element, BNode, BMesh, BPrimitive,
    _Scope,
)
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

    @property
    @abstractmethod
    def builder(self) -> _BuilderProtocol: ...
    @builder.setter
    def builder(self, builder: _BuilderProtocol): ...
    
    def __init__(self, /,
                buffer: BBuffer|None,
                children: Iterable[BNode]=(),
            ):
        self.buffer = buffer
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
                detached: bool=False,
                ) -> '_Node':
        '''
        Add a node to the builder or as a child of another node.
        if _detached_ is True, the node will not be added to the builder,
        but will be returned to serve as the root of an instancable object.
        '''
        root = isinstance(self, _BuilderProtocol) and not detached
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
    __builder: Optional[_BuilderProtocol]
    @property
    def builder(self) -> _BuilderProtocol:
        if self.__builder is None:
            raise ValueError('Node is not attached to a builder')
        return self.__builder
    
    @builder.setter
    def builder(self, builder: _BuilderProtocol):
        if self.__builder == builder:
            return
        if self.__builder is not None:
            raise ValueError('Node is already attached to a builder')
        self.__builder = builder
        for c in self.children:
            n = cast(_BNodeContainer, c)
            n.builder = builder

    def detach(self):
        self.__detatched = True
        self.__builder = None
        def flatten(n: _Node):
            yield n
            for c in n.children:
                yield from flatten(cast(_Node, c))
        for n in flatten(self):
            n.__builder = None
    __detached: bool
    @property
    def detached(self) -> bool:
        '''
        A detached node is not added to the builder, but is returned
        to be used as the root of an instanceable object.
        '''
        return self.__detached
    
    def __init__(self,
                 name: str ='', /,
                 builder: Optional[_BuilderProtocol]=None,
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
                 buffer: Optional[BBuffer]=None,
                 index: int=-1,
                 detached: bool=False,
                 ):
        super(Element, self).__init__(
                         name,
                         extras=extras,
                         extensions=extensions,
                         index=index,
                        )
        if buffer is None and builder is not None:
            buffer = builder.buffer
        _BNodeContainer.__init__(self,
                                buffer=buffer,
                                children=children,
                            )
        self.__builder = builder
        self.__detached = detached
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
            detached=self.root or self.detached,
        )
        
    def _do_compile(self, builder: _BuilderProtocol, scope: _Scope, phase: Phase):
        match phase:
            case Phase.COLLECT:
                self.builder = builder
                self.builder.nodes.add(self)
                return [
                    c.compile(builder, scope, phase)
                    for c in (self.mesh, self.camera, *self.children)
                    if c is not None
                ]
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
                    self.mesh.compile(builder, scope, phase)
                for child in self.children:
                    child.compile(builder, scope, phase)

    def create_mesh(self,
                 name: str='',
                  /, *,
                 primitives: Optional[Iterable['BPrimitive']]=None,
                 weights: Optional[Iterable[float]]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                 detached: bool=False,
            ) -> 'BMesh':
        mesh = self.builder.create_mesh(name=name,
                                    primitives=primitives,
                                    weights=weights,
                                    extras=extras,
                                    extensions=extensions,
                                    detached=detached or self.detached,
                                )
        if detached:
            return mesh
        self.mesh = mesh
        return mesh
    
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
        mesh=mesh.clone() if mesh else None,
        camera=camera.clone() if camera else None,
        translation=translation,
        rotation=rotation,
        scale=scale,
        matrix=matrix,
        extras=extras,
        extensions=extensions,
        builder=None,  # type: ignore
        detached=True
    )