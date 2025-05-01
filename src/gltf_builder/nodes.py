'''
Builder representation of a gltf node. This will be compiled down during
the build phase.
'''


from collections.abc import Iterable
from typing import Any, Optional, TYPE_CHECKING, cast

import pygltflib as gltf

import gltf_builder.builder as builder
from gltf_builder.compiler import _GLTF, _STATE, _CompileState
from gltf_builder.core_types import ExtensionsData, ExtrasData, Phase
from gltf_builder.attribute_types import (
    Vector3Spec, vector3, scale as to_scale,
)
from gltf_builder.matrix import Matrix4Spec, matrix as to_matrix
from gltf_builder.elements import (
    BBufferView, BCamera, Element, BNode, BMesh, BPrimitive,
)
from gltf_builder.meshes import mesh
from gltf_builder.quaternions import QuaternionSpec, quaternion
from gltf_builder.holders import _Holder
from gltf_builder.protocols import (
    _BNodeContainerProtocol, _GlobalBinary,
)
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


class _BNodeContainer(_BNodeContainerProtocol):
    _parent: Optional[BNode] = None
    @property
    def parent(self) -> Optional[BNode]:
        '''
        Return the parent of this node.
        '''
        return self._parent
    nodes: _Holder[BNode]
    @property
    def children(self):
        return self.nodes

    def __init__(self, /,
                parent: Optional[BNode]=None,
                nodes: Iterable[BNode]=(),
            ):
        self._parent = parent
        self._local_views = {}
        self.nodes = _Holder(BNode, *nodes)
        self.descendants: dict[str, BNode] = {}

    def create_node(self,
                name: str='',
                /, *,
                children: Iterable[BNode]=(),
                mesh: Optional[BMesh]=None,
                camera: Optional[BCamera]=None,
                translation: Optional[Vector3Spec]=None,
                rotation: Optional[QuaternionSpec]=None,
                scale: Optional[Vector3Spec]=None,
                matrix: Optional[Matrix4Spec]=None,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
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
        camera : Optional[BCamera]
            Camera of the node.
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
        root = isinstance(self, (_GlobalBinary, builder.Builder))
        node = _Node(name,
                    parent=cast(BNode, self) if not root else None,
                    children=children,
                    mesh=mesh,
                    camera=camera,
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
                n = n.parent
        return node

    def instantiate(self, node_or_mesh: BNode|BMesh,
                    name: str='', /,
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4Spec]=None,
                    extras: Optional[ExtrasData]=None,
                    extensions: Optional[ExtensionsData]=None,
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


class _NodeState(_CompileState[gltf.Node, '_NodeState', '_Node']):
    '''
    State for the compilation of a node.
    '''
    pass


class _Node(_BNodeContainer, BNode):
    '''
    Implementation class for `BNode`.
    '''

    @classmethod
    def state_type(cls):
        return _NodeState


    def __init__(self,
                 name: str ='', /,
                 parent: Optional[BNode]=None,
                 children: Iterable[BNode]=(),
                 mesh: Optional[BMesh]=None,
                 camera: Optional[BCamera]=None,
                 translation: Optional[Vector3Spec]=None,
                 rotation: Optional[QuaternionSpec]=None,
                 scale: Optional[Vector3Spec]=None,
                 matrix: Optional[Matrix4Spec]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
                 ):
        super(Element, self).__init__(
                         name,
                         extras=extras,
                         extensions=extensions,
                        )
        _BNodeContainer.__init__(self,
                            parent=parent,
                            nodes=children,
                            )

        self.mesh = mesh
        self.camera = camera
        self.translation = vector3(translation) if translation else None
        self.rotation = quaternion(rotation) if rotation else None
        self.scale = to_scale(scale) if scale else None
        self.matrix = to_matrix(matrix) if matrix else None
        self._local_views = _Holder(BBufferView)
        for c in self.children:
            c = cast(_Node, c)
            c._parent = self
        for c in self.children:
            print(f'child={c!r}')
        pass

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            children=[c.clone() for c in self.children],
            mesh=self.mesh,
            camera=self.camera.clone() if self.camera else None,
            translation=self.translation,
            rotation=self.rotation,
            scale=self.scale,
            matrix=self.matrix,
        )

    def _do_compile(self,
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _NodeState,
                    /):
        match phase:
            case Phase.COLLECT:
                globl.nodes.add(self)
                if self.camera is not None:
                    globl.cameras.add(self.camera)
                if self.mesh is not None:
                    globl.meshes.add(self.mesh)
                return (
                    c.compile(globl, phase)
                    for c in (self.mesh, self.camera, *self.children)
                    if c is not None
                )
            case Phase.SIZES:
                size = sum(
                    n.compile(globl, phase)
                    for n in self.children
                )
                if self.mesh is not None:
                    size += self.mesh.compile(globl, phase)
                return size
            case Phase.BUILD:
                if self.mesh is not None:
                    self.mesh.compile(globl, phase)
                def idx(c: Element[_GLTF, _STATE]) -> int:
                    return globl.idx(c)
                mesh_idx = idx(self.mesh) if self.mesh else None
                cam_idx = idx(self.camera) if self.camera else None
                return gltf.Node(
                    name=self.name,
                    mesh=mesh_idx,
                    camera=cam_idx,
                    children=[idx(child) for child in self.children],
                    translation=list(self.translation) if self.translation else None,
                    rotation=list(self.rotation) if self.rotation else None,
                    scale=list(self.scale) if self.scale else None,
                    matrix=list(float(v) for v in self.matrix) if self.matrix else None,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            case _:
                if self.mesh is not None:
                    self.mesh.compile(globl, phase)
                for child in self.children:
                    child.compile(globl, phase)



    def create_mesh(self,
                 name: str='',
                  /, *,
                 primitives: Optional[Iterable['BPrimitive']]=None,
                 weights: Optional[Iterable[float]]=None,
                 extras: Optional[ExtrasData]=None,
                 extensions: Optional[ExtensionsData]=None,
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
            ('children', len(self.children)),
            'root',
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
    extras: Optional[ExtrasData]=None,
    extensions: Optional[ExtensionsData]=None,
) -> _Node:
    '''
    Create a detached node with the given attributes.

    Parameters
    ----------
    name : str
        Name of the node.
    children : Iterable[BNode]
        List of children of the node.
    mesh : Optional[BMesh]
        Mesh of the node.
    camera : Optional[BCamera]
        Camera of the node.
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
    _Node
        A node object with the given attributes.
    '''
    for c in children:
        if not c.root:
            raise ValueError(
                f'{c.name!r} already has a parent: {c.parent!r}'
            )
    return _Node(
        name,
        children=children,
        mesh=mesh.clone() if mesh is not None else None,
        camera=camera.clone() if camera else None,
        translation=translation,
        rotation=rotation,
        scale=scale,
        matrix=matrix,
        extras=extras,
        extensions=extensions,
    )