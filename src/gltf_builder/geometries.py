'''
Prepackaged geometries (nodes with meshes), mostly useful for testing.
'''


from abc import abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional, Protocol

from gltf_builder.nodes import node
from gltf_builder.vertices import vertex
from gltf_builder.attribute_types import color, point, uv, vector3
from gltf_builder.core_types import JsonObject,  NamePolicy, PrimitiveMode
from gltf_builder.elements import BNode


@contextmanager
def _make(name: str,
         index_size: int = -1,
         name_policy: Optional[NamePolicy]=None,
         extras: Optional[JsonObject]=None,
         extensions: Optional[JsonObject]|None = None,
         ) -> Iterator[BNode]:
    '''
    Create a detached node to add geometry to.
    '''
    extras = extras or {}
    extensions = extensions or {}

    extras = {
            **extras,
            'gltf_builder': {
            'geometry': name,
        }
    }
    n = node(name,
            extras=extras,
            extensions=extensions,
            )
    yield n


_CUBE = tuple(point(p) for p in (
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
    (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
))
_CUBE_FACE1 = (0, 1, 2, 3)
_CUBE_FACE2 = (4, 5, 6, 7)
_CUBE_FACE3 = (0, 4, 5, 1)
_CUBE_FACE4 = (2, 6, 7, 3)
_CUBE_FACE5 = (0, 4, 7, 3)
_CUBE_FACE6 = (1, 5, 6, 2)

_CUBE_NORMAL1 = vector3(1, 0, 0)
_CUBE_NORMAL2 = vector3(-1, 0, 0)
_CUBE_NORMAL3 = vector3(0, 1, 0)
_CUBE_NORMAL4 = vector3(0, -1, 0)
_CUBE_NORMAL5 = vector3(0, 0, 1)
_CUBE_NORMAL6 = vector3(0, 0, -1)

_CUBE_COLORS = (
    color(0.25, 1.0, 0.0, 1.0),
    color(1.0, 0.25, 0.0, 1.0),
    color(1.0, 0.25, 0.25, 1.0),
    color(0.25, 1.0, 1.0, 1.0),
    color(0.0, 1.0, 0.0, 1.0),
    color(1.0, 0.0, 0.0, 1.0),
    color(0.0, 0.0, 1.0, 1.0),
    color(1.0, 0.0, 1.0, 1.0),
)

_CUBE_FACES = (
        (_CUBE_FACE1, _CUBE_NORMAL1),
        (_CUBE_FACE2, _CUBE_NORMAL2),
        (_CUBE_FACE3, _CUBE_NORMAL3),
        (_CUBE_FACE4, _CUBE_NORMAL4),
        (_CUBE_FACE5, _CUBE_NORMAL5),
        (_CUBE_FACE6, _CUBE_NORMAL6),
    )

_CUBE_VERTICES = tuple(
    vertex(_CUBE[p],
           NORMAL=n,
           COLOR_0=_CUBE_COLORS[p],
          TEXCOORD_0=uv(p % 2, p // 2),
    )
    for f, n in _CUBE_FACES
    for p in f
)


class GeometryFn(Protocol):
    @abstractmethod
    def __call__(self, node: BNode, /):
        '''
        Initialize a detached node with geometry.
        '''
        ...

class _GeometryFn:
    name: str
    _fn: GeometryFn
    index_size: int
    name_policy: Optional[NamePolicy]
    extras: Optional[JsonObject]
    extensions: Optional[JsonObject]
    __name__: str
    __qualname__: str
    __module__: str
    node: BNode|None = None

    def __init__(self, name: str,
                 fn: GeometryFn, /, *,
                 index_size: int = -1,
                 name_policy: Optional[NamePolicy]=None,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject] = None,
                 ):
        self.name = name
        self._fn = fn
        self.__qualname__ = fn.__qualname__
        self.__name__ = getattr(fn, '__name__', fn.__qualname__ )
        self.__module__ = fn.__module__
        self.index_size = index_size
        self.name_policy = name_policy
        extras = extras or {}
        self.extras = {
                **extras,
                'gltf_builder': {
                'geometry': name,
            }
        }
        self.extensions = extensions or {}

    def __call__(self) -> BNode:
        if self.node is not None:
            return self.node
        n = node(self.name,
                        extras=self.extras,
                        extensions=self.extensions,
                        )
        self._fn(n)
        #self.node = n
        return n


_GEOMETRIES: dict[str, _GeometryFn] = {}
'''
Predefined named geometry functions
'''


def define_geometry(name: str, fn: GeometryFn, /, *,
                    index_size: int = -1,
                    name_policy: Optional[NamePolicy]=None,
                    extras: Optional[JsonObject]=None,
                    extensions: Optional[JsonObject] = None,
                    ) -> None:
    '''
    Define a detached node to add geometry to.

    Creation is deferred until first use
    '''
    if name in _GEOMETRIES:
        raise ValueError(f'Geometry {name!r} already defined')
    _GEOMETRIES[name] = _GeometryFn(name,
                                    fn,
                                    index_size=index_size,
                                    name_policy=name_policy,
                                    extras=extras,
                                    extensions=extensions,
                                    )

def get_geometry(name: str) -> BNode:
    '''
    Get a predefined geometry node.
    '''
    if name not in _GEOMETRIES:
        raise ValueError(f'Geometry {name!r} not defined')
    return _GEOMETRIES[name]()


def _cube(cube: BNode):
    '''
    Create a cube with 6 faces.
    '''
    for i, (face, normal) in enumerate(_CUBE_FACES):
        name = f'FACE{i+1}'
        node = cube.create_node(name)
        mesh = node.create_mesh(name)

        uvx = uv(0.0, 0.0), uv(0.0, 1.0), uv(1.0, 1.0), uv(1.0, 0.0)
        mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[_CUBE_VERTICES[i] for i in face],)
    CUBE = cube

define_geometry('CUBE', _cube)
