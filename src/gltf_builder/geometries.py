'''
Prepackaged geometries (nodes with meshes), mostly useful for testing.
'''


from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from gltf_builder.attribute_types import color, point, uv, vector3
from gltf_builder.core_types import JsonObject, NameMode, PrimitiveMode
from gltf_builder.builder import Builder
from gltf_builder.element import BNode


@contextmanager
def make(name: str,
         name_mode: NameMode = NameMode.UNIQUE,
         index_size: int = -1,
         extras: Optional[JsonObject]=None,
         extensions: Optional[JsonObject]|None = None,
         ) -> Iterator[BNode]:
    '''
    Create a detatched node to add geometry to.
    '''
    extras = extras or {}
    extensions = extensions or {}

    extras = {
            **extras,
            'gltf_builder': {
            'geometry': name,
        }
    }
    b = Builder(index_size=index_size)
    node = b.create_node(name,
                      detached=True,
                      extras=extras,
                      extensions=extensions,
                      )
    yield node


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

with make('CUBE') as cube:
    for i, (face, normal) in enumerate(_CUBE_FACES):
        name = f'FACE{i+1}'
        node = cube.create_node(name)
        mesh = node.create_mesh(name)
               
        uvx = uv(0.0, 0.0), uv(0.0, 1.0), uv(1.0, 1.0), uv(1.0, 0.0)
        mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[_CUBE[i] for i in face],
                           NORMAL=4 *(normal,),
                           COLOR_0=[_CUBE_COLORS[i] for i in face],
                           TEXCOORD_0= uvx
                           )
    CUBE = cube

