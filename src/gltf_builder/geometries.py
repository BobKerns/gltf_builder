'''
Prepackaged geometries (nodes with meshes), mostly useful for testing.
'''


from collections.abc import Mapping, Iterator
from contextlib import contextmanager
from typing import Any

from gltf_builder.builder import Builder
from gltf_builder.element import EMPTY_MAP, NameMode, PrimitiveMode
from gltf_builder.node import BNode


@contextmanager
def make(name: str,
         name_mode: NameMode = NameMode.UNIQUE,
         index_size: int = -1,
         extras: Mapping[str, Any]|None = None,
         extensions: Mapping[str, Any]|None = None,
         ) -> Iterator[BNode]:
    '''
    Create a detatched node to add geometry to.
    '''
    extras = extras or EMPTY_MAP
    extensions = extensions or EMPTY_MAP

    extras = {
            **extras,
            'gltf_builder': {
            'geometry': name,
        }
    }
    b = Builder(index_size,)
    node = b.add_node(name=name,
                      detached=True,
                      extras=extras,
                      extensions=extensions,
                      )
    yield node


_CUBE = (
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
    (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
)
_CUBE_FACE1 = (0, 1, 2, 3)
_CUBE_FACE2 = (4, 5, 6, 7)
_CUBE_FACE3 = (0, 4, 5, 1)
_CUBE_FACE4 = (2, 6, 7, 3)
_CUBE_FACE5 = (0, 4, 7, 3)
_CUBE_FACE6 = (1, 5, 6, 2)

_CUBE_NORMAL1 = (1, 0, 0)
_CUBE_NORMAL2 = (-1, 0, 0)
_CUBE_NORMAL3 = (0, 1, 0)
_CUBE_NORMAL4 = (0, -1, 0)
_CUBE_NORMAL5 = (0, 0, 1)
_CUBE_NORMAL6 = (0, 0, -1)

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
        node = cube.add_node(name)
        mesh = node.add_mesh(name)
        mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[_CUBE[i] for i in face], NORMAL=4 *(normal,))
    CUBE = cube

