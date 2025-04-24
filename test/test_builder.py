'''
Test cases
'''

import pytest

from collections.abc import Callable
from typing import Iterable
from dataclasses import dataclass, field
import math

import pygltflib as gltf

from gltf_builder import (
    Builder, PrimitiveMode, BMesh, Quaternion as Q,
)
from gltf_builder.elements import BNode
from gltf_builder.geometries import (
    _CUBE,
    _CUBE_FACE1, _CUBE_FACE2, _CUBE_FACE3,
    _CUBE_FACE4, _CUBE_FACE5, _CUBE_FACE6,
    _CUBE_NORMAL1, _CUBE_NORMAL2, _CUBE_NORMAL3,
    _CUBE_NORMAL4, _CUBE_NORMAL5, _CUBE_NORMAL6,
)

@dataclass
class GeometryData:
    builder: Builder
    meshes: dict[str, BMesh] = field(default_factory=dict)
    nodes: dict[str, BNode] = field(default_factory=dict)
    save: Callable[[gltf.GLTF2], gltf.GLTF2] = lambda g, **kwargs: g
    def build(self, **kwargs):
        return self.save(self.builder.build(**kwargs))
    def __getitem__(self, name):
        return (
            self.nodes.get(name)
            or self.meshes.get(name)
            or self.builder[name]
        )
    @property
    def index_size(self):
        return self.builder.index_size
    @index_size.setter
    def index_size(self, size):
        self.builder.index_size = size


def test_empty_builder(save):
    b = Builder()
    g = b.build()
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    save(g)


@pytest.fixture(scope='function')
def cube(save):
    b = Builder()
    m = b.create_mesh('CUBE_MESH')
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE1))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE2))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE3))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE4))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE5))
    m.add_primitive(PrimitiveMode.LINE_LOOP, *(_CUBE[i] for i in _CUBE_FACE6))
    top = b.create_node('TOP')
    top.create_node('CUBE', mesh=m)
    yield GeometryData(builder=b,
                   meshes={'CUBE_MESH': m},
                   nodes={'TOP': top},
                   save=save,
                )


def test_cube(cube):
    cube.index_size = -1
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_faces(save):
    b = Builder(index_size=8)
    def face(name, indices: Iterable[int]):
        m = b.create_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[_CUBE[i] for i in indices])
        return b.create_node(name, mesh=m)
    b.create_node('CUBE',
                children=[
                    face('FACE1', _CUBE_FACE1),
                    face('FACE2', _CUBE_FACE2),
                    face('FACE3', _CUBE_FACE3),
                    face('FACE4', _CUBE_FACE4),
                    face('FACE5', _CUBE_FACE5),
                    face('FACE6', _CUBE_FACE6),
               ])
    g = b.build()
    save(g)
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6
    assert len(g.binary_blob()) == size

def test_faces2(save):
    b = Builder()
    cube = b.create_node('CUBE')
    def face(name, indices: Iterable[int]):
        m = b.create_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[_CUBE[i] for i in indices])
        return cube.create_node(name, mesh=m)
    face('FACE1', _CUBE_FACE1)
    face('FACE2', _CUBE_FACE2)
    face('FACE3', _CUBE_FACE3)
    face('FACE4', _CUBE_FACE4)
    face('FACE5', _CUBE_FACE5)
    face('FACE6', _CUBE_FACE6)
    g = b.build()
    save(g)
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 4 * 4 * 6
    assert len(g.binary_blob()) == size
    

@pytest.mark.parametrize('index_size', [-1, 0, 8, 16, 32])
def test_cubeX(index_size, cube):
    cube.builder.index_size = index_size
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.nodes) == 2
    match index_size:
        case -1:
            views, ibytes = 1, 0
        case 0:
            views, ibytes = 2, 1
        case _:
            views, ibytes = 2, (index_size + 7) // 8
    assert len(g.bufferViews) == views
    size = 6 * 3 * 4 * 4 + ibytes * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_instances_mesh(cube: GeometryData):
    cube.index_size = -1
    m = cube.meshes['CUBE_MESH']

    n = cube.nodes['TOP']
    n2 = n.create_node('CUBE1', mesh=m)
    n2.translation = (1.25, 0, 0)
    assert len(n.children) == 2
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 3
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_instances(cube):
    cube.index_size = -1
    c = cube.builder.create_node('CUBE', mesh=cube.meshes['CUBE_MESH'])
    n = cube.nodes['TOP']
    n.instantiate(c,
                  translation=(1.25, 1, 0),
                  rotation=Q.from_axis_angle((1, 1, 0), math.radians(30)),
    )
    n.instantiate(c,
                  translation=(-1.25, -1, 0),
                  rotation=Q.from_axis_angle((1, 1, 0), math.radians(-30)),
                  scale=(0.5, 0.5, 0.5),
    )
    #cube.builder.print_hierarchy()
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size

def test_normal(save):
    b = Builder(index_size=-1)
    m = b.create_mesh('CUBE_MESH')
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE1],
                    NORMAL=4 *(_CUBE_NORMAL1,))
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE2],
                    NORMAL=4 *(_CUBE_NORMAL2,))
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE3],
                    NORMAL=4 *(_CUBE_NORMAL3,))
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE4],
                    NORMAL=4 *(_CUBE_NORMAL4,))
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE5],
                    NORMAL=4 *(_CUBE_NORMAL5,))
    m.add_primitive(PrimitiveMode.LINE_LOOP,
                    *[_CUBE[i] for i in _CUBE_FACE6],
                    NORMAL=4 *(_CUBE_NORMAL6,))
    top = b.create_node('TOP')
    cube = top.create_node('CUBE', mesh=m, detached=True)
    top.instantiate(cube)
    g = b.build()
    save(g)
    size = 2 * 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size
    assert len(g.bufferViews) == 1
    assert len(g.accessors) == 12
    # nodes: TOP, CUBE, copy of CUBE, and instance of CUBE
    # instance of cube is to hold the transforms without overwriting
    # the original cube if it has transforms.
    assert len(g.nodes) == 4
    