'''
Test cases
'''

import pytest

from typing import Iterable
from dataclasses import dataclass, field

from gltf_builder import Builder, PrimitiveMode, BMesh

@dataclass
class Geometry:
    builder: Builder
    meshes: dict[str, BMesh] = field(default_factory=dict)
    nodes: dict[str, BMesh] = field(default_factory=dict)
    def build(self):
        return self.builder.build()
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

def test_empty_builder(outdir):
    b = Builder()
    g = b.build()
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    g.save_json(outdir / 'empty.gltf')
    

CUBE = (
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
    (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
)
CUBE_FACE1 = (0, 1, 2, 3)
CUBE_FACE2 = (4, 5, 6, 7)
CUBE_FACE3 = (0, 4, 5, 1)
CUBE_FACE4 = (2, 6, 7, 3)
CUBE_FACE5 = (0, 4, 7, 3)
CUBE_FACE6 = (1, 5, 6, 2)

CUBE_NORMAL1 = (1, 0, 0)
CUBE_NORMAL2 = (-1, 0, 0)
CUBE_NORMAL3 = (0, 1, 0)
CUBE_NORMAL4 = (0, -1, 0)
CUBE_NORMAL5 = (0, 0, 1)
CUBE_NORMAL6 = (0, 0, -1)


@pytest.fixture
def cube():
    b = Builder()
    m = b.add_mesh('CUBE_MESH')
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])
    top = b.add_node(name='TOP')
    top.add_node('CUBE', mesh=m)
    return Geometry(builder=b, meshes={'CUBE_MESH': m}, nodes={'TOP': top})


def test_cube(cube, outdir):
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
    g.save_json(outdir / 'cube.gltf')
    g.save_binary(outdir / 'cube.glb')


def test_faces(outdir):
    b = Builder()
    def face(name, indices: Iterable[int]):
        m = b.add_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in indices])
        return b.add_node(name=name, mesh=m)
    b.add_node(name='CUBE',
                children=[
                    face('FACE1', CUBE_FACE1),
                    face('FACE2', CUBE_FACE2),
                    face('FACE3', CUBE_FACE3),
                    face('FACE4', CUBE_FACE4),
                    face('FACE5', CUBE_FACE5),
                    face('FACE6', CUBE_FACE6),
               ])
    g = b.build()
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 4 * 4 * 6
    assert len(g.binary_blob()) == size
    g.save_binary(outdir / 'faces.glb')
    

def test_faces2(outdir):
    b = Builder()
    cube = b.add_node(name='CUBE')
    def face(name, indices: Iterable[int]):
        m = b.add_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in indices])
        return cube.add_node(name=name, mesh=m)
    face('FACE1', CUBE_FACE1)
    face('FACE2', CUBE_FACE2)
    face('FACE3', CUBE_FACE3)
    face('FACE4', CUBE_FACE4)
    face('FACE5', CUBE_FACE5)
    face('FACE6', CUBE_FACE6)
    g = b.build()
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 4 * 4 * 6
    assert len(g.binary_blob()) == size
    g.save_binary(outdir / 'faces2.glb')
    

def test_cube8(cube, outdir):
    cube.builder.index_size = 8
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'cube8.glb')


def test_cube16(cube, outdir):
    cube.index_size = 16
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + 2 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'cube16.glb')


def test_cube0(cube, outdir):
    cube.index_size = 0
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'cube0.glb')


def test_cube32(cube, outdir):
    cube.index_size = 32
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + 4 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'cube32.glb')


def test_instances_mesh(cube, outdir):
    cube.index_size = -1
    m = cube.meshes['CUBE_MESH']

    n = cube.nodes['TOP']
    n2 = n.add_node('CUBE1', mesh=m)
    n2.translation = (1.25, 0, 0)
    assert len(n.children) == 2
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 3
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'instances_mesh.glb')


def test_instances(cube, outdir):
    cube.index_size = -1
    c = cube['CUBE']
    n = cube['TOP']
    n.instantiate(c,
                  translation=(1.25, 1, 0),
                  rotation=(0.18257419, 0.36514837, 0.54772256, 0.73029674),
    )
    n.instantiate(c,
                  translation=(-1.25, -1, 0),
                  rotation=(0.47415988, -0.40342268,  0.73846026,  0.25903472),
                  scale=(0.5, 0.5, 0.5),
    )
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 6
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_binary(outdir / 'instances.glb')
