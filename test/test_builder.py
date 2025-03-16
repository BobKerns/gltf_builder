'''
Test cases
'''

import pytest

from typing import Iterable

from gltf_builder import Builder, PrimitiveMode

def test_empty_builder(tmp_path):
    b = Builder()
    g = b.build()
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    g.save_json('empty.gltf')
    

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
    m = b.add_mesh('CUBE')
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])
    b.add_node(name='TOP', mesh=m)
    return b

def test_cube(cube):
    cube.index_size = -1
    m = cube.meshes['CUBE']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 0
    g = cube.build()
    assert len(g.bufferViews) == 1
    assert len(g.nodes) == 1
    size = 6 * 3 * 4 * 4 + 0 * 4 * 6
    assert len(g.binary_blob()) ==  size
    g.save_json('cube.gltf')
    g.save_binary('cube.glb')


def test_faces():
    b = Builder()
    def face(name, indices: Iterable[int]):
        m = b.add_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in indices])
        return b.add_node(name=name, mesh=m, root=False)
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
    #g.save_json('cube.gltf')
    g.save_binary('faces.glb')
    


def test_faces2():
    b = Builder()
    cube = b.add_node(name='CUBE')
    def face(name, indices: Iterable[int]):
        m = b.add_mesh(name)
        m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in indices])
        return cube.add_node(name=name, mesh=m, root=False)
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
    #g.save_json('cube.gltf')
    g.save_binary('faces2.glb')
    

def test_cube8(cube):
    cube.index_size = 8
    m = cube.meshes['CUBE']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 0
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6
    assert len(g.binary_blob()) ==  size
    #g.save_json('cube.gltf')
    g.save_binary('cube8.glb')


def test_cube16(cube):
    cube.index_size = 16
    m = cube.meshes['CUBE']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 0
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    size = 6 * 3 * 4 * 4 + 2 * 4 * 6
    assert len(g.binary_blob()) ==  size
    #g.save_json('cube.gltf')
    g.save_binary('cube16.glb')


def test_cube0(cube):
    cube.index_size = 0
    m = cube.meshes['CUBE']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 0
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6
    assert len(g.binary_blob()) ==  size
    #g.save_json('cube.gltf')
    g.save_binary('cube0.glb')


def test_cube32(cube):
    cube.index_size = 32
    m = cube.meshes['CUBE']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 0
    g = cube.build()
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    size = 6 * 3 * 4 * 4 + 4 * 4 * 6
    assert len(g.binary_blob()) ==  size
    #g.save_json('cube.gltf')
    g.save_binary('cube32.glb')