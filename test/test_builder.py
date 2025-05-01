'''
Test cases
'''

from typing import Iterable
import math


from gltf_builder import (
    Builder, PrimitiveMode, Quaternion as Q,
    IndexSize, node,
)
from gltf_builder.geometries import (
    _CUBE,
    _CUBE_FACE1, _CUBE_FACE2, _CUBE_FACE3,
    _CUBE_FACE4, _CUBE_FACE5, _CUBE_FACE6,
    _CUBE_NORMAL1, _CUBE_NORMAL2, _CUBE_NORMAL3,
    _CUBE_NORMAL4, _CUBE_NORMAL5, _CUBE_NORMAL6,
)


def test_empty_builder(save):
    b = Builder()
    assert b.index_size == IndexSize.NONE
    g = b.build()
    assert b.index_size == IndexSize.NONE
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    save(g)


def test_empty_index_size_constuctor(index_sizes,
                                     save):
    index_size, idx_bytes, idx_views = index_sizes
    b = Builder(index_size=index_size)
    assert b.index_size == index_size
    g = b.build()
    assert b.index_size == index_size
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    save(g)


def test_empty_index_size_build(index_sizes,
                                     save):
    index_size, idx_bytes, idx_views = index_sizes
    b = Builder()
    assert b.index_size == IndexSize.NONE
    g = b.build(index_size=index_size)
    assert b.index_size == index_size
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 0
    assert len(g.bufferViews) == 0
    assert len(g.nodes) == 0
    save(g)


def test_cube(index_sizes, cube):
    index_size, idx_bytes, idx_views = index_sizes
    cube.index_size = index_size
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.bufferViews) == 1 + idx_views
    assert len(g.nodes) == 2
    size = 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_faces(DEBUG, index_sizes, save):
    index_size, idx_bytes, idx_buffers = index_sizes
    b = Builder(index_size=index_size)
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
    assert len(g.bufferViews) == 1 + idx_buffers
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + 1 * 4 * 6 * idx_bytes
    assert len(g.binary_blob()) == size

def test_faces2(index_sizes, save):
    index_size, idx_bytes, idx_buffers = index_sizes
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
    g = b.build(index_size=index_size)
    save(g)
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 1 + idx_buffers
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) == size


def test_cubeX(index_sizes, cube):
    index_size, idx_bytes, idx_views = index_sizes
    cube.builder.index_size = index_size
    m = cube.meshes['CUBE_MESH']
    assert len(m.primitives) == 6
    n = cube.nodes['TOP']
    assert len(n.children) == 1
    g = cube.build()
    assert len(g.nodes) == 2
    assert len(g.bufferViews) == 1 + idx_views
    size = 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_instances_mesh(index_sizes, cube):
    index_size, idx_bytes, idx_views = index_sizes
    cube.index_size = index_size
    m = cube.meshes['CUBE_MESH']

    n = cube.nodes['TOP']
    n2 = n.create_node('CUBE1', mesh=m)
    n2.translation = (1.25, 0, 0)
    assert len(n.children) == 2
    g = cube.build()
    assert len(g.bufferViews) == 1 + idx_views
    assert len(g.nodes) == 3
    size = 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) ==  size


def test_instances(index_sizes, cube):
    index_size, idx_bytes, idx_views = index_sizes
    cube.index_size = index_size
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
    assert len(g.bufferViews) == 1 + idx_views
    assert len(g.nodes) == 7
    size = 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) ==  size

def test_normal(index_sizes, save):
    index_size, idx_bytes, idx_views = index_sizes
    b = Builder(index_size=index_size)
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
    cube = node('CUBE', mesh=m)
    top.instantiate(cube)
    g = b.build()
    save(g)
    size = 2 * 6 * 3 * 4 * 4 + idx_bytes * 4 * 6
    assert len(g.binary_blob()) ==  size
    assert len(g.bufferViews) == 1 + idx_views
    assert len(g.accessors) == 12 + 6 * idx_views
    assert len(g.nodes) == 3
