'''
Test cases
'''

from typing import Iterable

from gltf_builder import Builder, PrimitiveMode

def test_empty_builder():
    b = Builder()
    g = b.build()
    blob = g.binary_blob()
    assert len(blob) == 0
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 0
    

CUBE = (
    (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
    (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
)
CUBE_FACE1 = (0, 1, 2, 3)
CUBE_FACE2 = (4, 5, 6, 7)
CUBE_FACE3 = (0, 4, 5, 1)
CUBE_FACE4 = (0, 4, 7, 3)
CUBE_FACE5 = (1, 2, 6, 5)
CUBE_FACE6 = (1, 5, 6, 2)

def test_cube():
    b = Builder()
    m = b.add_mesh('CUBE')
    
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
    m.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])
    assert len(m.primitives) == 6
    n = b.add_node(name='TOP', mesh=m)
    assert len(n.children) == 0
    g = b.build()
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    assert len(g.binary_blob()) == 6 * 3 * 4 * 4 + 4 * 4 * 6
    #g.save_json('cube.gltf')
    #g.save_binary('cube.glb')


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
    assert len(g.binary_blob()) == 6 * 3 * 4 * 4 + 4 * 4 * 6
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
    assert len(g.binary_blob()) == 6 * 3 * 4 * 4 + 4 * 4 * 6
    #g.save_json('cube.gltf')
    g.save_binary('faces2.glb')
    