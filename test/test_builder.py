from gltf_builder import Builder, PrimitiveType

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
CUBE_FACE6 = (1, 5, 7, 3)

def test_cube():
    b = Builder()
    n = b.add_node()
    
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE1)
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE2)
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE3)
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE4)
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE5)
    n.add_primitive(PrimitiveType.LINE_LOOP, *CUBE_FACE6)
    assert len(n.children) == 0
    assert len(n.primitives) == 6
    g = b.build()
    assert len(g.buffers) == 1
    assert len(g.bufferViews) == 2
    assert len(g.nodes) == 1
    assert len(g.binary_blob()) == 8 * 3 * 4 + 4 * 4 * 6
    