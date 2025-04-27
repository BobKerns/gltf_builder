'''
Test cases for creating meshes for the glTF Builder library
'''

from gltf_builder import mesh

from conftest import TEST_EXTRAS, TEST_EXTENSIONS


def test_empty_mesh():
    m = mesh()
    assert m.name == ''
    assert m.extras == {}
    assert m.extensions == {}
    assert m.primitives == []
    assert m.weights == []
    assert m._index == -1


def test_mesh():
    m = mesh(
        name='MESH',
        primitives=[],
        weights=[0.1, 0.2, 0.3],
        extras=TEST_EXTRAS,
        extensions=TEST_EXTENSIONS,
    )
    assert m.name == 'MESH'
    assert m.extras == TEST_EXTRAS
    assert m.extensions == TEST_EXTENSIONS
    assert m.primitives == []
    assert m.weights == [0.1, 0.2, 0.3]
    assert m._index == -1
    assert repr(m) == '<Mesh MESH 0 primitives>'

