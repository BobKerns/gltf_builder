''''
Test the scenes module for glTF.
'''

from conftest import TEST_EXTENSIONS, TEST_EXTRAS

from gltf_builder import (
    get_geometry, node, scene,
)

def test_create_empty_scene():
    s = scene()
    assert s.name == ''
    assert s.nodes == []
    assert s.extras == {}
    assert s.extensions == {}

def test_create_scene():
    n = node('NODE')
    s = scene('SCENE', n,
              extras=TEST_EXTRAS,
              extensions=TEST_EXTENSIONS,
    )
    assert s.name == 'SCENE'
    assert s.nodes == [n]
    assert s.extras == TEST_EXTRAS
    assert s.extensions == TEST_EXTENSIONS

def test_add_scene(test_builder):
    n = get_geometry('CUBE')
    s = scene('SCENE', n,
              extras=TEST_EXTRAS,
              extensions=TEST_EXTENSIONS,
    )
    with test_builder() as tb:
        tb.scenes.add(s)
        g = tb.build()
        assert len(g.nodes) == 7
        assert g.nodes[0].name == 'CUBE'

def test_default_scene(test_builder):
    with test_builder() as tb:
        tb.scenes.add(scene('SCENE'))
        g = tb.build()
        assert g.scene == 0
        assert g.scenes[0].name == 'SCENE'
        assert len(g.nodes) == 0
        assert g.scenes[0].nodes == []
        
