''''
Test the scenes module for glTF.
'''

from gltf_builder.core_types import JsonObject
from gltf_builder.geometries import get_geometry
from gltf_builder.nodes import node
from gltf_builder.scenes import scene


TEST_EXTRAS: JsonObject={"EXTRA": "DATA"}
TEST_EXTENSIONS: JsonObject={"TEST": {"EXTRA": "DATA"}}

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
    test_builder.scenes.add(s)
    g = test_builder.build()
    assert len(g.nodes) == 1
    assert g.nodes[0].name == 'CUBE'
