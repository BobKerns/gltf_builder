'''
Test that supplied geometries are correctly built
'''

from gltf_builder.geometries import get_geometry
from gltf_builder.nodes import node

def test_geo_cube(test_builder):
    CUBE = get_geometry('CUBE')
    top = node('TOP')
    top.instantiate(CUBE)


def test_geo_cube_detached(test_builder, save):
    CUBE = get_geometry('CUBE')
    top = node('TOP')
    top.instantiate(CUBE)
    test_builder.instantiate(top)
    save(test_builder.build())

    