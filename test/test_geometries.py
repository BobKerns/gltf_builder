'''
Test that supplied geometries are correctly built
'''

from gltf_builder import (
    get_geometry, node,
)

def test_geo_cube():
    CUBE = get_geometry('CUBE')
    top = node('TOP')
    top.instantiate(CUBE)


def test_geo_cube_detached(test_builder):
    CUBE = get_geometry('CUBE')
    top = node('TOP')
    top.instantiate(CUBE)
    with test_builder() as tb:
        tb.instantiate(top)
    