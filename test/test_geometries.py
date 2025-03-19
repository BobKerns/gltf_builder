'''
Test that supplied geometries are correctly built
'''

from gltf_builder.geometries import CUBE

def test_geo_cube(test_builder):
    top = test_builder.add_node(name='TOP')
    top.instantiate(CUBE)
