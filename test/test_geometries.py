'''
Test that supplied geometries are correctly built
'''

from gltf_builder.geometries import CUBE

def test_geo_cube(test_builder):
    top = test_builder.create_node('TOP')
    top.instantiate(CUBE)


def test_geo_cube_detached(test_builder, save):
    top = test_builder.create_node('TOP', detached=True)
    top.instantiate(CUBE)
    test_builder.instantiate(top)
    save(test_builder.build())

    