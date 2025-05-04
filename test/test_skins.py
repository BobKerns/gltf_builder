'''
Test cases for the glTF builder library.
'''

from gltf_builder import (
    node, skin, IDENTITY4,
)

from conftest import TEST_EXTENSIONS, TEST_EXTRAS


def test_create_skin():
    '''
    Test that a skin can be created with the default parameters.
    '''

    skel = node('Test Skeleton')
    j1 = skel.node('Joint 1')
    j2 = skel.node('Joint 2')

    s = skin(skel, 'Test Skin',
             inverseBindMatrices=IDENTITY4,
             joints=[j1, j2],
             extras=TEST_EXTRAS,
             extensions=TEST_EXTENSIONS,
             )
    assert s is not None
    assert s.name == 'Test Skin'
    assert s.inverseBindMatrices ==IDENTITY4
    assert s.skeleton is skel
    assert s.joints == [j1, j2]
    assert s.extras == TEST_EXTRAS
    assert s.extensions == TEST_EXTENSIONS