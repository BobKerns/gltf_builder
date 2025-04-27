'''
Test cases for the materials module
'''

from gltf_builder import (
    AlphaMode, material,
)

from conftest import TEST_EXTRAS, TEST_EXTENSIONS


def test_create_material():
    '''
    Test that a material can be created with the default parameters.
    '''
    m = material('Test Material',
                 baseColorFactor=(1.0, 1.0, 1.0, 1.0),
                 metallicFactor=1.0,
                 roughnessFactor=1.0,
                 normalTexture=None,
                 occlusionTexture=None,
                 emissiveTexture=None,
                 emissiveFactor=(0.0, 0.0, 0.0),
                 alphaMode=AlphaMode.OPAQUE,
                 alphaCutoff=0.5,
                 doubleSided=False,
                 extras=TEST_EXTRAS,
                 extensions=TEST_EXTENSIONS,
                 )
    assert m is not None
    assert m.name == 'Test Material'
    assert m.baseColorFactor == (1.0, 1.0, 1.0, 1.0)
    assert m.metallicFactor == 1.0
    assert m.roughnessFactor == 1.0
    assert m.normalTexture is None
    assert m.occlusionTexture is None
    assert m.emissiveTexture is None
    assert m.emissiveFactor == (0.0, 0.0, 0.0)
    assert m.alphaMode == AlphaMode.OPAQUE
    assert m.alphaCutoff == 0.5
    assert not m.doubleSided
    assert m.extras == TEST_EXTRAS
    assert m.extensions == TEST_EXTENSIONS