'''
Tests for the sampler module in the glTF builder package.
'''

from gltf_builder import (
    MagFilter, MinFilter, EntityType, WrapMode,
    sampler,
)

from  conftest import TEST_EXTRAS, TEST_EXTENSIONS


def test_create_sampler():
    '''
    Test that a sampler can be created with the default parameters.
    '''
    s = sampler('SAMPLER',
                magFilter=MagFilter.LINEAR,
                minFilter=MinFilter.LINEAR_MIPMAP_LINEAR,
                wrapS=WrapMode.REPEAT,
                wrapT=WrapMode.CLAMP_TO_EDGE,
                extras=TEST_EXTRAS,
                extensions=TEST_EXTENSIONS,
                )
    assert s is not None
    assert s.name == 'SAMPLER'
    assert s.wrapS == WrapMode.REPEAT
    assert s.wrapT == WrapMode.CLAMP_TO_EDGE
    assert s.magFilter == MagFilter.LINEAR
    assert s.minFilter == MinFilter.LINEAR_MIPMAP_LINEAR
    assert s.extras == TEST_EXTRAS
    assert s.extensions == TEST_EXTENSIONS
    assert s._entity_type == EntityType.SAMPLER

