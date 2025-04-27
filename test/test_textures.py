'''
Test cases for the textures module.
'''

from gltf_builder import (
    MagFilter, MinFilter, WrapMode,
    image, sampler, texture,
)

from conftest import TEST_EXTRAS, TEST_EXTENSIONS

def test_create_texture():
    '''
    Test that a texture can be created with the default parameters.
    '''
    src = image('Test Image', uri='test_image.png')
    s = sampler('Test Sampler',
                magFilter=MagFilter.LINEAR,
                minFilter=MinFilter.LINEAR_MIPMAP_LINEAR,\
                wrapS=WrapMode.REPEAT,
                wrapT=WrapMode.CLAMP_TO_EDGE,
                extras=TEST_EXTRAS,
                extensions=TEST_EXTENSIONS,
                )
    tex = texture('Test Texture',
                  sampler=s,
                  source=src,
                  extras=TEST_EXTRAS,
                  extensions=TEST_EXTENSIONS,
                  )
    assert tex is not None
    assert tex.name == 'Test Texture'
    assert tex.sampler == s
    assert tex.source == src
    assert tex.extras == TEST_EXTRAS
    assert tex.extensions == TEST_EXTENSIONS