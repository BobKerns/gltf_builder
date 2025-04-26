'''
Test cases for the textures module.
'''

from gltf_builder.core_types import MagFilter, MinFilter, WrapMode
from gltf_builder.images import image
from gltf_builder.samplers import sampler
from gltf_builder.textures import texture

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
                extras={'test': 'extras'},
                extensions={'test': 'extensions'},
                )
    tex = texture('Test Texture',
                  sampler=s,
                  source=src,
                  extras={'test': 'extras'},
                  extensions={'TEST_test': 'extensions'},
                  )
    assert tex is not None
    assert tex.name == 'Test Texture'
    assert tex.sampler == s
    assert tex.source == src
    assert tex.extras == {'test': 'extras'}
    assert tex.extensions == {'TEST_test': 'extensions'}