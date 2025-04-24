'''
Tests for the sampler module in the glTF builder package.
'''

from gltf_builder.core_types import MagFilter, MinFilter, ScopeName, WrapMode
from gltf_builder.samplers import sampler


def test_create_sampler():
    '''
    Test that a sampler can be created with the default parameters.
    '''
    s = sampler('SAMPLER',
                magFilter=MagFilter.LINEAR,
                minFilter=MinFilter.LINEAR_MIPMAP_LINEAR,
                wrapS=WrapMode.REPEAT,
                wrapT=WrapMode.CLAMP_TO_EDGE,
                extras={'test': 'extras'},
                extensions={'test': 'extensions'},
                )
    assert s is not None
    assert s.name == 'SAMPLER'
    assert s.wrapS == WrapMode.REPEAT
    assert s.wrapT == WrapMode.CLAMP_TO_EDGE
    assert s.magFilter == MagFilter.LINEAR
    assert s.minFilter == MinFilter.LINEAR_MIPMAP_LINEAR
    assert s.extras == {'test': 'extras'}
    assert s.extensions == {'test': 'extensions'}
    assert s._index == -1
    assert s._scope_name == ScopeName.SAMPLER

