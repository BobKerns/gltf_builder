'''
Test cases for the extensions module.
'''

import pytest

from gltf_builder.extensions import (
    extension,
    load_extensions
)
from gltf_builder.global_state import GlobalState
from gltf_builder.plugins.example import ExampleExtension
def test_extension_load(no_plugins):
    '''
    Test that the extension plugin is loaded correctly.
    '''
    load_extensions()

@pytest.mark.parametrize("input,output", (
    ({'valid': True}, True),
    ({'valid': False}, False),
    (None, False),
))
def test_extension(plugins,
                   test_builder,
                   input,
                   output):
    ext = extension('GLTFB_example', {'valid': True})
    assert isinstance(ext, ExampleExtension)
    if input is not None:
        ext.parse(input)
    assert ext.name == 'GLTFB_example'
    gbl = GlobalState(test_builder)
    state = gbl.state(ext)
    assert state.extension == ext
    assert ext in gbl.extension_objects
    assert ext.valid == output


