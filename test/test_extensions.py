'''
Test cases for the extensions module.
'''

from gltf_builder.extensions import (
    load_extensions
)
def test_extension_load():
    '''
    Test that the extension plugin is loaded correctly.
    '''
    load_extensions()

