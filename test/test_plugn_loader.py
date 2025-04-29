'''
Test cases for the plugin_loader module.
'''

import pytest

from gltf_builder.plugin_loader import (
    human_email,
    load_plugins,
)

@pytest.mark.parametrize("email,human", (
    ('', ''),
    ('foo', 'foo'),
    ('foo@bar', 'foo'),
    (' Larry Cat <larry2no10.uk.gov> ', 'Larry Cat'),
    ('<larry@no10.uk.gov>', 'larry'),
))
def test_human(email: str, human: str) -> None:
    '''
    Test the human_email function.
    '''
    assert human_email(email) == human

def test_load_plugins() -> None:
    '''
    Test the load_plugins function.
    '''
    from gltf_builder.plugins.example import ExamplePlugin
    from gltf_builder.assets import __version__

    plugins = load_plugins('gltf_builder.extensions')
    assert len(plugins) >= 1
    for plugin in plugins:
        if isinstance(plugin, ExamplePlugin):
            assert plugin.name == 'GLTFB_example'
            assert plugin.version == __version__
            assert plugin.author == 'Bob Kerns'
            assert plugin.summary == 'Example extension plugin.'
            return
    assert False, 'Example plugin not found'

