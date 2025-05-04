'''
Tests for the global_state module in the glTF builder package.
'''
from gltf_builder.core_types import ScopeName
from gltf_builder.global_state import GlobalState

def test_create_global_state(test_builder, builder_extras):
    '''
    Test that a global state can be created with the default parameters.
    '''
    with test_builder() as tb:
        tb.node('Test Node')
        gs = GlobalState(tb)
        assert gs.nodes['Test Node'] is not None
        assert gs.nodes['Test Node'].name == 'Test Node'
        assert gs.asset is tb.asset
        assert gs._scope_name == ScopeName.BUILDER
        assert gs.extras == builder_extras
        assert gs.extensions == {}