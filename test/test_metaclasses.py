'''
Module to test the decorators in the decorators.py file.
'''


import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.entities import Entity
from gltf_builder.metaclasses import EntityMetaclass, EntitySpec



class XBase(EntitySpec[gltf.Property, 'XEntity'], Entity[gltf.Property, 'XEntity']):
    '''
    Test class.
    '''
    key: str = ''
    value: int = 0



class XEntity(XBase, metaclass=EntityMetaclass[XBase, 'XEntity']):
    '''
    Test entity class.
    '''
    def __init__(self, name: str, /, value: int=0) -> None:
        self.value = value


def test_metaclass_default():
    '''
    Test the EntityMetaclass initialization.
    '''

    obj = XEntity('test')
    assert isinstance(obj, XEntity)
    assert obj.name == 'test'
    assert obj.value == 0
    cls = type(obj)._state_class # type: ignore[assignment]
    assert isinstance(cls, (EntityMetaclass, type))
    assert obj._initial_state is None
    state = obj._make_state()
    assert obj._initial_state is None

    assert isinstance(state, _CompileState)
    assert state.name == 'test'
    assert state.value == 0
    assert isinstance(state, XBase)
    assert not isinstance(state, XEntity)


def test_metaclass_initial_state():
    '''
    Test the EntityMetaclass initial state.
    '''

    obj = XEntity('test', 1)
    assert isinstance(obj, XEntity)
    assert obj.name == 'test'
    assert obj._initial_state is None
    assert obj.value == 0
    obj.value = 0
    assert obj._initial_state is None
    obj.value = 1
    state = obj._initial_state
    assert isinstance(state, XBase)
    assert isinstance(state, obj.__class__._state_class) # type: ignore[assignment]
    assert state.value == 1
    assert obj.value == 1