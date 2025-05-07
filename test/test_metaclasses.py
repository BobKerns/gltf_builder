'''
Module to test the decorators in the decorators.py file.
'''


from typing import TypeAlias, cast

import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.metaclasses import EntityMetaclass


class XBase:
    name: str
    '''
    Test class.
    '''
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class XEntity(XBase, metaclass=EntityMetaclass['_XEntityState', 'XEntity']):
    '''
    Test entity class.
    '''
    key: str
    value: int

    def __init__(self, name: str, value: int) -> None:
        super().__init__(name)
        self.value = value
#_XEntityState: TypeAlias = _CompileState[gltf.Property, '_XEntityState', 'XEntity']

def test_entity_metaclass():
    '''
    Test the entity_class decorator.
    '''

    obj = XEntity('test', 1)
    assert obj.name == 'test'
    assert obj.value == 1
    obj = cast(XEntity, obj)
    base = obj._state_class_base(obj, 'key', 1)

    foo = obj._state_class_base
    foo(obj, 'key', 1)

    assert isinstance(base, _CompileState)
    assert base.name == 'key'
    assert base.value == 1
    assert isinstance(base, XBase)
    assert not isinstance(base, XEntity)