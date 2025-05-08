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
    key: str
    value: int

    def __init__(self, entity: Entity[gltf.Property, 'XEntity'], name: str, value: int) -> None:
        self.key = name
        self.value = value

class XEntity(XBase, metaclass=EntityMetaclass[XBase]):
    '''
    Test entity class.
    '''
    def __init__(self, name: str, /, value: int) -> None:
        super().__init__(self, name, value)
        self.name = name
        self.value = value

XEntityState = XEntity.state_class()

def test_entity_metaclass():
    '''
    Test the entity_class decorator.
    '''

    obj = XEntity('test', 1)
    assert obj.name == 'test'
    assert obj.value == 1
    cls = obj._state_class
    base = cls(obj, 'key', 1)

    foo = obj._state_class
    foo(obj, 'key', 1)

    assert isinstance(base, _CompileState)
    assert base.name == 'key'
    assert base.value == 1
    assert isinstance(base, XBase)
    assert not isinstance(base, XEntity)