'''
Decorators to simplify and regularize code.
'''

from functools import wraps
from inspect import get_annotations, getmro
from typing import Any, Generic

from gltf_builder.compiler import _ENTITY




class InitialState(Generic[_ENTITY]):
    '''
    Class to insert the initial state of an entity.
    '''
    name: str
    _state_class: type[_ENTITY]
    _initial_state: _ENTITY|None = None
    '''
    The initial state of the entity.
    This is used to hold the non-default values of the entity before compilation.
    '''

    @property
    def initial_state(self) -> _ENTITY:
        '''
        Return the initial state of the entity.
        This is used to hold the non-default values of the entity before compilation.
        '''
        if self._initial_state is None:
            self._initial_state = self._state_class(self.name)
        return self._initial_state


def entity_class():
    '''
    Decorator to mark a class as an entity class.
    This is used to mark classes that are used to represent entities in the glTF file.
    '''

    @wraps(entity_class)
    def decorator(original):
        print(f'Decorating {original} as an entity class')
        ns = original.__dict__
        attrs: dict[str, Any] = get_annotations(original)
        name = original.__name__
        mro = getmro(original)
        state_type = type[original]

        def initial_state(self):
            "Get the initial state of the entity."
        return type(
            name,
            (InitialState, *mro[1:]),            {
                '__module__': original.__module__,
                '__qualname__': original.__qualname__,
                '__annotations__': {
                    '_state_class_base': state_type,
                    '_state_class': state_type,
                    **attrs,
                },
                '_state_class_base': original,
                '_state_class': original,
                '__doc__': original.__doc__,
                **{
                    k: v
                    for k, v in ns.items()
                    if not k.startswith('__')
                },
            },
        )
    return decorator

if __name__ == '__main__':
    # Test the decorator
    @entity_class()
    class TestEntity:
        x: int
        y: str = 'test'
        z: float = 0.0
        pass

    print(f'TestEntity is decorated: {get_annotations(TestEntity)}')