'''
A container for `Element` objects, indexable by name or index.
'''

from collections.abc import Iterable
from typing import TypeVar, Any, TYPE_CHECKING, overload

import pygltflib as gltf

if TYPE_CHECKING:
    from gltf_builder.elements import Element
    T = TypeVar('T', bound=Element[Any, Any])
else:
    T = TypeVar('T')

class _Holder(Iterable[T]):
    '''
    A container for `Element` instances, indexable by index or name.
    This also guarantees an item is added only once.
    '''
    __type: type[T]
    __by_index: list[T]
    __by_name: dict[str, T]
    __by_value: set[T]
    def __init__(self, type_: type[T], *items: T):
        self.__type = type_
        self.__by_index = []
        self.__by_name = {}
        self.__by_value = set()
        self.add(*items)

    def add(self, *items: T):
        '''
        Add items to the holder, if not already present.
        '''
        for item in items:
            if item not in self.__by_value:
                self.__by_value.add(item)
                self.__by_index.append(item)
                if item.name:
                    self.__by_name[item.name] = item

    @overload
    def get(self, key: str|int, default: T) -> T: ...
    @overload
    def get(self, key: str|int, default: None) -> T|None: ...
    @overload
    def get(self, key: str|int) -> T|None: ...
    def get(self, key: str|int, default: T|None=None) -> T|None:
        '''
        Get an item by index or name, or return `default` if not found.
        '''
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        '''
        We can iterate over all items in the `Holder`.
        '''
        return iter(self.__by_index)

    def __getitem__(self, key: str|int) -> T:
        '''
        We can get items by index (position) or name, if named.
        '''
        if isinstance(key, str):
            return self.__by_name[key]
        return self.__by_index[key]

    def __len__(self):
        '''
        The number of items held.
        '''
        return len(self.__by_index)

    def __contains__(self, item: T|str|int):
        '''
        Return `True` if the item, it's name, or its index is present.
        '''
        match item:
            case str():
                return item in self.__by_name
            case int():
                return item >= 0 and item < len(self)
            case _:
                return item in self.__by_index

    def __repr__(self):
        '''
        A string representation of the `Holder`.
        '''
        return f'<{self.__class__.__name__}({self.__type.__name__}, {len(self)})>'
