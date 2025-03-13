'''
A container for `Element` objects, indexable by name or index.
'''

from collections.abc import Iterable
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from gltf_builder.element import Element
    T = TypeVar('T', bound=Element)
else:
    T = TypeVar('T')

class Holder(Iterable[T]):
    '''
    A container for `Element` instances, indexable by index or name.
    This also guarantees an item is added only once.
    '''
    __by_index: list[T]
    __by_name: dict[str, T]
    __by_value: set[T]
    def __init__(self, *items: T):
        self.__by_index = []
        self.__by_name = {}
        self.__by_value = set()
        self.add(*items)
        
    def set_index(self, item: T):
        '''
        The default `Holder` class does not set the item's index.
        '''
        pass
        
    def add(self, *items: T):
        '''
        Add itens to the holder, if not already present.
        '''
        for item in items:
            if item not in self.__by_value:
                self.set_index(item)
                self.__by_value.add(item)
                self.__by_index.append(item)
                if item.name:
                    self.__by_name[item.name] = item
                
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


class MasterHolder(Holder[T]):
    '''
    A `Holder` that determines the index position.
    '''
    def set_index(self, item: T):
        '''
        Set the index position to where we are about to insert it.
        '''
        item.index = len(self)