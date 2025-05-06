'''
A container for `Entity` objects, indexable by name or index.
'''

from collections.abc import Iterable
from typing import TypeVar, Any, TYPE_CHECKING, overload


if TYPE_CHECKING:
    from gltf_builder.entities import Entity
    T = TypeVar('T', bound=Entity[Any, Any])
else:
    T = TypeVar('T')

class _RO_Holder(Iterable[T]):
    '''
    A read-only container for `Entity` instances, indexable by index or name.
    '''
    __type: type[T]
    @property
    def type(self) -> type[T]:
        '''
        The type of the items in the holder.
        '''
        return self.__type
    _by_index: list[T]
    _by_name: dict[str, T]

    def __init__(self, type_: 'type[T]', *items: T):
        self.__type = type_
        self._by_index = []
        self._by_name = {}
        self.add_from(items)

    def add_from(self, items: Iterable[T]):
        '''
        Add items to the holder.
        '''
        for item in items:
            self._by_index.append(item)
            if item.name:
                self._by_name[item.name] = item

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
        return iter(self._by_index)
    def __getitem__(self, key: str|int) -> T:
        '''
        We can get items by index (position) or name, if named.
        '''
        if isinstance(key, str):
            return self._by_name[key]
        return self._by_index[key]

    def __len__(self):
        '''
        The number of items in the `Holder`.
        '''
        return len(self._by_index)

    def __bool__(self):
        '''
        The `Holder` is truthy if it contains items.
        '''
        return bool(self._by_index)


class _Holder(_RO_Holder[T]):
    '''
    A container for `Entity` instances, indexable by index or name.
    This also guarantees an item is added only once.
    '''
    def __init__(self, type_: type[T], *items: T):
        super().__init__(type_)
        self.add_from(items)

    def add(self, item: T):
        '''
        Add an item to the holder, if not already present.
        '''
        if item not in self._by_index:
            self._by_index.append(item)
            if item.name:
                self._by_name[item.name] = item

    def add_from(self, items: Iterable[T]):
        '''
        Add items to the holder, if not already present.
        '''
        for item in items:
            if item not in self._by_index:
                self._by_index.append(item)
                if item.name:
                    self._by_name[item.name] = item

    def __setitem__(self, key: str|int, value: T):
        '''
        We can set items by index (position) or name, if named.
        '''
        if isinstance(key, str):
            self._by_name[key] = value
            if value not in self._by_index:
                self._by_index.append(value)
        else:
            self._by_index[key] = value
            if value.name:
                self._by_name[value.name] = value

    def __delitem__(self, key: str|int):
        '''
        We can delete items by index (position) or name, if named.
        '''
        if isinstance(key, str):
            item = self._by_name.pop(key)
            self._by_index.remove(item)
        else:
            item = self._by_index.pop(key)
            if item.name:
                self._by_name.pop(item.name, None)

    def __repr__(self):
        '''
        A string representation of the `Holder`.
        '''
        return f'<{self.__class__.__name__}[{self.type.__name__}]({len(self)})>'
