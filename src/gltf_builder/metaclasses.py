"""
Decorators to simplify and regularize code.
"""

from inspect import get_annotations, isclass
from typing import Any, Generic, TypeVar, cast
from types import GenericAlias

from gltf_builder.compiler import _ENTITY, _GLTF, _CompileState, ExtensionsData, ExtrasData
import gltf_builder.entities as entities_module
import gltf_builder.global_state as GS


from gltf_builder.holders import _Holder
from gltf_builder.extensions import Extension
from gltf_builder.utils import EMPTY, first

_STATE = TypeVar('_STATE', bound='_CompileState')
_SPEC = TypeVar('_SPEC', bound='EntitySpec')


class EntitySpec(Generic[_GLTF, _SPEC], _CompileState[_GLTF, _SPEC, _SPEC]):
    """
    Class to hold the entity spec.
    This is used to hold the entity spec for the entity class.
    """

    def __init__(self,
                 entity: _SPEC,
                 name: str = "",
                 *args,
                 **kwargs):
        """
        Initialize the entity spec.
        This is used to initialize the entity spec with the entity and the name.
        """
        self.name = name
        self._initial_state = None
        for k, v in kwargs.items():
            if k in self.__dict__ or k in self.__class__.__annotations__:
                setattr(self, k, v)
            else:
                raise TypeError(
                    f"Invalid keyword argument {k} for {self.__class__.__name__}"
                )

ENTITY_NS = {
    k:v
    for d in (
        entities_module.__dict__,
        {
            'GlobalState': GS.GlobalState,
            '_SPEC': _SPEC,
            '_GLTF': _GLTF,
        },
    )
    for v in d.values()
    if isclass(v) or isinstance(v, GenericAlias)
    for k in (v.__name__, v.__qualname__)
}
''''
Namespace for the entity classes.
This enables annotation resolution across the modules,
plus the generated state classes.

It is initialized with the types from entities module,
plus other selected types.
'''

class ProtoStateClass(
    Generic[_GLTF, _STATE, _ENTITY], _CompileState[_GLTF, _STATE, _ENTITY]
):
    """
    Class to supply the `__init__` method for state classes.
    """

    def __init__(self,
                 entity: _ENTITY,
                 keys: set[str],
                 name: str = ""):
        """
        Initialize the state class.
        This is used to initialize the state class with the entity and the name.
        """
        _CompileState.__init__(self, entity, name or entity.name)
        for k in keys:
            setattr(self, k, getattr(entity, k, None))

class ProtoEntityClass(Generic[_GLTF, _SPEC], EntitySpec[_GLTF, _SPEC]):
    """
    Class to insert the initial state of an entity.
    """

    name: str
    '''
    CLASS VARIABLE

    The state class of the entity.
    This is used to create state instances for the entity, both initial and during
    compilation.
    '''
    @classmethod
    def state_class(cls) -> type['_SPEC']:
        """
        Return the state class of the entity.
        This is used to create state instances for the entity, both initial and during
        compilation.
        """
        return cls._state_class # type: ignore[assignment]

    _initial_state: '_SPEC|None' = None
    """
    The initial state of the entity.
    This is used to hold the non-default values of the entity before compilation.
    """

    @property
    def initial_state(self) -> '_SPEC':
        """
        Return the initial state of the entity.
        This is used to hold the non-default values of the entity before compilation.
        """
        if self._initial_state is None:
            # Actually set in the constructor due to the True argument.
            # We do it that way to avoid infinite recursion.
            # This assignment is a noop.
            self._initial_state = self.state_class()(self, self.name, True)
        return self._initial_state


    def _make_state(self, name: str = "") -> '_SPEC':
        """
        Create a new state instance for the entity.
        This is used to create a new state instance for the entity.
        """
        return self.state_class()(self, name)

    def __init__(self, name: str = "", *args, **kwargs):
        """
        Initialize the entity class.
        This is used to initialize the entity class with the name and the state class.
        """
        super().__init__(cast(_SPEC, self), name, *args, **kwargs)

    def __setattr__(self, name: str, value: Any):
        """
        Set the attribute of the entity class.
        This is used to set the attribute of the entity class.
        """
        if self._initial_state is None:
            if name in ("_initial_state", 'name'):
                super().__setattr__(name, value)
                return
        if name == "_initial_state":
            # This is a special case; it's already set in the constructor.
            return
        cls = self.state_class()

        if cls and any(name in c.__dict__ or name in c.__annotations__
                       for c in cls.__mro__):
            state = self._initial_state
            if state is None:
                default = first((
                                    c.__dict__.get(name)
                                    for c in cls.__mro__
                                    if name in c.__dict__
                                ),
                                EMPTY)
                if default is EMPTY or value != default:
                    setattr(self.initial_state, name, value)
                return

            state = self.initial_state
            if getattr(state, name, EMPTY) != value:
                setattr(state, name, value)
        else:
            super().__setattr__(name, value)

    def __getattribute__(self, name: str):
        """
        Get the attribute of the entity class.
        This is used to get the attribute of the entity class.
        """
        if name == "_initial_state" or name.startswith('__'):
            return super().__getattribute__(name)
        state = self._initial_state
        if state is None:
            #return self.__getattribute__(name)
            return super().__getattribute__(name)
        else:
            try:
                return getattr(state, name)
            except AttributeError:
                return super().__getattribute__(name)

class EntityMetaclass(type, Generic[_SPEC, _ENTITY]):
    """
    Metaclass to create a new class with the same name as the original class.
    This is used to create a new class with the same name as the original class,
    but with a different base class.
    """

    _name: str = ''
    _state_class: type[_SPEC]
    @classmethod
    def state_class(cls) -> type[_SPEC]:
        """
        Return the state class of the entity.
        This is used to create state instances for the entity, both initial and during
        compilation.
        """
        return cls._state_class
    '''
    These need to be here so the type checker can see that these class variables
    will exist, and their type. Since we substitute a regular `type`, we're tricking
    the type checker into thinking that these are the class variables they will
    see.

    '''


    name: str
    extensions: ExtensionsData
    extras: ExtrasData
    extension_objects: _Holder['Extension']

    def __new__(cls, name, bases, entity_attrs, /):
        """
        Create a new class with the same name as the original class, but with
        our declared attributes, so the type checker can see them.

        """
        module = entity_attrs.get("__module__", "")
        qualname = entity_attrs.get("__qualname__", module)
        if qualname != "":
            qualname = f"{qualname}."
        state_class_name = f'{name}State'
        state_mro = (ProtoStateClass, *bases)
        entity_mro = (ProtoEntityClass, *bases)
        annotations = entity_attrs.get("__annotations__", {})
        state_qualname = f"{qualname}{state_class_name}"
        entity_qualname = f"{qualname}{name}"
        entity_init = entity_attrs.get("__init__", None)
        annotations = {
            k:v
            for cls in state_mro
            for k, v in get_annotations(cls, globals=ENTITY_NS, eval_str=True).items()
        }

        keys = set(annotations.keys())
        def state_init(self, entity: '_SPEC', name: str='', init: bool=False, /):
            """
            Initialize the state class.
            This is used to initialize the state class with the entity and the name.
            """
            if init:
                entity._initial_state = self
            ProtoStateClass.__init__(self, entity, keys, name)
        state_init.__qualname__ = f'{state_qualname}.__init__'
        state_init.__name__ = '__init__'
        state_init.__module__ = entity_init.__module__
        state_init.__doc__ = entity_init.__doc__
        state_attrs = {
            **entity_attrs,
            "__qualname__": state_qualname,
            "__annotations__": {
                **{
                    k: v
                    for k, v in get_annotations(ProtoEntityClass, eval_str=True).items()
                    if k not in ("_state_class", "_state_class_base", "__initial_state")
                },
                **{
                    k: v
                    for k, v in annotations.items()
                    if k not in ("_state_class", "_state_class_base", "__initial_state")
                },
                "_state_class": state_class_name,
                "_state_class_base": state_class_name,
            },
            "__init__": state_init,
        }
        state_class = type(state_class_name, state_mro, state_attrs)
        ENTITY_NS[state_class_name] = state_class
        ENTITY_NS[state_qualname] = state_class
        entity_attrs = {
            **entity_attrs,
            '__init__': ProtoEntityClass.__init__,
            "__annotations__": {
                **get_annotations(ProtoEntityClass, eval_str=True),
                **annotations,
            },
            "_state_class_base": state_class,
            "_state_class": state_class,
            "__qualname__": entity_qualname,

        }

        entity_class = type(name, entity_mro, entity_attrs)
        ENTITY_NS[name] = entity_class
        ENTITY_NS[entity_qualname] = entity_class
        assert issubclass(entity_class, ProtoEntityClass)
        # Create a new class with the same name as the original class
        return entity_class

    def __init__(self, name: str, /):
        super(EntityMetaclass, self).__init__(self)
