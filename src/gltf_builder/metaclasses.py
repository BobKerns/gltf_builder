"""
Decorators to simplify and regularize code.
"""

from functools import wraps
from inspect import Parameter, Signature, get_annotations, getmro, isclass, signature
from typing import Any, Generic, TypeVar, cast
from types import GenericAlias


from gltf_builder.compiler import _ENTITY, _GLTF, _CompileState, ExtensionsData, ExtrasData
import gltf_builder.entities as entities_module
import gltf_builder.global_state as GS


from gltf_builder.holders import _Holder
from gltf_builder.extensions import Extension


ENTITY_NS = {
    k:v
    for d in (
        entities_module.__dict__,
        {
            'GlobalState': GS.GlobalState
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


_STATE = TypeVar('_STATE', bound='_CompileState')
_SPEC = TypeVar('_SPEC', bound='EntitySpec')

class EntitySpec(Generic[_GLTF, _SPEC], _CompileState[_GLTF, _SPEC, _SPEC]):
    """
    Class to hold the entity spec.
    This is used to hold the entity spec for the entity class.
    """

    def __init__(self, entity: '_SPEC', name: str = "", *args, **kwargs):
        """
        Initialize the entity spec.
        This is used to initialize the entity spec with the entity and the name.
        """
        self.name = name
        self.entity = entity
        self._state_class = None
        self._initial_state = None
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise TypeError(
                    f"Invalid keyword argument {k} for {self.__class__.__name__}"
                )


class ProtoStateClass(
    Generic[_GLTF, _STATE, _ENTITY], _CompileState[_GLTF, _STATE, _ENTITY]
):
    """
    Class to supply the `__init__` method for state classes.
    """

    def __init__(self, entity: _ENTITY, /, name: str = "", *args, **kwargs):
        """
        Initialize the state class.
        This is used to initialize the state class with the entity and the name.
        """
        _CompileState.__init__(self, entity, name)
        annotations = {
            k:v
            for cls in getmro(self.__class__)
            for k, v in get_annotations(cls, globals=ENTITY_NS, eval_str=True).items()
        }
        for k, v in kwargs.items():
            if k in self.__dict__ or k in annotations:
                setattr(self, k, v)
            else:
                raise TypeError(
                    f'Invalid keyword argument "{k}" for {self.__class__.__name__}'
                )


class ProtoEntityClass(Generic[_GLTF, _SPEC], EntitySpec[_GLTF, _SPEC]):
    """
    Class to insert the initial state of an entity.
    """

    name: str
    _state_class: type['_SPEC']
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
        return cls._state_class

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
            self._initial_state = self._state_class(self, self.name)
        return self._initial_state

    def __init__(self, name: str = "", *args, **kwargs):
        """
        Initialize the entity class.
        This is used to initialize the entity class with the name and the state class.
        """
        super().__init__(cast(_SPEC, self), name, *args, **kwargs)
        self.name = name
        self._initial_state = None

    def __setattr__(self, name: str, value: Any):
        """
        Set the attribute of the entity class.
        This is used to set the attribute of the entity class.
        """
        if name in self._state_class.__dict__:
            setattr(self.initial_state, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        """
        Get the attribute of the entity class.
        This is used to get the attribute of the entity class.
        """
        if self._initial_state and  name in self._state_class.__dict__:
            return getattr(self.initial_state, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

class EntityMetaclass(type, Generic[_SPEC]):
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
        I guess the switcheroo isn't really necessary; but I don't want to complicate
    further subclassing of these types. Removing it won't help with the circular
    dependency, at least in Python 3.11. 3.12 and later might stand a chance with
    `typedef`.
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
        entity_sig = signature(entity_init)
        entity_params = entity_sig.parameters
        state_params = (
            Parameter("self", Parameter.POSITIONAL_ONLY),
            Parameter("entity", Parameter.POSITIONAL_ONLY),
            Parameter("name", Parameter.POSITIONAL_OR_KEYWORD),
            *(
                param
                for param in entity_params.values()
                if param.name not in ("self", "name", "entity")
            ),
        )
        state_sig = Signature(state_params)
        state_class_sig = Signature(state_params[1:],
                                    return_annotation=_STATE)

        def state_init(self, entity: '_SPEC', name: str='', /,
                       *args,
                       **kwargs):
            """
            Initialize the state class.
            This is used to initialize the state class with the entity and the name.
            """
            more_positional_ok = False
            more_kwargs_ok = False
            new_positional = []
            new_kwargs = {}
            idx = 0
            for param in state_sig.parameters.values():
                if param.name in ('self', 'name', 'entity'):
                    continue
                match param.kind:
                    case Parameter.POSITIONAL_ONLY:
                        if idx >= len(args):
                            raise TypeError(f'Missing argument "{param.name}"')
                        new_kwargs[param.name] = args[idx]
                        idx += 1
                    case Parameter.POSITIONAL_OR_KEYWORD:
                        if param.name in kwargs:
                            new_kwargs[param.name] = kwargs.pop(param.name)
                        elif idx >= len(args):
                            raise TypeError(f'Missing argument "{param.name}"')
                        else:
                            new_kwargs[param.name] = args[(idx := idx + 1) - 1]
                    case Parameter.KEYWORD_ONLY:
                        new_kwargs[param.name] = kwargs.pop(param.name)
                    case Parameter.VAR_POSITIONAL:
                        more_positional_ok = True
                    case Parameter.VAR_KEYWORD:
                        more_kwargs_ok = True

            new_positional = args[:idx]
            if not more_positional_ok and idx < len(args):
                raise TypeError(
                    f"Too many positional arguments for {state_qualname}"
                )
            if not more_kwargs_ok and kwargs:
                raise TypeError(
                    f"Too many keyword arguments for {state_qualname}"
                )

            ProtoStateClass.__init__(self, entity, name,
                                     *new_positional,
                                     **new_kwargs)
        state_init.__qualname__ = f'{state_qualname}.__init__'
        state_init.__name__ = '__init__'
        state_init.__module__ = entity_init.__module__
        state_init.__doc__ = entity_init.__doc__
        state_init._original_globals = entity_init.__globals__ # type: ignore
        cast(Any, state_init).__signature__ = state_sig
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
            '__signature__': state_class_sig,
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


def entity_class():
    """
    Decorator to mark a class as an entity class.
    This is used to mark classes that are used to represent entities in the glTF file.
    """

    @wraps(entity_class)
    def decorator(original):
        print(f"Decorating {original} as an entity class")
        ns = original.__dict__
        attrs: dict[str, Any] = get_annotations(original, eval_str=True)
        name = original.__name__
        mro = getmro(original)
        state_class_name = f"_{name[1:]}State"
        state_class = type(state_class_name, (original, *getmro(_CompileState)), {})
        ENTITY_NS[state_class_name] = state_class
        # For export; type checkers won't know.
        state_type = type[state_class]

        return type(
            name,
            (ProtoEntityClass, *mro[1:]),
            {
                "__module__": original.__module__,
                "__qualname__": original.__qualname__,
                "__annotations__": {
                    "_state_class_base": state_type,
                    "_state_class": state_type,
                    **attrs,
                },
                "_state_class_base": state_class,
                "_state_class": state_class,
                "__doc__": original.__doc__,
                '_name': '',
                **{k: v for k, v in ns.items() if not k.startswith("__")},
            },
        )

    return decorator
