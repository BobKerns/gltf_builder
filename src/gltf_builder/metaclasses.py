"""
Decorators to simplify and regularize code.
"""

from functools import wraps, update_wrapper
from inspect import Parameter, Signature, get_annotations, getmro, signature
from typing import Any, Generic, cast
from types import FunctionType


from gltf_builder.compiler import _ENTITY, _GLTF, _STATE, _CompileState
from gltf_builder.entities import Entity


def copy_function_with_new_closure(func, new_closure):
    """
    Copies a function and substitutes its __closure__.

    This would be one way to adapt the __init__ method
    to a different mro. However, I am leaning towards
    hardwiring a generic __init__ method to the class
    and using a common protocol for the base classes.
    """
    copied_func = FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    copied_func = update_wrapper(copied_func, func)
    cast(Any, copied_func).__kwdefaults__ = func.__kwdefaults__
    return copied_func


class ProtoStateClass(
    Generic[_GLTF, _STATE, _ENTITY], _CompileState[_GLTF, "_STATE", "_ENTITY"]
):
    """
    Class to supply the `__init__` method for state classes.
    """

    def __init__(self, entity: _ENTITY, /, name: str = "", *args, **kwargs):
        """
        Initialize the state class.
        This is used to initialize the state class with the entity and the name.
        """
        super().__init__(entity, name)
        annotations = get_annotations(self.__class__, eval_str=True)
        for k, v in kwargs.items():
            if k in self.__dict__ or k in annotations:
                setattr(self, k, v)
            else:
                raise TypeError(
                    f"Invalid keyword argument {k} for {self.__class__.__name__}"
                )


class InitialState(Generic[_STATE]):
    """
    Class to insert the initial state of an entity.
    """

    name: str
    _state_class: type[_STATE]
    _initial_state: _STATE | None = None
    """
    The initial state of the entity.
    This is used to hold the non-default values of the entity before compilation.
    """

    @property
    def initial_state(self) -> _STATE:
        """
        Return the initial state of the entity.
        This is used to hold the non-default values of the entity before compilation.
        """
        if self._initial_state is None:
            self._initial_state = self._state_class(cast(Entity, self), self.name)
        return self._initial_state


class EntityMetaclass(type, Generic[_STATE, _ENTITY]):
    """
    Metaclass to create a new class with the same name as the original class.
    This is used to create a new class with the same name as the original class,
    but with a different base class.
    """

    _state_class_base: type['_STATE']
    _state_class: type['_STATE']

    def __new__(cls, name, bases, entity_attrs, /):
        """
        Create a new class with the same name as the original class, but with
        our declared attributes, so the type checker can see them.
        """
        module = entity_attrs.get("__module__", "")
        qualname = entity_attrs.get("__qualname__", module)
        if qualname != "":
            qualname = f"{qualname}."
        if name.startswith("_"):
            state_class_name = f"{name[1:]}State"
        elif name.startswith("B") and name[1:].isupper():
            state_class_name = f"_{name[1:]}State"
        else:
            state_class_name = f"_{name[1:]}StateBase"
        state_class_name = f"_{name}StateBase"
        state_mro = (ProtoStateClass, *bases)
        entity_mro = (InitialState, *bases)
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

        def state_init(self, entity: _ENTITY, name: str='', /,
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
        cast(Any, state_init).__signature__ = state_sig
        state_attrs = {
            **entity_attrs,
            "__qualname__": state_qualname,
            "__annotations__": {
                **{
                    k: v
                    for k, v in get_annotations(InitialState, eval_str=True).items()
                    if k not in ("_state_class", "_state_class_base", "__initial_state")
                },
                **{
                    k: v
                    for k, v in annotations.items()
                    if k not in ("_state_class", "_state_class_base", "__initial_state")
                },
            },
            "__init__": state_init,
            '__signature__': state_class_sig,
        }
        state_class = type(state_class_name, state_mro, state_attrs)
        entity_attrs = {
            **entity_attrs,
            "__annotations__": {
                **get_annotations(InitialState, eval_str=True),
                **annotations,
            },
            "_state_class_base": state_class,
            "_state_class": state_class,
            "__qualname__": entity_qualname,
        }

        entity_class = type(name, entity_mro, entity_attrs)
        assert issubclass(entity_class, InitialState)
        # Create a new class with the same name as the original class
        return entity_class

    def __init__(self, name: str, /):
        super(EntityMetaclass, self).__init__(self)

    def __instancecheck__(self, instance: Any) -> bool:
        return super().__instancecheck__(instance)

    def __subclasscheck__(self, subclass: Any) -> bool:
        return super().__subclasscheck__(subclass)


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
        state_class_name = f"_{name[1:]}StateBase"
        state_class = type(state_class_name, (original, *getmro(_CompileState)), {})
        # For export; type checkers won't know.
        state_type = type[state_class]

        return type(
            name,
            (InitialState, *mro[1:]),
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
                **{k: v for k, v in ns.items() if not k.startswith("__")},
            },
        )

    return decorator
