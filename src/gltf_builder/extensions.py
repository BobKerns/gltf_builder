'''
Code to handle glTF extensions
'''

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, entry_points, version
from typing import Generic, Protocol, TypeAlias, TypeVar, TypedDict, runtime_checkable, TYPE_CHECKING

from gltf_builder.accessors import Phase, cast
from gltf_builder.attribute_types import Iterable
from gltf_builder.compiler import _CompileState, _Scope, _Collected, _DoCompileReturn
from gltf_builder.core_types import(
    ExtensionData, JsonObject,
)
from gltf_builder.global_state import _GlobalState
from gltf_builder.elements import (
    _EXT_DATA, _EXT_PLUGIN, BExtension,
)


_EXT = TypeVar('_EXT', bound='BExtension', covariant=True)


class _ExtensionState(_CompileState[JsonObject, '_ExtensionState']):
    '''
    State for the compilation of an extension.
    '''
    pass

_EXT_STATE = TypeVar('_EXT_STATE', bound=_ExtensionState)
'''
Type variable for the class implementing the state for the extension.
'''


class _Extension(BExtension[_EXT_STATE, _EXT_PLUGIN, _EXT_DATA]):
    '''
    Implementation class for `BExtension`.
    '''

    @classmethod
    def state_type(cls) -> type[_EXT_STATE]:
        '''
        Return the type of the state for the extension.
        This is used by the compiler to create the state for the extension.
        '''
        ...

    def parse(self, state: _EXT_STATE):
        '''
        Load the extension state from the original JSON data.

        The data is in the `data` attribute of the extension.
        '''
        pass

    def unparse(self, state: _EXT_STATE) -> _EXT_DATA:
        '''
        Unparse the extension plugin to its metadata.
        This is used by the compiler to create the JSON data for the extension.
        '''
        return self.data

    def collect(self, gbl: _GlobalState, scope: _Scope, state: _EXT_STATE) -> list[_Collected]:
        '''
        Collect any additional elements that the extension needs to add
        to the global state.

        Parameters
        ----------
        gbl : _GlobalState
            The global state of the compilation process.
        scope : _Scope
            The scope of the compilation process. It should be ignored
            for now, but it may be used in the future to manage the sharing
            of data and names between different elements.
        '''
        return []

    def _do_compile(self,
                    gbl: _GlobalState,
                    scope: _Scope,
                    phase: Phase,
                    state: _EXT_STATE,
    ) -> _DoCompileReturn[_EXT_DATA]:
        '''
        Compile the extension.
        This is called by the compiler in each phase of the compilation
        process. The extension should return the JSON data for the
        extension in the `BUILD` phase. In the `COLLECT` phase, it
        should return a list of the additional elements that it has
        added to the global state.
        '''
        match phase:
            case Phase.COLLECT:
                # If the extension needs to add additional elements to the
                # global state, it should add them, and return them here.
                # If it needs to keep track of the elements it has added,
                # it should subclass _ExtensionState and add the elements
                # to the state.
                return self.collect(gbl, scope, state)
            case Phase.SIZES|Phase.OFFSETS:
                # These are used to calculate the size and location of binary
                # data in the glTF file. If the extension does not need to
                # manage binary data, it can simply return 0. If it uses
                # accessors or buffer views, and sets the sizes, they will
                # handle this.
                return 0
            case Phase.BUILD:
                # In most cases, you can simply implement the unparse method
                # and return the JSON data for the extension.

                # If you need to manage binary data, you may need to interact
                # wth other phases.
                return self.unparse(state)
            case _: pass


@runtime_checkable
class ExtensionPlugin(Protocol[_EXT]):
    '''
    Protocol for extension plugins. This is implemented
    by the extension plugins. Plugin processing happens
    in three main phases:
    1. `parse`: The plugin is given the JSON data of the extension
       and can parse it to create a `BExtension` object.
    2. `compile`: The `BExtension` object is invoked with each
        phase of the compilation process. `_do_compile()` method
        is called with the global state, scope, phase, and state.
    3. `build`: This occurs in the `BUILD` phase of the compilation process.
       The plugin should return the JSON data for the extension.

    In most cases, you can implement these phases by overriding the
    `parse`, `compile`, and `build` methods of the `BExtension` class.

    NOTE: The `BExtension` instance should be treated as immutable.
    Any state belongs in the `_ExtensionState` object, which is passed
    to the `compile` method. The plugin should subclass this, and
    return the class in the `state_type` method.

    The simplest plugin just return the original JSON data
    in the `build` phase. If it does nothing else, it indicates that
    the extension is a valid extension that will be interpreted by the
    glTF viewer. Realistically, it should at least perform validation
    of the JSON data in the `parse` phase.

    The caller can create the `BExtension` object programmatically
    and attach it to the initial data.  This replaces the `parse` phase,
    and the plugin is only invoked in the `compile` and `build` phases.

    In the `COLLECT` compilation phase, the plugin can add additional elements
    to the global state. This is useful for extensions that need to add
    additional elements to the glTF file, such as additional nodes or
    materials. The plugin should return a list of the additional elements
    that it has added to the global state. These elements will be
    '''

    name: str
    version: str

    @classmethod
    def extension_class(cls) -> type[_Extension]:
        '''
        Return the class implementing the extension.
        This is used by the compiler to create the extension object.
        '''
        return _Extension

    def __init__(self, name: str, version: str):
        '''
        Initialize the extension plugin with its name and version.
        The name is used to identify extension in the glTF file.
        The version is used to indicate the version of the extension.

        These are provided by the loading mechanism from the plugin
        metadata. The plugin should not change them.

        Parameters
        ----------
        name : str
            Name of the extension.
        version : str
            Version of the extension.
        '''
        self.name = name
        self.version = version

EXTENSION_PLUGINS: dict[str, ExtensionPlugin] = {}
'''
Dictionary of extension plugins, keyed by the name of the extension.
'''

def load_extensions():
    '''
    Load the extensions plugins from their metadata.
    '''
    def find_version(cls: type) -> str:
        '''
        Find the package that contains the extension plugin.
        This is used to access the plugin metadata.
        '''
        name = plugin_class.__module__
        if not name:
            return '0.0.0'
        sep = '.'
        while sep:
            with suppress(PackageNotFoundError):
                return version(name)
            name, sep, _ = name.rpartition('.')
        return '0.0.0'

    extensions = entry_points(group='gltf_builder.extensions')
    for ext in extensions:
        if ext.name in EXTENSION_PLUGINS:
            raise ValueError(f'Extension plugin {ext.name} already loaded')
        try:
            plugin_class = ext.load()
            module = plugin_class.__module__
            if not module:
                raise ValueError(f'Extension plugin {ext.name} has no module')
            plugin = plugin_class(ext.name, find_version(plugin_class))
        except Exception as e:
            raise ValueError(f'Failed to load extension plugin {ext.name}') from e
        if not isinstance(plugin, ExtensionPlugin):
            raise TypeError(f'Extension plugin {ext.name} is not an instance of ExtensionPlugin')
        EXTENSION_PLUGINS[ext.name] = plugin

class ExampleState(_ExtensionState):
    '''
    Example extension state.
    '''

    valid: bool=False

class ExampleJson(TypedDict, total=True):
    valid: bool


class ExampleExtension(_Extension[ExampleState, 'ExamplePlugin', ExtensionData]):
    '''
    Example extension.
    '''

    def parse(self, state: ExampleState) -> None:
        '''
        Parse the JSON data of the extension and return a `BExtension` object.
        '''
        # Can't avoid the cast in 3.11's generics.
        data = cast(ExampleJson, self.data)
        state.valid = bool(data.get('valid', False))

class ExamplePlugin(ExtensionPlugin[ExampleExtension]):
    '''
    Example extension plugin.
    '''
    @classmethod
    def extension_class(cls) -> type[ExampleExtension]:
        '''
        Return the class implementing the extension.
        This is used by the compiler to create the extension object.
        '''
        return ExampleExtension
