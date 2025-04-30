'''
This module provides the infrastructure for handling glTF extension in the gltf_builder package.

It defines the base classes, protocols, and mechanisms for registering, loading, and compiling glTF extensions.
Key components include:

- `_ExtensionState`: Base class for extension compilation state.
- `Extension`: Base implementation for glTF extension logic, including parsing, collecting, and compiling extension data.
- `ExtensionPlugin`: Protocol for extension plugins, specifying the required interface for plugin discovery and integration.
- `load_extensions()`: Function to discover and load extension plugins via Python entry points.
- `extension()`: Factory function to create extension objects from plugin metadata.

Example classes (`ExampleState`, `ExampleJson`, `ExampleExtension`, `ExamplePlugin`)
demonstrating how to implement a custom extension and plugin. may be found in the
`gltf_builder.plugins.example` module.

This module enables extensibility of the gltf_builder system by allowing third-party extensions to be integrated via a standardized plugin interface, supporting custom parsing, validation, and compilation phases for glTF extensions.

'''

from typing import Generic, Optional, TypeVar, cast

from gltf_builder.accessors import Phase
from gltf_builder.compiler import _CompileState, _Collected, _DoCompileReturn
from gltf_builder.core_types import(
    ExtensionData, JsonData, ScopeName,
)
from gltf_builder.elements import Element
from gltf_builder.global_state import GlobalState
from gltf_builder.plugin_loader import Plugin, load_plugins


_EXT_KEY = TypeVar('_EXT_KEY', bound=str)
'''
Type variable for the key of the extension.
'''

_EXT = TypeVar('_EXT', bound='Extension', covariant=True)
'''
Type variable for the class implementing the extension.
'''

_EXT_PLUGIN = TypeVar('_EXT_PLUGIN', bound='ExtensionPlugin')
'''
Type variable for the class implementing the extension plugin.
'''

_EXT_DATA = TypeVar('_EXT_DATA', bound=ExtensionData, covariant=True)
'''
Type variable for the JSON data representation of the extension.
'''


_EXT_STATE = TypeVar('_EXT_STATE', bound='ExtensionState')
'''
Type variable for the class implementing the state for the extension.
'''


class ExtensionState(Generic[_EXT, _EXT_DATA], _CompileState[_EXT_DATA, 'ExtensionState', 'Extension']): # type: ignore
    '''
    State for the compilation of an extension.
    '''

    @property
    def extension(self) -> _EXT:
        '''
        Return the extension object that this state is for.
        '''
        return cast(_EXT, self.element)


    @property
    def json_data(self) -> _EXT_DATA|None:
        '''
        Return the JSON data of the extension.
        This is used by the compiler to create the extension object.

        For user-supplied extensions, this may be `None`.
        '''
        return cast(_EXT_DATA|None, self.element.data)


class Extension(Generic[_EXT_DATA, _EXT_STATE, _EXT_PLUGIN], Element[_EXT_DATA, _EXT_STATE]):
    '''
    Implementation class for `Extension`.
    '''

    @classmethod
    def state_type(cls) -> type[_EXT_STATE]:
        '''
        Return the type of the state for the extension.
        This is used by the compiler to create the state for the extension.
        '''
        ...

    _scope_name = ScopeName.EXTENSION

    plugin: _EXT_PLUGIN
    __data: Optional[_EXT_DATA]
    @property
    def data(self) -> _EXT_DATA|None:
        return self.__data

    def __init__(self, plugin: _EXT_PLUGIN, data: Optional[_EXT_DATA]=None):
        super().__init__(plugin.name)
        self.__data = data
        self.plugin = plugin

    def parse(self, data: JsonData):
        '''
        Load the extension state from the original JSON data.

        The data is in the `data` attribute of the extension.
        '''
        pass

    def unparse(self, state: _EXT_STATE, /) -> Optional[_EXT_DATA]:
        '''
        Unparse the extension plugin to its metadata.
        This is used by the compiler to create the JSON data for the extension.
        '''
        return self.data

    def collect(self, globl: GlobalState, state: _EXT_STATE) -> list[_Collected]:
        '''
        Collect any additional elements that the extension needs to add
        to the global state.

        Parameters
        ----------
        globl : _GlobalState
            The global state of the compilation process.
        scope : _Scope
            The scope of the compilation process. It should be ignored
            for now, but it may be used in the future to manage the sharing
            of data and names between different elements.
        '''
        return []

    def _do_compile(self,
                    globl: GlobalState,
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
                return self.collect(globl, state)
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


class ExtensionPlugin(Generic[_EXT], Plugin):
    '''
    Protocol for extension plugins. This is implemented
    by the extension plugins. Plugin processing happens
    in three main phases:
    1. `parse`: The plugin is given the JSON data of the extension
       and can parse it to create a `Extension` object.
    2. `compile`: The `Extension` object is invoked with each
        phase of the compilation process. `_do_compile()` method
        is called with the global state, scope, phase, and state.
    3. `build`: This occurs in the `BUILD` phase of the compilation process.
       The plugin should return the JSON data for the extension.

    In most cases, you can implement these phases by overriding the
    `parse`, `compile`, and `build` methods of the `Extension` class.

    NOTE: The `Extension` instance should be treated as immutable.
    Any state belongs in the `ExtensionState` object, which is passed
    to the `compile` method. The plugin should subclass this, and
    return the class in the `state_type` method.

    The simplest plugin just returns the original JSON data
    in the `build` phase. If it does nothing else, it indicates that
    the extension is a valid extension that will be interpreted by the
    glTF viewer. Realistically, it should at least perform validation
    of the JSON data in the `parse` phase.

    The caller can create the `Extension` object programmatically
    and attach it to the initial data.  This replaces the `parse` phase,
    and the plugin is only invoked in the `compile` and `build` phases.

    In the `COLLECT` compilation phase, the plugin can add additional elements
    to the global state. This is useful for extensions that need to add
    additional elements to the glTF file, such as additional nodes or
    materials. The plugin should return a list of the additional elements
    that it has added to the global state. These elements will be
    '''

    @classmethod
    def extension_class(cls) -> type[_EXT]:
        '''
        Return the class implementing the extension.
        This is used by the compiler to create the extension object.
        '''
        ...


_EXTENSION_PLUGINS: dict[str, ExtensionPlugin] = {}
'''
Dictionary of extension plugins, keyed by the name of the extension.
'''

def load_extensions():
    '''
    Load the extensions plugins from their metadata.
    '''
    for plugin in load_plugins('gltf_builder.extensions'):
        _EXTENSION_PLUGINS[plugin.name] = cast(ExtensionPlugin, plugin)

def extension(name: str, data: ExtensionData) -> Extension:
    '''
    Create an extension object from the name and data.
    This is used by the compiler to create the extension object.
    '''
    if name not in _EXTENSION_PLUGINS:
        name = 'UNKNOWN'
    plugin = _EXTENSION_PLUGINS[name]
    ext_class = plugin.extension_class()
    return ext_class(plugin, data)
