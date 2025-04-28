'''
An example extension plugin.
'''


from typing import TypedDict, cast
from gltf_builder.core_types import ExtensionData
from gltf_builder.extensions import Extension, _ExtensionState, ExtensionPlugin


class ExampleState(_ExtensionState):
    '''
    Example extension state.
    '''

    valid: bool=False

class ExampleJson(TypedDict, total=True):
    '''
    This is the JSON schema for the extension.
    It is used to parse the extension data.
    '''
    valid: bool


class ExampleExtension(Extension[ExampleState, 'ExamplePlugin', ExtensionData]):
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
    Example extension plugin. This is identified in the `project.toml`
    file as an entry-point in the `gltf_builder.extensions` group, e.g.:
    .. code-block:: ini

        [project.entry-points.'gltf_builder.extensions']
        GLTFB_example = "gltf_builder.extensions:ExamplePlugin"

    - GLTFB_example is the name of the extension as it appears in the glTF file.
    - gltf_builder.extensions is the module that contains the plugin.
    - ExamplePlugin is the class that implements the plugin.
    '''
    @classmethod
    def extension_class(cls) -> type[ExampleExtension]:
        '''
        Return the class implementing the extension.
        This is used by the compiler to create the extension object.
        '''
        return ExampleExtension

