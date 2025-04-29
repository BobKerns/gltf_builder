'''
An example extension plugin.
'''


from gltf_builder.core_types import ExtensionData, JsonData
from gltf_builder.extensions import Extension, ExtensionState, ExtensionPlugin


class ExampleState(ExtensionState['ExampleExtension', ExtensionData]):
    '''
    Example extension state.
    '''
    pass


class ExampleExtension(Extension[ExtensionData, ExampleState, 'ExamplePlugin']):
    '''
    Example extension.
    '''
    @classmethod
    def state_type(cls) -> type[ExampleState]:
        '''
        Return the type of the state for the extension.
        This is used by the compiler to create the state for the extension.

        It must be a subclass of `ExtensionState`.
        '''
        return ExampleState


    valid: bool=False
    def parse(self, data: JsonData) -> None:
        '''
        Parse the JSON data of the extension and initialize this object.
        '''
        if not isinstance(data, dict):
            raise TypeError(f'Invalid data for extension {self.name}: {data}')
        self.valid = bool(data.get('valid', False))

    def unparse(self, state: ExtensionState, /) -> ExtensionData:
        '''
        Unparse the extension plugin to its metadata.
        This is used by the compiler to create the JSON data for the extension.
        '''
        return {
            'valid': self.valid,
        }


class ExamplePlugin(ExtensionPlugin[ExampleExtension]):
    '''
    Example extension plugin.

    This is identified in the `project.toml`
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

