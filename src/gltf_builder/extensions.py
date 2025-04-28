'''
Code to handle glTF extensions
'''

import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.core_types import JsonObject


class _ExtensionState(_CompileState[JsonObject, '_ExtensionState']):
    '''
    State for the compilation of an extension.
    '''
    pass



