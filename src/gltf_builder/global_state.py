'''
global compilation state
'''

from gltf_builder.protocols import _BuilderProtocol, _Scope
from gltf_builder.utils import std_repr

class _GlobalState:
    '''
    Global state for the compilation of a glTF document.
    '''
    builder: _BuilderProtocol
    def __init__(self, builder: _BuilderProtocol) -> None:
        self.builder = builder

        self.asset = builder.asset
        self.meshes = builder.meshes
        self.cameras = builder.cameras
        self._buffers = builder._buffers
        self._views = builder._views
        self._accessors = builder._accessors
        self.images = builder.images
        self.materials = builder.materials
        self.samplers = builder.samplers
        self.scenes = builder.scenes
        self.skins = builder.skins
        self.textures = builder.textures
        self.index_size = builder.index_size
        self.extras = builder.extras or {}
        self.extensions = builder.extensions or {}
        self.scene = builder.scene
        self.extensionsUsed = list(builder.extensionsUsed or ())
        self.extensionsRequired = list(builder.extensionsRequired or ())

    def __repr__(self) -> str:
        return std_repr(self, (
            'builder',
        ))