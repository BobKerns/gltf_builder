'''
Wrapper for textures in glTF.
This module provides the `_Texture` class, which represents a texture in a glTF file.
'''

from typing import Any
import pygltflib as gltf

from gltf_builder.compiler import _Scope, _CompileState
from gltf_builder.core_types import Phase
from gltf_builder.elements import BImage, BSampler, BTexture
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.utils import std_repr


class _Texture(BTexture):
    '''
    Builder representation of a glTF Texture
    '''

    def __init__(self,
                    name: str='', /, *,
                    sampler: BSampler,
                    source: BImage,
                    extras: dict|None=None,
                    extensions: dict|None=None,
                    ):
            super().__init__(
                name=name,
                extras=extras,
                extensions=extensions)
            self.sampler = sampler
            self.source = source

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            sampler=self.sampler,
            source=self.source,
        )


    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _CompileState,
                    /):
        match phase:
            case Phase.COLLECT:
                return (
                    s.compile(builder, scope, phase)
                    for s in (
                        self.sampler,
                        self.source,
                    )
                )
            case Phase.BUILD:
                return gltf.Texture(
                    name=self.name,
                    sampler=self.sampler._index,
                    source=self.source._index,
                    extras=self.extras,
                    extensions=self.extensions,
                )
    
    def __repr__(self):
        return std_repr(self, (
            'sampler',
            'source',
        ))
            
def texture(
    name: str='', /, *,
    sampler: BSampler,
    source: BImage,
    extras: dict|None=None,
    extensions: dict|None=None,
):
    '''
    Create a texture for glTF.
    
    Args:
        name (str): Name of the texture.
        sampler (BSampler): Sampler for the texture.
        source (BImage): Source image for the texture.
        extras (dict|None): Extra data for the texture.
        extensions (dict|None): Extensions for the texture.
    
    Returns:
        _Texture: A texture object.
    '''
    return _Texture(
        name,
        sampler=sampler,
        source=source,
        extras=extras,
        extensions=extensions,
    )

