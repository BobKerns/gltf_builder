'''
Wrapper for textures in glTF.
This module provides the `_Texture` class, which represents a texture in a glTF file.
'''

from typing import Any, TYPE_CHECKING

import pygltflib as gltf

from gltf_builder.compiler import _CompileState
from gltf_builder.core_types import Phase
from gltf_builder.elements import BImage, BSampler, BTexture
from gltf_builder.utils import std_repr
if TYPE_CHECKING:
    from gltf_builder.global_state import GlobalState


class _TextureState(_CompileState[gltf.Texture, '_TextureState', '_Texture']):
    '''
    State for the compilation of a texture.
    '''
    pass


class _Texture(BTexture):
    '''
    Builder representation of a glTF Texture
    '''

    @classmethod
    def state_type(cls):
        return _TextureState

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
                    globl: 'GlobalState',
                    phase: Phase,
                    state: _TextureState,
                    /):
        match phase:
            case Phase.COLLECT:
                return (
                    s.compile(globl, phase)
                    for s in (
                        self.sampler,
                        self.source,
                    )
                )
            case Phase.BUILD:
                return gltf.Texture(
                    name=self.name,
                    sampler=globl.idx(self.sampler),
                    source=globl.idx(self.source),
                    extras=self.extras,
                    extensions=self.extensions,
                )

    def __repr__(self):
        return std_repr(self, (
            'name',
            ('sampler', self.sampler.name or id(self.sampler)),
            ('source', self.source.name or id(self.source)),
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

