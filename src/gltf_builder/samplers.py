'''
Texture samplers for glTF.
'''

from typing import Any, Optional

import pygltflib as gltf

from gltf_builder.compile import _Scope, _CompileStates
from gltf_builder.core_types import MagFilter, MinFilter, Phase, WrapMode
from gltf_builder.elements import BSampler
from gltf_builder.protocols import _BuilderProtocol
from gltf_builder.utils import std_repr


class _Sampler(BSampler):
    '''
    Builder representation of a glTF Sampler
    '''
    def __init__(self,
                 name: str='', /, *,
                 magFilter: Optional[MagFilter]=None,
                 minFilter: Optional[MinFilter]=None,
                 wrapS: Optional[WrapMode]=None,
                 wrapT: Optional[WrapMode]=None,
                 extras=None,
                 extensions=None,
                ):
        super().__init__(
            name=name,
            extras=extras,
            extensions=extensions)
        self.magFilter=magFilter
        self.minFilter=minFilter
        self.wrapS=wrapS
        self.wrapT=wrapT

    def _clone_attributes(self) -> dict[str, Any]:
        return dict(
            magFilter=self.magFilter,
            minFilter=self.minFilter,
            wrapS=self.wrapS,
            wrapT=self.wrapT,
        )

    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    states: _CompileStates,
                    /):
        match phase:
            case Phase.BUILD:
                return gltf.Sampler(
                    magFilter=self.magFilter,
                    minFilter=self.minFilter,
                    wrapS=self.wrapS,
                    wrapT=self.wrapT,
                    extras=self.extras,
                    extensions=self.extensions,
                )
            
    def __repr__(self):
        return std_repr(self, (
            'magFilter',
            'minFilter',
            'wrapS',
            'wrapT',
        ))


def sampler(
    name: str='', /, *,
    magFilter: Optional[MagFilter]=None,
    minFilter: Optional[MinFilter]=None,
    wrapS: Optional[WrapMode]=None,
    wrapT: Optional[WrapMode]=None,
    extras=None,
    extensions=None,
):
    '''
    Create a sampler for a texture.
    
    Parameters
    ----------
    builder : _BuilderProtocol
        The builder instance.
    name : str, optional
        The name of the sampler.
    magFilter : MagFilter, optional
        The magnification filter.
    minFilter : MinFilter, optional
        The minification filter.
    wrapS : WrapMode, optional
        The wrap mode for the S axis.
    wrapT : WrapMode, optional
        The wrap mode for the T axis.
    extras : dict, optional
        Extra properties for the sampler.
    extensions : dict, optional
        Extensions for the sampler.

    Returns
    -------
    _Sampler
        A new sampler instance.
    '''
    return _Sampler(
        name,
        magFilter=magFilter,
        minFilter=minFilter,
        wrapS=wrapS,
        wrapT=wrapT,
        extras=extras,
        extensions=extensions,
        )

