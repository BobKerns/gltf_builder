'''
Implementation of the Material class and its related classes for glTF Builder.
'''

from typing import Optional, cast

import pygltflib as gltf

from gltf_builder.compiler import _CompileState, _Scope, _DoCompileReturn, _ReturnCollect
from gltf_builder.core_types import AlphaMode, JsonObject, Phase
from gltf_builder.elements import BMaterial, BTexture
from gltf_builder.protocols import _BuilderProtocol


class _MaterialState(_CompileState[gltf.Material, '_MaterialState']):
    '''
    State for the compilation of a material.
    '''
    pass

class _Material(BMaterial):
    '''
    Implementation class for `BMaterial`.
    '''

    @classmethod
    def state_type(cls):
        return _MaterialState

    def __init__(self,
                 name: str='',
                 /,
                baseColorFactor: Optional[tuple[float, float, float, float]]=None,
                baseColorTexture: Optional[BTexture]=None,
                metallicRoughnessTexture: Optional[BTexture]=None,
                metallicFactor: Optional[float]=None,
                roughnessFactor: Optional[float]=None,
                normalTexture: Optional[BTexture]=None,
                normalScale: Optional[float]=None,
                occlusionTexture: Optional[BTexture]=None,
                occlusionStrength: Optional[float]=None,
                emissiveTexture: Optional[BTexture]=None,
                emissiveFactor: Optional[tuple[float, float, float]]=None,
                alphaMode: AlphaMode=AlphaMode.OPAQUE,
                alphaCutoff: Optional[float]=None,
                doubleSided: bool=False,
                 extras: Optional[JsonObject]=None,
                 extensions: Optional[JsonObject]=None,
                ):
        super().__init__(name, extras, extensions)
        self.baseColorFactor = baseColorFactor
        self.baseColorTexture = baseColorTexture
        self.metallicRoughnessTexture = metallicRoughnessTexture
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.normalTexture = normalTexture
        self.normalScale = normalScale
        self.occlusionTexture = occlusionTexture
        self.occlusionStrength = occlusionStrength
        self.emissiveTexture = emissiveTexture
        self.emissiveFactor = emissiveFactor
        self.alphaMode = alphaMode
        self.alphaCutoff = alphaCutoff
        self.doubleSided = doubleSided

    def _do_compile(self,
                    builder: _BuilderProtocol,
                    scope: _Scope,
                    phase: Phase,
                    state: _MaterialState) -> _DoCompileReturn[gltf.Material]:
        '''
        Compile the material.
        '''
        match phase:
            case Phase.COLLECT:
                textures = [
                    t for t in (
                        self.baseColorTexture,
                        self.metallicRoughnessTexture,
                        self.normalTexture,
                        self.occlusionTexture,
                        self.emissiveTexture,
                    )
                    if t
                ]
                for t in textures:
                    t.compile(builder, scope, phase)
                return cast(_ReturnCollect, textures)
            case Phase.BUILD:
                return gltf.Material(
                    name=self.name,
                    pbrMetallicRoughness=gltf.PbrMetallicRoughness(
                        baseColorFactor=list(self.baseColorFactor) if self.baseColorFactor else None,
                        baseColorTexture=gltf.TextureInfo(
                            index=self.baseColorTexture._index
                        ) if self.baseColorTexture else None,
                        metallicFactor=self.metallicFactor,
                        roughnessFactor=self.roughnessFactor,
                        metallicRoughnessTexture=gltf.TextureInfo(
                            index=self.metallicRoughnessTexture._index,
                        ) if self.metallicRoughnessTexture else None,
                    ),
                    normalTexture=gltf.NormalMaterialTexture(
                        index=self.normalTexture._index,
                        scale = self.normalScale or 1.0,
                    ) if self.normalTexture else None,
                    occlusionTexture=gltf.OcclusionTextureInfo(
                        index=self.occlusionTexture._index,
                        strength=self.occlusionStrength,
                    ) if self.occlusionTexture else None,
                    emissiveTexture=gltf.TextureInfo(
                        index=self.emissiveTexture._index,
                     ) if self.emissiveTexture else None,
                    emissiveFactor=list(self.emissiveFactor) if self.emissiveFactor else None,
                    alphaMode=self.alphaMode.value,
                    alphaCutoff=self.alphaCutoff,
                    doubleSided=self.doubleSided,
                )

def material(
    name: str='',
    baseColorFactor: Optional[tuple[float, float, float, float]]=None,
    baseColorTexture: Optional[BTexture]=None,
    metallicRoughnessTexture: Optional[BTexture]=None,
    metallicFactor: Optional[float]=None,
    roughnessFactor: Optional[float]=None,
    normalTexture: Optional[BTexture]=None,
    normalScale: Optional[float]=None,
    occlusionTexture: Optional[BTexture]=None,
    occlusionStrength: Optional[float]=None,
    emissiveTexture: Optional[BTexture]=None,
    emissiveFactor: Optional[tuple[float, float, float]]=None,
    alphaMode: AlphaMode=AlphaMode.OPAQUE,
    alphaCutoff: Optional[float]=0.5,
    doubleSided: bool=False,
    extras: Optional[JsonObject]=None,
    extensions: Optional[JsonObject]=None,
) -> _Material:
    '''
    Create a new material.

    Parameters
    ----------
    name : str, optional
        The name of the material.
    baseColorFactor : tuple[float, float, float, float], optional
        The base color factor of the material.
    baseColorTexture : BTexture, optional
        The base color texture of the material.
    metallicRoughnessTexture : BTexture, optional
        The metallic roughness texture of the material.
    metallicFactor : float, optional
        The metallic factor of the material.
    roughnessFactor : float, optional
        The roughness factor of the material.
    normalTexture : BTexture, optional
        The normal texture of the material.
    normalScale : float, optional
        The normal scale of the material.
    occlusionTexture : BTexture, optional
        The occlusion texture of the material.
    occlusionStrength : float, optional
        The occlusion strength of the material.
    emissiveTexture : BTexture, optional
        The emissive texture of the material.
    emissiveFactor : tuple[float, float, float], optional
        The emissive factor of the material.
    alphaMode : AlphaMode, optional
        The alpha mode of the material.
    alphaCutoff : float, optional
        The alpha cutoff of the material.
    doubleSided : bool, optional
        Whether the material is double-sided.
    extras : JsonObject, optional
        Application-specific data.
    extensions : JsonObject, optional
        Application-specific data.

    Returns
    -------
    _Material
        A new material instance.
    '''
    return _Material(name,
                    baseColorFactor=baseColorFactor,
                    baseColorTexture=baseColorTexture,
                    metallicRoughnessTexture=metallicRoughnessTexture,
                    metallicFactor=metallicFactor,
                    roughnessFactor=roughnessFactor,
                    normalTexture=normalTexture,
                    normalScale=normalScale,
                    occlusionTexture=occlusionTexture,
                    occlusionStrength=occlusionStrength,
                    emissiveTexture=emissiveTexture,
                    emissiveFactor=emissiveFactor,
                    alphaMode=alphaMode,
                    alphaCutoff=alphaCutoff,
                    doubleSided=doubleSided,
                    extras=extras,
                    extensions=extensions,
                     )