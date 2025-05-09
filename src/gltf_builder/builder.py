'''
The initial object that collects the geometry info and compiles it into
a glTF object.
'''

from collections.abc import Iterable, Mapping
from typing import Optional
from pathlib import Path
import re

import pygltflib as gltf

from gltf_builder.attribute_types import (
    BTYPE, Color, Joint, Point, Tangent,
    UvPoint, Vector3, Vector3Spec, Weight,
    color, point, tangent, uv, vector3,
)
from gltf_builder.core_types import (
     ExtensionsData, ExtrasData, ImageType, IndexSize, JsonObject,
     NameMode, NamePolicy,
     ElementType, ComponentType, EntityType,
)
from gltf_builder.extensions import Extension, load_extensions
from gltf_builder.global_state import GlobalState
from gltf_builder.holders import _Holder
from gltf_builder.buffers import _Buffer
from gltf_builder.matrix import Matrix4Spec
from gltf_builder.quaternions import QuaternionSpec
from gltf_builder.meshes import _Mesh
from gltf_builder.nodes import _BNodeContainer
from gltf_builder.images import _Image

from gltf_builder.protocols import (
    _AttributeParser,  AttributeType,
)
from gltf_builder.entities import (
     BAsset, BBuffer, BCamera, BImage, BMaterial,
     BMesh, BNode, BPrimitive, BSampler, BScene, BSkin, BTexture,
)
from gltf_builder.global_shared import _GlobalShared
from gltf_builder.log import GLTF_LOG
from gltf_builder.utils import std_repr, count_iter

LOG = GLTF_LOG.getChild(Path(__file__).stem)

DEFAULT_NAME_MODE = NameMode.AUTO
DEFAULT_NAME_POLICY: NamePolicy = {
    EntityType.NODE: NameMode.AUTO,
    EntityType.MESH: NameMode.AUTO,
    EntityType.PRIMITIVE: NameMode.AUTO,
    EntityType.ASSET: NameMode.MANUAL,
    EntityType.ACCESSOR: NameMode.NONE,
    EntityType.ACCESSOR_INDEX: NameMode.NONE,
    EntityType.BUFFER: NameMode.NONE,
    EntityType.BUFFER_VIEW: NameMode.NONE,
    EntityType.BUILDER: NameMode.NONE,
    EntityType.IMAGE: NameMode.AUTO,
    EntityType.MATERIAL: NameMode.AUTO,
    EntityType.TEXTURE: NameMode.AUTO,
    EntityType.SAMPLER: NameMode.AUTO,
    EntityType.CAMERA: NameMode.AUTO,
    EntityType.SKIN: NameMode.AUTO,
    EntityType.SCENE: NameMode.AUTO,
    EntityType.ANIMATION: NameMode.AUTO,
    EntityType.ANIMATION_SAMPLER: NameMode.AUTO,
    EntityType.ANIMATION_CHANNEL: NameMode.AUTO,
    EntityType.EXTENSION: NameMode.MANUAL,
}
'''
Default naming mode for each entity type.
'''

_RE_ATTRIB_NAME = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?:_\d+)$')


class Builder(_BNodeContainer, _GlobalShared):
    '''
    The global input state for the compilation of the glTF file.
    This is used to store the global state of the compilation process.

    While it is temporarily stored in the builder, it will be allocated
    on each build.
    '''

    _entity_type = EntityType.BUILDER

    __asset: Optional[BAsset]
    @property
    def asset(self) -> Optional[BAsset]:
        '''
        The asset information for the glTF file.
        '''
        return self.__asset
    @asset.setter
    def asset(self, asset: Optional[BAsset], /):
        if asset is not None:
            if not isinstance(asset, BAsset):
                raise TypeError(f'Asset must be of type BAsset, not {type(asset)}')
        self.__asset = asset

    __scene: Optional[BScene]
    @property
    def scene(self) -> Optional[BScene]:
        '''
        The scene to use as the default scene for the glTF file.
        '''
        return self.__scene
    @scene.setter
    def scene(self, scene: Optional[BScene], /):
        if scene is not None:
            if not isinstance(scene, BScene):
                raise TypeError(f'Scene must be of type BScene, not {type(scene)}')
        self.__scene = scene

    __extension_objects: _Holder[Extension]
    @property
    def extension_objects(self) -> _Holder[Extension]:
        '''
        The extensions used by the glTF file.
        '''
        return self.__extension_objects
    @extension_objects.setter
    def extension_objects(self, extensions: Iterable[Extension], /):
        if not isinstance(extensions, Iterable):
            raise TypeError(f'Extensions must be iterable, not {type(extensions)}')
        self.__extension_objects.add_from(extensions)

    __index_size: IndexSize = IndexSize.AUTO
    '''
    Number of bytes to use for indices. Default is NONE, meaning no index.

    This is used to determine the component type of the indices.

    A value of AUTO will use the smallest possible size for a particular
    mesh.

    A value of NONE will disable indexed geometry.

    This is only used when creating the indices buffer view.
    '''

    @property
    def index_size(self) ->  IndexSize:
        '''
        The number of bytes to use for indices.
        '''
        return self.__index_size

    @index_size.setter
    def index_size(self, size: IndexSize, /):
        self.__index_size = size

    attr_type_map: dict[str, AttributeType]
    '''
    The mapping of attribute names to their types.
    '''
    name_policy: dict[EntityType, NameMode]
    '''
    The mode for handling names, for each `ScopeName`

    AUTO: Automatically generate names for objects that do not have one.
    MANUAL: Use the name provided.
    UNIQUE: Ensure the name is unique.
    NONE: Do not use names.
    '''

    @property
    def buffer(self) -> 'BBuffer':
        '''
        The primary `BBuffer` for the glTF file.
        '''
        return self.buffers[0]

    def __init__(self, /,
                asset: Optional[BAsset]=None,
                cameras: Iterable[BCamera]=(),
                meshes: Iterable[BMesh]=(),
                images: Iterable[BImage]=(),
                materials: Iterable[BMaterial]=(),
                nodes: Iterable[BNode] =(),
                samplers: Iterable[BSampler]=(),
                skins: Iterable[BSkin]=(),
                scenes: Iterable[BScene]=(),
                textures: Iterable[BTexture]=(),
                buffers: Iterable[_Buffer]=(),
                extras: Optional[JsonObject]=None,
                extensions: Optional[JsonObject]=None,
                scene: Optional[BScene]=None,
                index_size: Optional[IndexSize]=None,
                name_policy: Mapping[EntityType, NameMode]|None = None,
                extensionsUsed: Optional[list[str]]=None,
                extensionsRequired: Optional[list[str]]=None,
        ):
        _BNodeContainer.__init__(self, nodes=nodes)
        _GlobalShared.__init__(self)
        self.__extension_objects = _Holder(Extension)

        name_policy = name_policy or {}
        self.name_policy = {
            entity_type: name_policy.get(entity_type, DEFAULT_NAME_POLICY[entity_type])
            for entity_type in EntityType
        }
        if index_size is None:
            index_size = IndexSize.NONE
        self.asset = asset
        self.index_size = IndexSize.NONE if index_size is None else index_size
        self.extras = extras or {}
        self.extensions = extensions or {}
        self.scene = scene
        self.attr_type_map = {}
        self.define_attrib('POSITION', ElementType.VEC3, ComponentType.FLOAT, Point, point)
        self.define_attrib('NORMAL', ElementType.VEC3, ComponentType.FLOAT, Vector3, vector3)
        self.define_attrib('COLOR', ElementType.VEC4, ComponentType.FLOAT, Color, color)
        self.define_attrib('TEXCOORD', ElementType.VEC2, ComponentType.FLOAT, UvPoint, uv)
        self.define_attrib('TANGENT', ElementType.VEC4, ComponentType.FLOAT, Tangent, tangent)
        self.define_attrib('JOINTS', ElementType.VEC4, ComponentType.UNSIGNED_SHORT, Joint)
        self.define_attrib('WEIGHTS', ElementType.VEC4, ComponentType.FLOAT, Weight)
        self._id_counters = {}

    def add_mesh(self,
                name: str='',
                primitives: Optional[Iterable[BPrimitive]]=None,
                weights: Optional[Iterable[float]]=None,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
                ):
        mesh = _Mesh(name,
                     primitives=primitives or (),
                     weights=weights or (),
                     extras=extras,
                     extensions=extensions,
        )
        return mesh


    def build(self, /,
            index_size: Optional[IndexSize]=None,
        ) -> gltf.GLTF2:
        load_extensions()
        if index_size is not None:
            self.index_size = index_size
        def flatten(node: BNode) -> Iterable[BNode]:
            yield node
            for n in node.children:
                yield from flatten(n)

        nodes = list({
            i
            for n in self.nodes
            for i in flatten(n)
        })
        # Add all the child nodes.
        self.nodes.add_from(n for n in nodes if not n.root)
        global_state = GlobalState(self)
        return global_state.build()

    def define_attrib(self,
                      name: str,
                      elementType: ElementType,
                      componentType: ComponentType,
                      type: type[BTYPE],
                      parser: _AttributeParser[BTYPE]|None=None,
                ):
        '''
        Define the type of an attribute.
        - TANGENT: VEC4/FLOAT/Tangent
        - TEXCOORD_0: VEC2/FLOAT
        - TEXCOORD_1: VEC2/FLOAT
        - COLOR_0: VEC4/FLOAT
        - JOINTS_0: VEC4/UNSIGNED_SHORT
        - WEIGHTS_0: VEC4/FLOAT
        '''
        self.attr_type_map[name] = AttributeType(name, elementType, componentType, type, parser)

    def get_attribute_type(self, name: str) -> AttributeType:
        '''
        Get the type information for an attribute by name.

        If the attribute is not defined, but ends in _<digits>,
        the suffix is stripped and the attribute is looked up again.
        If the attribute is still not found, a ValueError is raised.
        '''
        attr = self.attr_type_map.get(name)
        if attr:
            return attr
        m = _RE_ATTRIB_NAME.match(name)
        if m:
            attr = self.attr_type_map.get(m.group(1))
        if attr is None:
            raise ValueError(f'Attribute {name} is not defined.')
        self.attr_type_map[name] = attr
        return attr

    def _get_index_size(self, max_value: int) -> IndexSize:
        '''
        Get the index size based on the configured size or the maximum value.

        If the index size is set to other than AUTO or NONE,
        the configured size is validated against the maximum value and returned.

        If the index size is set to AUTO, the maximum value is used to determine
        the index size.
        '''
        match self.index_size:
            case IndexSize.UNSIGNED_INT:
                if max_value < 4294967295:
                    return IndexSize.UNSIGNED_INT
                raise ValueError("Index size is too large.")
            case IndexSize.UNSIGNED_SHORT:
                if max_value < 65535:
                    return IndexSize.UNSIGNED_SHORT
                return IndexSize.UNSIGNED_INT
            case IndexSize.UNSIGNED_BYTE:
                if max_value < 255:
                    return IndexSize.UNSIGNED_BYTE
                return IndexSize.UNSIGNED_SHORT
            case IndexSize.AUTO:
                if max_value < 0:
                    raise ValueError("Index size is negative.")
                if max_value < 255:
                    return IndexSize.UNSIGNED_BYTE
                if max_value < 65535:
                    return IndexSize.UNSIGNED_SHORT
                if max_value < 4294967295:
                    return IndexSize.UNSIGNED_INT
                # Unlikely!
                raise ValueError("Index size is too large.")
            case IndexSize.NONE:
                return IndexSize.NONE
            case _:
                raise ValueError(f'Invalid index size: {self.index_size}')

    def create_image(self,
                imageType: ImageType,
                name: str='',
                /, *,
                blob: Optional[bytes]=None,
                uri: Optional[str|Path]=None,
                extras: Optional[ExtrasData]=None,
                extensions: Optional[ExtensionsData]=None,
            ) -> BImage:
        '''
        Implementation of `BImage`.
        '''
        return _Image(
            name=name,
            blob=blob,
            uri=uri,
            imageType=imageType,
            extras=extras,
            extensions=extensions,
        )

    def instantiate(self, node_or_mesh: 'BNode|BMesh',
                    name: str='', /,
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4Spec]=None,
                    extras: Optional[ExtrasData]=None,
                    extensions: Optional[ExtensionsData]=None,
                ) -> 'BNode':
        node = super().instantiate(node_or_mesh,
                    name,
                    translation=translation,
                    rotation=rotation,
                    scale=scale,
                    matrix=matrix,
                    extras=extras,
                    extensions=extensions,
                )
        self.nodes.add(node)
        return node

    def __repr__(self):
        return std_repr(self, (
            ('cameras', count_iter(self.cameras)),
            ('meshes', count_iter(self.meshes)),
            ('images', count_iter(self.images)),
            ('materials', count_iter(self.materials)),
            ('nodes', count_iter(self.nodes)),
            ('samplers', count_iter(self.samplers)),
            ('skins', count_iter(self.skins)),
            ('scenes', count_iter(self.scenes)),
            ('textures', count_iter(self.textures)),
            ('buffers', count_iter(self.buffers)),
            'index_size',
        ))