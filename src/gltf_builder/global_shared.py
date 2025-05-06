'''
Configuration global to the glTF, and shared by `Builder` and `GlobalState`.

This allows adding and retrieving global data that is not specific to a single
node or other entity.
'''


from abc import abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from gltf_builder.attribute_types import BTYPE, AttributeData, Vector3Spec
from gltf_builder.compiler import _GLTF, _STATE, _Compilable, _CompileState
from gltf_builder.core_types import BufferViewTarget, ComponentType, ElementType, ExtensionsData, ExtrasData, IndexSize, NPTypes, EntityType
from gltf_builder.holders import _RO_Holder, _Holder
from gltf_builder.matrix import Matrix4
from gltf_builder.protocols import _BNodeContainerProtocol, AttributeType
from gltf_builder.quaternions import QuaternionSpec
from gltf_builder.accessors import _Accessor
from gltf_builder.assets import _Asset
from gltf_builder.buffers import _Buffer
from gltf_builder.cameras import _Camera
from gltf_builder.images import _Image
from gltf_builder.materials import _Material
from gltf_builder.meshes import _Mesh
from gltf_builder.nodes import _Node
from gltf_builder.samplers import _Sampler
from gltf_builder.scenes import _Scene
from gltf_builder.skins import _Skin
from gltf_builder.textures import _Texture
from gltf_builder.views import _BufferView
if TYPE_CHECKING:
    from gltf_builder.extensions import Extension
    from gltf_builder.entities import (
        BAccessor, BAsset, BBuffer, BBufferView, BCamera, BImage, BMaterial, BMesh, BNode, BSampler,
        BScene, BTexture, Entity, BSkin,
    )


class _GlobalShared(_BNodeContainerProtocol):
    '''
    Protocol for the global configuration of the glTF file.
    '''
    @property
    @abstractmethod
    def asset(self) -> Optional['BAsset']:
        '''
        The asset information for the glTF file.
        '''
        ...

    __meshes: _Holder['BMesh']
    @property
    def meshes(self) -> _RO_Holder['BMesh']:
        '''
        The meshes in the glTF file.
        '''
        return self.__meshes

    __cameras: _Holder['BCamera']
    @property
    def cameras(self) -> _RO_Holder['BCamera']:
        '''
        The cameras in the glTF file.
        '''
        return self.__cameras

    __images: _Holder['BImage']
    @property
    def images(self) -> _RO_Holder['BImage']:
        '''
        The images in the glTF file.
        '''
        return self.__images

    __materials: _Holder['BMaterial']
    @property
    def materials(self) -> _RO_Holder['BMaterial']:
        '''
        The materials in the glTF file.
        '''
        return self.__materials

    __nodes: _Holder['BNode']
    @property
    def nodes(self) -> _RO_Holder['BNode']:
        '''
        The nodes in the glTF file.
        '''
        return self.__nodes

    __samplers: _Holder['BSampler']
    @property
    def samplers(self) -> _RO_Holder['BSampler']:
        '''
        The samplers in the glTF file.
        '''
        return self.__samplers

    __scenes: _Holder['BScene']
    @property
    def scenes(self) -> _RO_Holder['BScene']:
        '''
        The scenes in the glTF file.
        '''
        return self.__scenes

    @property
    @abstractmethod
    def scene(self) -> Optional['BScene']:
        '''
        The initial scene.
        '''

    __skins: _Holder['BSkin']
    @property
    def skins(self) -> _RO_Holder['BSkin']:
        '''
        The skins in the glTF File
        '''
        return self.__skins

    __textures: _Holder['BTexture']
    @property
    def textures(self) -> _RO_Holder['BTexture']:
        '''
        The textures in the glTF file.
        '''
        return self.__textures

    __extension_objects: _Holder['Extension']
    @property
    @abstractmethod
    def extension_objects(self) -> _RO_Holder['Extension']:
        '''
        The extension objects for the glTF file.
        '''
        return self.__extension_objects

    __buffers: _Holder['BBuffer']
    @property
    def buffers(self) -> _RO_Holder['BBuffer']:
        '''
        The buffers in the glTF file.
    '''
        return self.__buffers

    __buffer_views: _Holder['BBufferView']
    @property
    def views(self) -> _RO_Holder['BBufferView']:
        '''
        The buffer views in the glTF file.
    '''
        return self.__buffer_views

    __accessors: _Holder['BAccessor[NPTypes, AttributeData]']
    @property
    def accessors(self) -> _RO_Holder['BAccessor[NPTypes, AttributeData]']:
        '''
        The accessors in the glTF file.
    '''
        return self.__accessors

    extras: dict[str, Any]
    '''
    The extras for the glTF file.
    '''
    extensions: dict[str, Any]
    '''
    The extensions for the glTF file.
    '''
    extensionsUsed: set[str]
    '''
    The extensions used in this file
    '''
    extensionsRequired: set[str]
    '''
    The extensions required to load this file.
    '''

    @property
    @abstractmethod
    def index_size(self) -> IndexSize:
        '''
        The size of the index buffer.
        '''
        ...

    @abstractmethod
    def get_attribute_type(self, name: str) -> AttributeType:
        ...

    @abstractmethod
    def instantiate(self, node_or_mesh: 'BNode|BMesh', /,
                    name: str='',
                    translation: Optional[Vector3Spec]=None,
                    rotation: Optional[QuaternionSpec]=None,
                    scale: Optional[Vector3Spec]=None,
                    matrix: Optional[Matrix4]=None,
                    extras: Optional[ExtrasData]=None,
                    extensions: Optional[ExtensionsData]=None,
                ) -> 'BNode':
        '''
        Instantiate a node or mesh with the given parameters.
        PARAMETERS
        ----------
        node_or_mesh: BNode|BMesh
            The node or mesh to instantiate.
        name: str
            The name of the node.
        translation: Vector3Spec
            The translation of the node.
        rotation: QuaternionSpec
            The rotation of the node.
        scale: Vector3Spec
            The scale of the node.
        matrix: Matrix4
            The transformation matrix of the node.
        extras: JsonObject
            Extra data for the node.
        extensions: JsonObject
            Extensions for the node.
        RETURNS
        -------
        BNode
            The instantiated node.
        '''
    def __init__(self):
        import gltf_builder.entities as elt
        import gltf_builder.extensions as ext
        self.__meshes = _Holder(elt.BMesh)
        self.__cameras = _Holder(elt.BCamera)
        self.__images = _Holder(elt.BImage)
        self.__materials = _Holder(elt.BMaterial)
        self.__nodes = _Holder(elt.BNode)
        self.__samplers = _Holder(elt.BSampler)
        self.__scenes = _Holder(elt.BScene)
        self.__skins = _Holder(elt.BSkin)
        self.__textures = _Holder(elt.BTexture)
        self.__extension_objects = _Holder(ext.Extension)
        self.__buffers = _Holder(elt.BBuffer)
        self.__buffer_views = _Holder(elt.BBufferView)
        self.__accessors = _Holder(elt.BAccessor)
        self.extras = {}
        self.extensions = {}
        self.extensionsUsed = set()
        self.extensionsRequired = set()
        global Extension
        from gltf_builder.extensions import Extension

    def add(self, elt: 'Entity') -> None:
        '''
        Add an entity to the global state.
        '''
        match elt:
            # In rough order of frequency of use
            case _Accessor():
                self.__accessors.add(elt)
            case _BufferView():
                self.__buffer_views.add(elt)
            case _Node():
                self.__nodes.add(elt)
            case _Mesh():
                self.__meshes.add(elt)
            case _Camera():
                self.__cameras.add(elt)
            case _Image():
                self.__images.add(elt)
            case _Material():
                self.__materials.add(elt)
            case _Sampler():
                self.__samplers.add(elt)
            case _Scene():
                self.__scenes.add(elt)
            case _Skin():
                self.__skins.add(elt)
            case _Texture():
                self.__textures.add(elt)
            case Extension():
                self.__extension_objects.add(elt)
            case _Buffer():
                self.__buffers.add(elt)
            case _Asset():
                if self.asset is not None:
                    raise ValueError('Asset already set')
                self.__asset = elt


class _GlobalSharedState(_GlobalShared):
    '''
    Protocol for the current configuration of the glTF file.
    This is used by the compiler to keep track of the current state of the,
    beyond the global state.
    '''

    @property
    @abstractmethod
    def buffer(self) -> 'BBuffer':
        '''
        The main buffer for the glTF file.
        '''
        ...

    _states: dict[int, _CompileState]
    '''
    The per-entity states for the compilation of the glTF file.
    '''

    @abstractmethod
    def _get_index_size(self, max_value: int) -> IndexSize:
        ...

    @abstractmethod
    def _gen_name(self,
                  obj: _Compilable[_GLTF, _STATE], /, *,
                  prefix: str='',
                  entity_type: EntityType|None=None,
                  index: Optional[int]=None,
                  suffix: str=''
                  ) -> str:
        '''
        Generate a name for an object according to the current `NameMode` policy.

        PARAMETERS
        ----------
        obj: Entity
            The object to generate a name for.
        gen_prefix: str|object
            The prefix to use for the generated name.
            If the prefix is an object, its `__class__.__name__` will be used.
        '''
        ...

    @abstractmethod
    def _create_accessor(self,
                elementType: ElementType,
                componentType: ComponentType,
                btype: type[BTYPE],
                name: str='',
                normalized: bool=False,
                buffer: Optional['BBuffer']=None,
                count: int=0,
                target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> 'BAccessor[NPTypes, BTYPE]':
        '''
        Create a `BAccessor` for the given element type and component type.
        PARAMETERS
        ----------
        elementType: ElementType
            The element type for the accessor.
        componentType: ComponentType
            The component type for the accessor.
        btype: type[BTYPE]
            The type of the accessor data.
        name: str
            The name of the accessor.
        normalized: bool
            Whether the accessor data is normalized.
        target: BufferViewTarget
            The target for the buffer view.
        RETURNS
        -------
        BAccessor[NPTypes, BTYPE]
            The created accessor.
        '''
        ...

    @abstractmethod
    def state(self, elt: 'Entity[_GLTF, _STATE]') -> _STATE:
        '''
        Get the state for the given entity.
        '''
        ...

    def idx(self, elt: 'Entity[_GLTF, _STATE]') -> int:
        '''
        Get the index of the given entity.
        '''
        return self.state(elt).index

    def __init__(self):
        super().__init__()
        self._states = {}
