'''
Configuration global to the glTF, and shared by `Builder` and `GlobalState`.

This allows adding and retrieving global data that is not specific to a single
node or other element.
'''


from abc import abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from gltf_builder.attribute_types import BTYPE, AttributeData, Vector3Spec
from gltf_builder.compiler import _GLTF, _STATE, _Compilable, _CompileState
from gltf_builder.core_types import BufferViewTarget, ComponentType, ElementType, ExtensionsData, ExtrasData, IndexSize, NPTypes, ScopeName
from gltf_builder.holders import _Holder
from gltf_builder.matrix import Matrix4
from gltf_builder.protocols import _BNodeContainerProtocol, AttributeType
from gltf_builder.quaternions import QuaternionSpec
if TYPE_CHECKING:
    from gltf_builder.extensions import Extension
    from gltf_builder.elements import (
        BAccessor, BAsset, BBuffer, BBufferView, BCamera, BImage, BMaterial, BMesh, BNode, BSampler,
        BScene, BTexture, Element, BSkin,
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
    def meshes(self) -> _Holder['BMesh']:
        '''
        The meshes in the glTF file.
        '''
        return self.__meshes

    __cameras: _Holder['BCamera']
    @property
    def cameras(self) -> _Holder['BCamera']:
        '''
        The cameras in the glTF file.
        '''
        return self.__cameras

    __images: _Holder['BImage']
    @property
    def images(self) -> _Holder['BImage']:
        '''
        The images in the glTF file.
        '''
        return self.__images

    __materials: _Holder['BMaterial']
    @property
    def materials(self) -> _Holder['BMaterial']:
        '''
        The materials in the glTF file.
        '''
        return self.__materials

    __nodes: _Holder['BNode']
    @property
    def nodes(self) -> _Holder['BNode']:
        '''
        The nodes in the glTF file.
        '''
        return self.__nodes

    __samplers: _Holder['BSampler']
    @property
    def samplers(self) -> _Holder['BSampler']:
        '''
        The samplers in the glTF file.
        '''
        return self.__samplers

    __scenes: _Holder['BScene']
    @property
    def scenes(self) -> _Holder['BScene']:
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
    def skins(self) -> _Holder['BSkin']:
        '''
        The skins in the glTF File
        '''
        return self.__skins

    __textures: _Holder['BTexture']
    @property
    def textures(self) -> _Holder['BTexture']:
        '''
        The textures in the glTF file.
        '''
        return self.__textures

    __extension_objects: _Holder['Extension']
    @property
    @abstractmethod
    def extension_objects(self) -> _Holder['Extension']:
        '''
        The extension objects for the glTF file.
        '''
        return self.__extension_objects

    __buffers: _Holder['BBuffer']
    @property
    def buffers(self) -> _Holder['BBuffer']:
        '''
        The buffers in the glTF file.
    '''
        return self.__buffers

    __buffer_views: _Holder['BBufferView']
    @property
    def views(self) -> _Holder['BBufferView']:
        '''
        The buffer views in the glTF file.
    '''
        return self.__buffer_views

    __accessors: _Holder['BAccessor[NPTypes, AttributeData]']
    @property
    def accessors(self) -> _Holder['BAccessor[NPTypes, AttributeData]']:
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
        import gltf_builder.elements as elt
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


class _CurrentGlobalShared(_GlobalShared):
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
    The per-element states for the compilation of the glTF file.
    '''

    @abstractmethod
    def _get_index_size(self, max_value: int) -> IndexSize:
        ...

    @abstractmethod
    def _gen_name(self,
                  obj: _Compilable[_GLTF, _STATE], /, *,
                  prefix: str='',
                  scope: ScopeName|None=None,
                  index: Optional[int]=None,
                  suffix: str=''
                  ) -> str:
        '''
        Generate a name for an object according to the current `NameMode` policy.

        PARAMETERS
        ----------
        obj: Element
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
    def state(self, elt: 'Element[_GLTF, _STATE]') -> _STATE:
        '''
        Get the state for the given element.
        '''
        ...

    def idx(self, elt: 'Element[_GLTF, _STATE]') -> int:
        '''
        Get the index of the given element.
        '''
        return self.state(elt).index

    def __init__(self):
        super().__init__()
        self._states = {}
