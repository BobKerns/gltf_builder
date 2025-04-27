'''
global compilation state
'''

from typing import Optional, TYPE_CHECKING
from itertools import count

from gltf_builder.accessors import _Accessor
from gltf_builder.attribute_types import BTYPE
from gltf_builder.compiler import _GLTF, _STATE, _BaseCompileState, _Compileable
from gltf_builder.core_types import BufferViewTarget, ComponentType, ElementType, IndexSize, NPTypes, NameMode, ScopeName
from gltf_builder.elements import BAccessor, BBuffer, BNode, Element
from gltf_builder.nodes import _BNodeContainer
from gltf_builder.protocols import  _BuilderProtocol, _Scope, AttributeType
from gltf_builder.utils import decode_dtype, std_repr
if TYPE_CHECKING:
    from gltf_builder.builder import Builder

class _GlobalState(_BNodeContainer, _BuilderProtocol):
    __builder: 'Builder'
    _scope_name: ScopeName = ScopeName.BUILDER
    _id_counters: dict[str, count]
    _states: dict[int, _BaseCompileState] = {}

    @property
    def builder(self) -> 'Builder':
        return self.__builder
    
    @property
    def buffer(self) -> BBuffer:
        '''
        The default buffer for the glTF document.
        '''
        return self.builder.buffer
    
    @property
    def index_size(self) -> IndexSize:
        '''
        The size of the index type for the glTF document.
        '''
        return self.builder.index_size
    
    def get_attribute_type(self, name: str) -> AttributeType:
        '''
        Get the attribute type for the given name.
        '''
        return self.builder.get_attribute_type(name)

    '''
    Global state for the compilation of a glTF document.
    '''
    def __init__(self, builder: 'Builder') -> None:
        super().__init__(builder.nodes)
        self.__builder = builder

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
        self.extras = builder.extras or {}
        self.extensions = builder.extensions or {}
        self.scene = builder.scene
        self.extensionsUsed = list(builder.extensionsUsed or ())
        self.extensionsRequired = list(builder.extensionsRequired or ())
        self._id_counters = {}

    def _get_index_size(self, max_value):
        return self.builder._get_index_size(max_value)
    

    __names: set[str] = set()

    def _gen_name(self,
                  obj: _Compileable[_GLTF, _STATE], /, *,
                  prefix: str|object='',
                  scope: ScopeName|None=None,
                  index: int|None=None,
                  suffix: str|None=None,
                  ) -> str:
        '''
        Generate a name according to the current name mode policy
        '''
        scope = scope or obj._scope_name
        def get_count(obj: object) -> int:
            tname = type(obj).__name__[1:]
            counters = self._id_counters
            if tname not in counters:
                counters[tname] = count()
            return next(counters[tname])
        
        def gen(obj: _Compileable[_GLTF, _STATE]) -> str:
            nonlocal prefix, suffix
            name_mode = self.builder.name_policy[scope]
            match obj:
                case Element() if obj.name and name_mode != NameMode.UNIQUE:
                    # Increment the count anyway for stability.
                    # Naming one node should not affect the naming of another.
                    get_count(obj)
                    return obj.name
                case _:
                    if prefix == '':
                        prefix = type(obj).__name__[1:]
                    else:
                        prefix = prefix
                    suffix = suffix or ''
                    if index is not None:
                        suffix = f'{suffix}[{index}]'
                    return f'{prefix}{get_count(obj)}{suffix}'
        
        def register(name: object|None) -> str:
            match name:
                case str():
                    name = name.strip()
                case Element():
                    name = name.name.strip()
                case _:
                    raise ValueError(f'Invalid name: {name}')
            if not name:
                return ''
            self.__names.add(name)
            return name
        name_mode = self.builder.name_policy[scope]
        match name_mode:
            case NameMode.AUTO:
                return register(gen(obj))
            case NameMode.MANUAL:
                return register(obj)
            case NameMode.UNIQUE:
                name = gen(obj)
                while obj in self.__names:
                    name = gen(obj)
                return register(name)
            case NameMode.NONE:
                return ''
            case _:
                raise ValueError(f'Invalid name mode: {self.name_mode}') # pragma: no cover

    def _create_accessor(self,
                elementType: ElementType,
                componentType: ComponentType,
                btype: type[BTYPE],
                name: str='',
                normalized: bool=False,
                buffer: Optional['BBuffer']=None,
                count: int=0,
                target: BufferViewTarget=BufferViewTarget.ARRAY_BUFFER,
                ) -> BAccessor[NPTypes, BTYPE]:
            dtype = decode_dtype(elementType, componentType)
            return _Accessor(
                elementType=elementType,
                componentType=componentType,
                btype=btype,
                buffer=buffer or self._buffers[0],
                name=name,
                dtype=dtype,
                count=count,
                normalized=normalized,
                target=target,
            )

    def __repr__(self) -> str:
        return std_repr(self, (
            'builder',
        ))