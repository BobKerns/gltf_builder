'''
Types describing attribute values, as well as node properties such as scale.

For example, node translation uses a Vector3 for translation.

In general, these functions take 4 types of parameters:
- The float or integer values of the components of the type.
- A tuple of float or integer values.
- A numpy array of float or integer values.
- An object of the type being created.
'''

from abc import abstractmethod
from typing import Generic, NamedTuple, TypeAlias, Literal, TypeVar, cast, overload, Optional, Any, Self
from math import sqrt
from collections.abc import Generator, Iterable, Callable, Sequence, Mapping

import numpy as np

from gltf_builder.core_types import (
    IntScalar, Scalar, float01, ByteSize, ByteSizeAuto,
)

EPSILON = 1e-12
'''
A small value for floating point comparisons.
'''
EPSILON2 = EPSILON * EPSILON
'''
The square of `EPSILON` for efficient near-zero length tests.
'''


class _Arrayable(NamedTuple):
    '''
    A tuple of values that can be converted to a numpy array.
    '''
    def __array__(self,
                  dtype: np.dtype[Any] = np.dtype(np.float32),
                  copy: bool=False,
                ) -> (
                    np.ndarray[tuple[int, ...], np.dtype[np.float32]]
                    |np.ndarray[tuple[int, ...], np.dtype[np.int16]]
                    |np.ndarray[tuple[int, ...], np.dtype[np.int8]]
                ):
        return np.fromiter(self, dtype=dtype)


class _Floats2(NamedTuple):
    '''
    A tuple of two floats following the x,y,z,w naming convention.
    '''
    x: float
    y: float


class _Floats3(NamedTuple):
    '''
    A tuple of three floats following the x,y,z,w naming convention.
    '''
    x: float
    y: float
    z: float


class _Floats4(NamedTuple):
    '''
    A tuple of four floats following the x,y,z,w naming convention.
    '''
    x: float
    y: float
    z: float
    w: float


V = TypeVar('V', bound='Vector2|Vector3|Vector4|Tangent')
VT = TypeVar('VT', bound='_Float2Spec|_Float3Spec|_Float4Spec')
class VectorLike(NamedTuple, Generic[V, VT]):
    '''
    Types that directly support vector operations such as length, addition, and dot products.
    '''
    def __bool__(self) -> bool:
        return sum(v*v for v in self) >= EPSILON2
    
    @property
    def length(self):
        return sqrt(sum(v*v for v in self))
    
    def _t1(self, x: V) -> V:
        return x
    
    def _t2(self, x: VT) -> VT:
        return x

    def _other_vector(self, x: Scalar|V|VT) -> VT|None:
        match x:
            case VectorLike():
                return cast(VT, x)
            # Yeah, in len(x), x is partially unknown. It supports len and doesn't depend on the unknown!
            case tuple()|np.ndarray() if len(x) == len(self): # type: ignore
                vtype = _VTYPES[len(x) - 2] # type: ignore
                return cast(VT, vtype(*x))
            case _:
                return None
    
    def __add__(self, other: V|VT) -> Self: # type: ignore
        o = self._other_vector(other)
        if o is None:
            return NotImplemented
        return type(self)(*(a+b for a,b in zip(self, other)))
    
    def __sub__(self, other: V|VT) -> Self:
        o = self._other_vector(other)
        if o is None:
            return NotImplemented
        return type(self)(*(a-b for a,b in zip(self, other)))   
                                                                                                                                                                                                                                                                                                                                                                                                        
    def __neg__(self) -> Self:
        return type(self)(*(-a for a in self))
    
    def __mul__(self, other: Scalar|V|VT) -> Self|float: # type: ignore
        match other:
            case float()|np.floating()|int()|np.integer():
                other = float(other) # type: ignore
                return type(self)(*(a*other for a in self))
            case VectorLike()|tuple()|np.ndarray():
                o = self._other_vector(other)
                if o is None:
                    return NotImplemented
                return sum((a*b for a, b in zip(self, o)))
            case _:
                return NotImplemented

    def __rmul__(self, other: Scalar|V|VT) -> Self: # type: ignore
        return self.__mul__(other) # type: ignore

    def __truediv__(self, other: Scalar) -> Self:
        match other:
            case float()|np.floating()|int()|np.integer():
                return type(self)(*(a/other for a in self))
            case _:
                return NotImplemented
    
    def dot(self, other: V|VT) -> float:
        o = self._other_vector(other)
        if o is None:
            raise TypeError('Invalid vector dot product')
        return sum(a*b for a,b in zip(self, other))


VEC = TypeVar('VEC', bound='Vector2|Vector3')
class PointLike(NamedTuple, Generic[VEC]):
    '''
    Pointlike quantities have meaningful scalar distances
    '''
    @overload
    def __sub__(self: 'Point', other: 'Point') -> 'Vector3': ...
    @overload
    def __sub__(self: 'UvPoint', other: 'UvPoint') -> 'Vector2': ...
    @overload
    def __sub__(self: 'Point|UvPoint', other: 'Point|UvPoint') -> 'Vector2|Vector3': ...
    @overload
    def __sub__(self: 'PointLike[VEC]', other: 'PointLike[VEC]') -> VEC: ...
    @overload
    def __sub__(self: 'UvPoint', other: 'Vector2') -> 'Vector2': ...
    @overload
    def __sub__(self: 'Point', other: 'Vector3') -> 'Vector2': ...
    @overload
    def __sub__(self: 'Point|UvPoint', other: 'Vector3|Vector2') -> 'Vector2': ...
    @abstractmethod
    def __sub__(self: 'PointLike[VEC]|Point|UvPoint',
                other: 'PointLike[VEC]|Point|UvPoint|Vector2|Vector3'
                ) -> 'Point|UvPoint|Vector2|Vector3':
        '''Return a vector from one point to another.'''
        if type(other) is type(self):
            vargs = (a - b for a, b in zip(self, other))
            match len(self):
                case 2:
                    return Vector2(*vargs)
                case 3:
                    return Vector3(*vargs)
                case _:
                    raise TypeError(f'Invalid point type: {type(self)}')
            return type(self)(*(a - b for a, b in zip(self, other)))
        elif isinstance(other, (Vector2, Vector3, Tangent)) and len(other) == len(self):
            vtype = _VTYPES[len(self)]
            return cast(VEC, vtype(*(a - b for a, b in zip(self, other))))
        return NotImplemented

    def __add__(self, other: VEC) -> Self: # type: ignore
        '''
        Return a new point by adding a vector to this point.
        '''
        if not isinstance(cast(Any, other), (Vector2, Vector3)) and len(other) == len(self):
            return NotImplemented
        return type(self)(*(a + b for a, b in zip(self, other)))

    @staticmethod
    def distance(p1: 'PointLike[VEC]', p2: 'PointLike[VEC]') -> float:
        if type(p1) is not type(p2):
            raise TypeError('Points must be of the same type: {p1} {p2}')
        return sqrt(sum((a-b)**2 for a,b in zip(p1, p2)))


class Vector2(_Floats2, VectorLike['Vector2', 'Vector2Spec'], _Arrayable): # type: ignore
    '''
    A 2D vector, x and y.
    '''
    pass

class Vector3(_Floats3, VectorLike['Vector3', 'Vector3Spec'], _Arrayable): # type: ignore
    '''
    A 3D vector, x, y, and z. 3D vectors support cross products.
    '''
    
    def __matmul__(self, other: 'Self|Tangent') -> 'Vector3':
        '''
        Return the cross product of this vector and another.
        '''
        # Tangent is a 3D vector with additional info
        match other:
            case Tangent():
                sign = 1 if other.w == 1 else -1
            case VectorLike():
                sign = 1
            case _:
                return NotImplemented
        return Vector3(
            sign * (self.y * other.z - self.z * other.y),
            sign * (self.z * other.x - self.x * other.z),
            sign * (self.x * other.y - self.y * other.x) 
        )


class Vector4(_Floats4, VectorLike['Vector4', 'Vector4Spec'], _Arrayable): # type: ignore
    '''
    A 4D vector, x, y, z, and w.
    '''


class Point(_Floats3, PointLike[Vector3]): # type: ignore
    '''
    A point in 3D space. Required for any `Vertex`.
    '''
    pass

class Scale(_Floats3):
    '''
    Scale factors for a 3D object node.
    '''
    pass

class Tangent_(NamedTuple):
    x: float
    y: float
    z: float
    w: Literal[-1, 1]

class Tangent(Tangent_, VectorLike['Tangent|Vector3', 'Vector3Spec|TangentSpec']): # type: ignore
    '''
    A tangent vector. The w value is -1 or 1, indicating the direction of the bitangent.
    '''

    def __bool__(self) -> bool:
        '''
        Ignore w, which is always -1 0r 1
        '''
        x, y, z, _ = self
        return (x*x + y*y + z*z) > EPSILON*EPSILON
    
    def __add__(self, other: 'Vector3Spec|TangentSpec') -> 'Tangent': # type: ignore
        o = self._other_vector(other)
        if o is None:
            return NotImplemented
        if len(o) == 4:
            o = cast('TangentSpec', o)
            if o[3] not in (-1.0, 1.0, -1, 1):
                return NotImplemented
        # Should not be needed, but an Unknown slips past the flow analysis.
        return Tangent(float(self.x + o[0]), float(self.y + o[1]), float(self.z + o[2]), self.w)
    
    def __sub__(self, other: 'Vector3Spec|TangentSpec') -> 'Tangent':
        match other:
            case Tangent()|Vector3():
                pass
            case tuple()|np.ndarray():
                match len(other):
                    case 3:
                        other = vector3(*other)
                    case 4 if float(other[3]) in (-1.0, 1.0): # type: ignore
                        other = tangent(*other) # type: ignore
                    case _:
                        return NotImplemented
            case _:
                return NotImplemented
        # Should not be needed, but an Unknown slips past the flow analysis.
        o = cast('Tangent|Vector3', other)
        return Tangent(self.x - o.x, self.y - o.y, self.z - o.z, self.w)
    
    @overload
    def __mul__(self, other: Scalar) -> 'Tangent': ...
    @overload
    def __mul__(self, other: 'Vector3Spec|TangentSpec') -> float: ...
    def __mul__(self, other: 'Scalar|Vector3Spec|TangentSpec') -> 'Tangent|float':  # type: ignore
        match other:
            case 1|1.0:
                return self
            case Vector3()|Tangent():
                o = other
            case _:
                o = self._other_vector(other)
                if o is None:
                    return NotImplemented
        return self.x * float(o[0]) + self.y * float(o[1]) + self.z * float(o[2])
    
    @property
    def length(self):
        x, y, z, _ = self
        return sqrt(x*x + y*y + z*z)
    
    def __neg__(self) -> 'Tangent':
        return Tangent(-self.x, -self.y, -self.z, self.w)
    
    def __truediv__(self, other: Scalar) -> 'Tangent':
        match other:
            case float()|np.floating()|int()|np.integer():
                other = float(cast(Any, other))
            case _:
                return NotImplemented
        return Tangent(self.x / other, self.y / other, self.z / other, self.w)
    
    def dot(self, other: 'Vector3Spec|TangentSpec') -> float:
        match other:
            case Tangent():
                pass
            case  VectorLike() if len(other) == 3:
                pass 
            case _:
                raise TypeError('Invalid vector dot product')
        return self.x * other.x + self.y * other.y + self.z * other.z

    __matmul__ = Vector3.__matmul__


class _Sized(_Arrayable):
    '''
    An attribute value which may be stored in different formats.
    '''
    @property
    @abstractmethod
    def size(self) -> ByteSizeAuto: ...


class _Sized8(_Sized):
    '''Things stored in 8-bit format'''

    @property
    def size(self) -> Literal[1]:
        return 1


class _Sized16(_Sized):
    '''Things stored in 16-bit format'''

    @property
    def size(self) -> Literal[2]:
        return 2


class _Sizedf(_Sized):
    '''Things stored in 32-bit floating format'''

    @property
    def size(self) -> Literal[4]:
        return 4


class UvPoint(NamedTuple):
    '''
    A 2D texture coordinate (in U and V) in floating point or ints.
    '''
    pass


class _Uvf(NamedTuple):
    '''
    A 2D texture coordinate (in U and V) in floating point.
    '''
    u: float
    v: float

    @property
    def x(self) -> float:
        return self.u

    @property
    def y(self) -> float:
        return self.v


class UvfFloat(_Uvf, PointLike[Vector2], UvPoint, _Sizedf): # type: ignore
    '''
    A 2D texture coordinate (in U and V) in floating point.
    '''
    
    def __sub__(self, other: Self) -> Vector2: # type: ignore
        '''Return a vector from one point to another.'''
        if type(self) is not type(other):
            return NotImplemented
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __add__(self, other: Vector2) -> Self: # type: ignore
        '''
        Return a new point by adding a vector to this point.
        '''
        if type(other) is not Vector2:
            return NotImplemented
        return type(self)(self.x + other.x, self.y + other.y)


class _UvInt(NamedTuple):
    '''
    A 2D texture coordinate (in U and V) in normalied ints.
    '''
    u: int
    v: int
    
    @property
    def x(self) -> int:
        return self.u

    @property
    def y(self) -> int:
        return self.v

    def __sub__(self, other: Self) -> Vector2:
        '''Return a vector from one point to another.'''
        if type(self) is not type(other):
            return NotImplemented
        return Vector2(float(self.x - other.x), float(self.y - other.y))
    
    def __add__(self, other: Vector2) -> '_UvInt': # type: ignore
        '''
        Return a new point by adding a vector to this point.
        '''
        if type(other) is not Vector2:
            return NotImplemented
        return _UvInt(round(self.x + other.x), round(self.y + other.y))


class Uv8(_UvInt, PointLike, UvPoint, _Sized8): # type: ignore
    '''
    A 2D texture coordinate (in U and V) in 8-bit integers.
    '''
    pass


class Uv16(_UvInt, PointLike, UvPoint, _Sized16): # type: ignore
    '''
    A 2D texture coordinate (in U and V) in 16-bit integers.
    '''
    pass


class Color(NamedTuple):
    '''
    A color, RGB or RGBA, with floating point values between 0.0 and 1.0, inclusive,
    or int values between 0 and 255, inclusive or 0 and 65535, inclusive.

    This is a marker class for any color type.  The individual color classes determine
    how the values are interpreted.
    '''
    pass


class _RGBf(NamedTuple):
    '''Floating-point color'''
    r: float
    g: float
    b: float


class _RGBAf(NamedTuple):
    '''Floating-point color'''
    r: float
    g: float
    b: float
    a: float

class RGB(_RGBf, _Sizedf, Color): # type: ignore
    '''_
    A RGB color with float values between 0..1, inclusive.
    '''
    pass

class RGBA(_RGBAf, _Sizedf, Color): # type: ignore
    '''
    A RGBA color with floating point values between 0.0 and 1.0, inclusive.
    '''
    pass


class _RGBI(NamedTuple):
    '''Integer-valued color'''
    r: int
    g: int
    b: int


class _RGBAI(NamedTuple):
    '''Integer-valued color with alpha'''
    r: int
    g: int
    b: int
    a: int


class RGB8(_RGBI, _Sized8, Color): # type: ignore
    '''
    An RGB color with 8-bit integer values between 0 and 255, inclusive.
    '''
    pass


class RGBA8(_RGBAI, _Sized8, Color): # type: ignore
    '''
    An RGAB color with 8-bit integer values between 0 and 255, inclusive.
    '''
    pass


class RGB16(_RGBI, _Sized16, Color): # type: ignore
    '''
    An RGB color with 16-bit integer values between 0 and 255, inclusive.
    '''


class RGBA16(_RGBAI, _Sized16, Color): # type: ignore
    '''
    An RGBA color with 16-bit integer values between 0 and 255, inclusive.
    '''


class _Joint(NamedTuple):
    '''
    A tuple of four integers representing a joint index.
    '''
    j1: int
    j2: int
    j3: int
    j4: int


class Joint(_Joint, _Sized): # type: ignore
    '''An object containing up to 4 joint indexes'''
    pass


class _Joint8(Joint, _Sized8): # type: ignore
    '''
    A tuple of four 8-bit integers representing a joint index
    '''
    pass


class _Joint16(Joint, _Sized16): # type: ignore
    '''
    A tuple of four 16-bit integers representing a joint index.
    '''
    pass


class Weight(NamedTuple):
    '''
    A weioght for a joint animation. May be in 8-bit, 16-bit, or float format.
    '''
    pass


class _Weightf_(NamedTuple):
    '''
    A tuple of four floats representing a morph target weight.
    '''
    w1: float
    w2: float
    w3: float
    w4: float


class _Weightf(_Weightf_, _Sizedf, Weight): # type: ignore
    pass


class _WeightX(NamedTuple):
    '''
    A tuple of four floats representing a morph target weight.
    '''
    w1: int
    w2: int
    w3: int
    w4: int


class _Weight8(_WeightX, _Sized8, Weight): # type: ignore
    '''
    A tuple of four 8-bit ints representing a morph target weight.
    '''
    pass


class _Weight16(_WeightX, _Sized16, Weight): # type: ignore
    '''
    A tuple of four 16-bit ints representing a morph target weight.
    '''
    pass


_NP2Vector: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.float32]]
'''Numpy float32 representation of a 2D vector.'''
_NP3Vector: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]
'''Numpy float32 representation of a 3D vector.'''
_NP4Vector: TypeAlias = np.ndarray[tuple[Literal[4]], np.dtype[np.float32]]
'''Numpy float32 representation of a 4D vector.'''

_NP4IVector16: TypeAlias = np.ndarray[tuple[Literal[4]], np.dtype[np.uint16]]
'''Numpy uint16 representation of a 4D vector.'''
_NP4IVector8: TypeAlias = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]
'''Numpy uint8 representation of a 4D vector.'''

_NP3IVector16: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.uint16]]
'''Numpy uint16 representation of a 3D vector.'''
_NP3IVector8: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.uint8]]
'''Numpy uint8 representation of a 3D vector.'''

_NPIVector8: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint8]]
_NPIVector16: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
_NPIVector: TypeAlias = _NPIVector8|_NPIVector16

_NPVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]


_Tuple2Floats: TypeAlias = tuple[Scalar, Scalar]
'''tuple representation of a 2 floats.'''
_Tuple3Floats: TypeAlias = tuple[Scalar, Scalar, Scalar]
'''tuple representation of  3 floats.'''
_Tuple4Floats: TypeAlias = tuple[Scalar, Scalar, Scalar, Scalar]
'''tuple representation of 4 floats.'''

_Tuple4Ints: TypeAlias = tuple[IntScalar, IntScalar, IntScalar, IntScalar]
'''tuple integer representation of for integer.'''


_Float2Spec: TypeAlias = _Tuple2Floats|_NP2Vector
'''A specification for 2D float values,'''
_Float3Spec: TypeAlias = _Tuple3Floats|_NP3Vector
'''A specification for 3D float values.'''
_Float4Spec: TypeAlias = _Tuple4Floats|_NP4Vector
'''A specification for 4D float values.'''

_Int4Spec: TypeAlias = _Tuple4Ints|_NP4IVector8|_NP4IVector16
'''A specification for 4 integer values, x, y, z, and w.'''

Vector2Spec: TypeAlias = _Float2Spec|Vector2
'''A specification for a 2D vector, x and y.'''
Vector3Spec: TypeAlias = _Float3Spec|Vector3
'''A specification for a 3D vector, x, y, and z.'''
Vector4Spec: TypeAlias = _Float4Spec|Vector4
'''A specification for a 4D vector, x, y, z, and w.'''

VectorSpec: TypeAlias = Vector2Spec|Vector3Spec|Vector4Spec
'''A specification for a vector of dimensions 2, 3, o4 4, with , x, y, z, and w.'''

PointSpec: TypeAlias = _Float3Spec|Point
'''A specification for a point in 3D space, x, y, and z.'''
TangentSpec: TypeAlias = tuple[Scalar, Scalar, Scalar, Literal[-1, 1]]|_NP4Vector|Tangent
'''
A specification for a tangent vector, x, y, z, and w.
The w value is -1 or 1, indicating the direction of the bitangent.
'''
UvSpec: TypeAlias = _Float2Spec|UvPoint
'''
A texture coordinate object
'''

NormalSpec: TypeAlias = Vector3Spec
'''Specificaton for a normal vector, x, y, and z. (an ordinary Vector3)'''
ScaleSpec: TypeAlias = Vector3Spec|Scalar|Scale
'''A specification for a scale factor, x, y, and z, or a float to scale uniformly.'''
ColorSpec: TypeAlias = Color|_Float3Spec|_Float4Spec|_NP4IVector8|_NP4IVector16|_NP3IVector8|_NP3IVector16
'''A soecification for a color, RGB or RGBA.'''
JointSpec: TypeAlias = Joint|_Int4Spec
'''Specification for up to 4 joint nodes'''
WeightSpec: TypeAlias = Weight|_Float4Spec|_Int4Spec
'''Specification for up to 4 weights.'''

AttributeDataItem: TypeAlias = (
    int
    |float
    |PointSpec
    |NormalSpec
    |UvSpec
    |TangentSpec
    |ColorSpec
    |JointSpec
    |WeightSpec
    |Vector2Spec
    |Vector3Spec
    |Vector4Spec
    |tuple[IntScalar, ...]
    |tuple[Scalar, ...]
    |np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    |np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
    |np.ndarray[tuple[int, ...], np.dtype[np.uint16]]
    |np.ndarray[tuple[int, ...], np.dtype[np.uint32]]
    |np.ndarray[tuple[int, ...], np.dtype[np.int8]]
    |np.ndarray[tuple[int, ...], np.dtype[np.int16]]
)
'''
Valid types for an attribute data item.
'''


@overload
def point() -> Point: ...
@overload
def point(p: PointSpec, /) -> Point: ...
@overload
def point(p: np.ndarray[tuple[int], np.dtype[np.float32]], /) -> Point: ...
@overload
def point(x: Scalar, y: Scalar, z: Scalar) -> Point: ...
def point(x: Optional[Scalar|PointSpec|np.ndarray[tuple[int], np.dtype[np.float32]]|tuple[Scalar,Scalar,Scalar]]=None,
        y: Optional[Scalar]=None,
        z: Optional[Scalar]=None) -> Point:
    '''
    Validate and return a canonicalized point object.
    Only the type is canonicalized, not the values.

    Parameters
    ----------
        x: The x value of the point, or a point object of some type.
        y: The y value of the point.
        z: The z value of the point. 

    Returns
    -------
        A `_P{oint` object (a `NamedTuple`)
    '''
    match x, y, z:
        case None, None, None:
            return Point(0.0, 0.0, 0.0)
        case Point(), None, None:
            return x
        case float()|int()|np.floating()|np.integer(), \
                float()|int()|np.floating()|np.integer(), \
                float()|int()|np.floating()|np.integer():
            return Point(float(x), float(y), float(z))
        case (float()|int()|np.floating()|np.integer(),
              float()|int()|np.floating()|np.integer(),
              float()|int()|np.floating()|np.integer()), None, None:
            return Point(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return Point(float(x[0]), float(x[1]), float(x[2]))
        case _:
            raise ValueError('Invalid point')
        

@overload
def uv() -> UvPoint: ...
@overload
def uv(v: UvSpec, /) -> UvPoint: ...
@overload
def uv(u: Scalar, v: Scalar, /) -> UvPoint: ...
def uv(u: Optional[Scalar|UvSpec]=None,
        v: Optional[Scalar]=None, /, *,
        size: ByteSizeAuto|Literal['inf']=4) -> UvPoint:
    '''
    Return a canonicalized Uv texture coordinate object.

    Parameters
    ----------
        u: The x value of the texture coordinate, or a Uv object of some type.
        v: The y value of the texture coordinate.
        size: The size of the texture coordinate. 4 is the default, and
            indicates a floating point value. 8 or 16 indicate an integer
            value.

    Returns
    -------
        A `_Uv` object (a `NamedTuple`)
    '''
    match size:
        case 1:
            _uv  = Uv8
            def scale(v): # type: ignore
                v = min(1.0, max(0.0, float(v)))  # type: ignore
                return round(v * 255)
        case 2:
            _uv  = Uv16
            def scale(v): # type: ignore
                v = min(1.0, max(0.0, float(v))) # type: ignore
                return round(v * 65535)
        case 4|'inf':
            _uv = UvfFloat
            def scale(v): # type: ignore
                return min(1.0, max(0.0, float(v))) # type: ignore
        case _:
            raise ValueError(f'Invalid size for uv = {size}')
    def unscale(v: UvfFloat|Uv8|Uv16):
        match v:
            case UvfFloat():
                return v
            case Uv8():
                return UvfFloat(float(v.u) / 255, float(v.v) / 255)
            case Uv16():
                return UvfFloat(float(v.u) / 65535, float(v.v) / 65535)
            case _:
                return v
    match u, v:
        case None, None:
            return _uv(scale(0.0), scale(0.0)) # type: ignore
        case UvfFloat()|Uv8()|Uv16(), None:
            if type(u) is _uv:
               return cast(UvPoint, u)
            u = unscale(u)
            return _uv(scale(u.u), scale(u.v))  # type: ignore
        case (float()|int()|np.floating(), float()|int()|np.floating()), None:
            return _uv(scale(u[0]), scale(u[1])) # type: ignore
        case np.ndarray(), None if u.shape == (2,):
            return _uv(scale(u[0]), scale(u[1])) # type: ignore
        case float()|int()|np.floating(), float()|int()|np.floating():
            return _uv(scale(u), scale(v)) # type: ignore
        case _:
            raise ValueError('Invalid uv')     


@overload
def vector2() -> Vector2: ...
@overload
def vector2(v: Vector2Spec, /) -> Vector2: ...
@overload
def vector2(x: Scalar, y: Scalar) -> Vector2: ...
def vector2(x: Optional[Scalar|Vector2Spec]=None,
            y: Optional[Scalar]=None) -> Vector2:
    match x,y:
        case None, None:
            return Vector2(0.0, 0.0)
        case Vector2(), None:
            return x 
        case (float()|int()|np.floating(), float()|int()|np.floating()), None:
            return Vector2(float(x[0]), float(x[1]))
        case np.ndarray(), None if x.shape == (2,):
            return Vector2(float(x[0]), float(x[1]))
        case float()|int()|np.floating(), float()|int()|np.floating():
            return Vector2(float(x), float(y))
        case _:
            raise ValueError('Invalid vector2')  


@overload
def vector3() -> Vector3: ...
@overload
def vector3(v: Vector3Spec, /) -> Vector3: ...
@overload
def vector3(x: Scalar, y: Scalar, z: Scalar) -> Vector3: ...
def vector3(x: Optional[Scalar|Vector3Spec]=None,
            y: Optional[Scalar]=None,
            z: Optional[Scalar]=None) -> Vector3:
    match x, y, z:
        case None, None, None:
            return Vector3(0.0, 0.0, 0.0)
        case Vector3(), None, None:
            return x
        case (float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating()), None, None:
            return Vector3(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return Vector3(float(x[0]), float(x[1]), float(x[2]))
        case float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating():
            return Vector3(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid vector3')   


@overload
def vector4() -> Vector4: ...
@overload
def vector4(v: Vector4Spec, /) -> Vector4: ...
@overload
def vector4(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Vector4: ...
def vector4(x: Optional[Scalar|Vector4Spec]=None,
        y: Optional[Scalar]=None,
        z: Optional[Scalar]=None,
        w: Optional[Scalar]=None
    ) -> Vector4:
    match x, y, z, w:
        case None, None, None, None:
            return Vector4(0.0, 0.0, 0.0, 0.0)
        case Vector4(), None, None, None:
            return x
        case (float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating()), None, None, None:
            return Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case np.ndarray(), None, None, None if x.shape == (4,):
            return Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating():
            return Vector4(float(x), float(y), float(z), float(w))
        case _:
            raise ValueError('Invalid vector4')


_VTYPES = (vector2, vector3, vector4)
'''
A tuple of vector type constructors, indexed by the number of dimensions - 2.
'''


@overload
def scale() -> Scale: ...
@overload
def scale(p: ScaleSpec, /) -> Scale: ...
@overload
def scale(x: Scalar, y: Scalar, z: Scalar) -> Scale: ...
def scale(x: Optional[Scalar|ScaleSpec]=None,
        y: Optional[Scalar]=None,
        z: Optional[Scalar]=None) -> Scale:
    match x, y, z:
        case None, None, None:
            return Scale(1.0, 1.0, 1.0)
        case float()|int()|np.floating(), None, None if y is None and z is None:
            x = float(x)
            return Scale(x, x, x)
        case Scale(), None, None:
            return x
        case (float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating()), None, None:
            return Scale(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return Scale(float(x[0]), float(x[1]), float(x[2]))
        case float()|int()|np.floating(), float()|int()|np.floating(), float()|int()|np.floating():
            return Scale(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid scale')


@overload
def tangent( t: TangentSpec, /) -> Tangent: ...
@overload
def tangent(x: Scalar,
            y: Optional[Scalar]=None,
            z: Optional[Scalar]=None,
            w: Optional[Literal[-1, 1]] = None,
        ) -> Tangent: ...
def tangent(x: Scalar|TangentSpec,
            y: Optional[Scalar]=None,
            z: Optional[Scalar]=None,
            w: Optional[Literal[-1, 1]] = 1,
        ) -> Tangent:
    w = w or 1
    if w not in (-1, 1):
        raise ValueError(f'Invalid tangent w value {w}')
    match x, y, z, w:
        case Tangent(), None, None, -1|1:
            return x
        case (float()|int()|np.floating(),
              float()|int()|np.floating(),
              float()|int()|np.floating(), 
              -1|1
              ), None, None, -1|1:
            w = 1 if x[3] == 1 else -1
            return Tangent(float(x[0]), float(x[1]), float(x[2]), w)
        case np.ndarray(), None, None, -1|1 if (
            x.shape == (4,)
            and x[3] in (-1, 1)
        ):
            w = 1 if x[3] == 1 else -1
            return Tangent(float(x[0]), float(x[1]), float(x[2]), w)
        case (
            float()|int()|np.floating(),
            float()|int()|np.floating(),
            float()|int()|np.floating(),
            -1|1
        ):
            w = 1 if w == 1 else -1
            return Tangent(float(x), float(y), float(z), w)
        case _:
            raise ValueError('Invalid tangent')

_RGB_TYPES: dict[int, tuple[int|float, type, type]]= {
        1: (255, RGB8, RGBA8),
        2: (65535, RGB16, RGBA16),
        4: (1.0, RGB, RGBA),
}
@overload
def color(*, size: ByteSize=4) -> RGB: ...
@overload
def color(c: RGB|_NP3Vector|_Tuple3Floats, /) -> RGB: ...
@overload
def color(c: RGBA|_NP4Vector|_Tuple4Floats, /) -> RGBA: ...
@overload
def color(c: RGB8|_NP3IVector8, /) -> RGB8: ...
@overload
def color(c: RGBA8|_NP4IVector8, /) -> RGBA8: ...
@overload
def color(c: RGB16|_NP3IVector16, /) -> RGB16: ...
@overload
def color(c: RGBA16|_NP4IVector16, /) -> RGBA16: ...
@overload
def color(c: ColorSpec, /,) ->  Color: ...
@overload
def color(r: float01, g: float01, b: float01, a: float01, /, ) -> RGBA: ...
@overload
def color(r: float01, g: float01, b: float01, /, ) -> RGB: ...
@overload
def color(c: ColorSpec, /, *, size: Literal[1]) ->  RGB|RGBA: ...
@overload
def color(r: float01, g: float01, b: float01, a: float01, /, *, size: Literal[1]) -> RGBA8: ...
@overload
def color(r: float01, g: float01, b: float01, /, *, size: Literal[1]) -> RGB8: ...
@overload
def color(c: ColorSpec, /, *, size: Literal[2]) ->  RGB16|RGBA16: ...
@overload
def color(r: float01, g: float01, b: float01, a: float01, /, *, size: Literal[2]) -> RGBA16: ...
@overload
def color(r: float01, g: float01, b: float01, /, *, size: Literal[2]) -> RGB16: ...
@overload
def color(c: ColorSpec, /, *, size: Literal[4]) ->  RGB|RGBA: ...
@overload
def color(r: float01, g: float01, b: float01, a: float01, /, *, size: Literal[4]) -> RGBA: ...
@overload
def color(r: float01, g: float01, b: float01, /, *, size: Literal[4]) -> RGB: ...
@overload
def color(c: ColorSpec, /, *, size: ByteSize) ->  Color: ...
@overload
def color(r: float01, g: float01, b: float01, a: float01, /, *, size: ByteSize) -> Color: ...
@overload
def color(r: float01, g: float01, b: float01, /, *, size: ByteSize) -> Color: ...
def color(r: Optional[float01|ColorSpec]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None, /, *,
         size: ByteSize=4,
    ) -> RGB|RGBA|RGB8|RGBA8|RGB16|RGBA16|Color:
    limit, rgb, rgba = _RGB_TYPES[size]
    def scale(v: Scalar) -> int:
        return int(max(0, min(limit, round(v * limit))))
    def clamp(v: Scalar) -> float:
        return max(0.0, min(1.0, float(v)))
    rescale = clamp if size == 4 else scale
    match r, g, b, a:
        case None, None, None, None:
            return rgb(0, 0, 0)
        case (xr, xg, xb), None, None, None:
            return rgb(rescale(xr), rescale(xg), rescale(xb))
        case (xr, xg, xb, xa), None, None, None:
            return rgba(rescale(xr), rescale(xg), rescale(xb), rescale(xa))
        case Color(), None, None, None if isinstance(r, rgb|rgba):
            return r
        case RGB()|RGBA(), None, None, None:
            return rgb(rescale(r.r), rescale(r.g), rescale(r.b))
        case RGB8(), None, None, None:
            return color(round(r.r / 255), round(r.g / 255), round(r.b / 255),
                         size = size
                    )
        case RGBA8(), None, None, None:
            return color(round(r.r / 255), round(r.g / 255), round(r.b / 255), round(r.a / 255),
                         size = size
                    )
        case RGB16(), None, None, None:
            return color(r.r / 65535, r.g / 65535, r.b / 65535,
                        size = size
                    )
        case RGBA16(), None, None, None:
            return color(r.r / 65535, r.g / 65535, r.b / 65535, r.a / 65535,
                        size = size
                    )
        case np.ndarray(), None, None, None if r.shape == (4,):
            return rgba(rescale(r[0]), rescale(r[1]), rescale(r[2]), rescale(r[3]))
        case np.ndarray(), None, None, None if r.shape == (3,):
            return rgb(rescale(r[0]), rescale(r[1]), rescale(r[2]))
        case float()|0|1, float()|0|1, float()|0|1, float()|0|1:
            return rgba(rescale(r), rescale(g), rescale(b), rescale(a))
        case float()|0|1, float()|0|1, float()|0|1, None:
            return rgb(rescale(r), rescale(g), rescale(b))
        case _:
            raise ValueError('Invalid color')

@overload
def rgb8(r: int, g: int, b: int) -> RGB8: ...
@overload
def rgb8(r: int, g: int, b: int, a: int) -> RGBA8: ...
def rgb8(r: int, g: int, b: int, a: Optional[int]=None) -> RGB8|RGBA8:
    '''
    Create a RGB8 or RGBA8 color object directly from 8-bit integer values.
    '''
    def clamp(v: int) -> int:
        return max(0, min(255, v))
    if a is None:
        return RGB8(clamp(r), clamp(g), clamp(b))
    return RGBA8(clamp(r), clamp(g), clamp(b), clamp(a))

@overload
def rgb16(r: int, g: int, b: int) -> RGB16: ...
@overload
def rgb16(r: int, g: int, b: int, a: int) -> RGBA16: ...
def rgb16(r: int, g: int, b: int, a: Optional[int]=None) -> RGB16|RGBA16:
    '''
    Create a RGB16 or RGBA16 color object directly from 16-bit integer values.
    '''
    def clamp(v: int) -> int:
        return max(0, min(65536, v))
    if a is None:
        return RGB16(clamp(r), clamp(g), clamp(b))
    return RGBA16(clamp(r), clamp(g), clamp(b), clamp(a))


def joints(weights: Mapping[IntScalar, Scalar]|Iterable[tuple[IntScalar,Scalar]], /,
           size: Literal[0, 1, 2]=0,
           precision: Literal[0, 1, 2, 4]=0,
        ) -> tuple[tuple[Joint, ...], tuple['Weight', ...]]:
    '''
    Validate and return tuples of `Joint` objects and `Weight` objects. Each object holds up to 4 valuies.
    '''
    match weights:
        case Mapping():
           # An Iterable[X] is incorrectliy promotted to a Mapping[X, Unknown]
           w = cast(Mapping[IntScalar, Scalar], weights)
           return joint(*w.keys(), size=size), weight(*w.values(), precision=precision)
        case Sequence():
            return joint(*[i[0] for i in weights], size=size), weight([i[1] for i in weights], precision=precision)
        case Iterable():
            return joints(tuple(weights), size=size, precision=precision)


_NpIntDtype: TypeAlias = type[np.uint8]|type[np.uint16]
JOINT_RANGES: list[tuple[type|None, int, tuple[_NpIntDtype, _NpIntDtype]|None]] = [
        (None, 0, None),
        (_Joint8, 255, (np.uint8, np.uint16),),
        (_Joint16, 65535, (np.uint8, np.uint16,)),
    ]


@overload
def joint(ids: Iterable[IntScalar]|_NPIVector, /, *,
        size: Literal[1],
        ) -> tuple[_Joint8]: ...
@overload
def joint(ids: Iterable[IntScalar]|_NPIVector, /, *,
        size: Literal[2],
        ) -> tuple[_Joint16]: ...
@overload
def joint(ids: Iterable[IntScalar]|_NPIVector, /, *,
        size: int=0,
        ) -> tuple[_Joint8]|tuple[_Joint16]: ...
@overload
def joint(*ids: IntScalar,
          size: Literal[1],
          ) -> tuple[_Joint8,...]: ...
@overload
def joint(*ids: IntScalar,
          size: Literal[2],
          ) -> tuple[_Joint16,...]: ...
@overload
def joint(*ids: IntScalar,
          size: ByteSizeAuto=0,
          ) -> tuple[_Joint8,...]|tuple[_Joint16,...]: ...
def joint(*ids: IntScalar|Iterable[IntScalar]|_NPIVector,
          size: ByteSizeAuto|int=0,
          ) -> tuple[_Joint8, ...]|tuple[_Joint16,...]:
    '''
    Validate and return a tuple of joint objects, in groups of four.

    Parameters
    ----------
        ids: A tuple of joint indices, or a numpy array of joint indices.
        size: The byte size of the joint indices. 0 for unspecified, 1 for 8-bit, 2 for 16.
    '''
    validated: bool
    match ids:
        case (Sequence(),):
            id_list = ids[0]
        case (np.ndarray(),):
            id_list = cast(_NPIVector, ids[0].flatten())
        case (Iterable(),):
            id_list = ids[0]
        case _:
            id_list = cast(tuple[IntScalar], ids)
    # Doing match on id_list makes the type analysis makes it forget the type arguments to
    # np.ndarray, so we have to do this.
    id_list2 = id_list
    match size:
        case 0:
            match id_list2:
                case np.ndarray():
                    match id_list.dtype: # type: ignore
                        case np.uint8:
                            size, validated = 1, True
                        case np.uint16:
                            size, validated = 2, True
                        case _: # type: ignore
                            raise ValueError(f'Invalid joint dtype {id_list.dtype}') # type: ignore
                case Sequence() if all(
                                    isinstance(i, (int, np.integer)) # type: ignore
                                    and i <= 255
                                    for i in id_list
                                ):
                    size, validated = 1, True
                case Sequence() if all(
                                    isinstance(i, (int, np.integer)) # type: ignore
                                    and i <= 65535
                                    for i in id_list
                                ):
                    size, validated = 2, True
                case Sequence():
                    raise ValueError('Joint ids out of range')
                case Iterable():
                    # So we avoid two passes (identifying size, then collecting), which may not be posssible
                    # if the iterable is e.g. a generator.
                    return joint(tuple(id_list), size=0)
                case _:
                    raise ValueError('Invalid joint ids')
        case 1|2:
            match id_list2:
                case np.ndarray() if id_list.dtype == np.uint8: # type: ignore
                    validated = True
                case _:
                    validated = False
        case _:
            raise ValueError(f'Invalid joint {size=}')
    jtype, lim, np_dtype = JOINT_RANGES[size]
    if jtype is None or np_dtype is None:
        raise ValueError('Invalid joint size')
    if not validated:
        if not all(isinstance(i, (int, np.integer)) and i <= lim for i in id_list):
            raise ValueError('Invalid joint ids')
        validated = True
    return tuple(jtype(*chunk) for chunk in chunk4i(id_list))  


def chunk4(values: Iterable[Scalar|None]) -> Iterable[tuple[float, float, float, float]]:
    '''
    Chunk an iterable of float values into groups of 4. The last chuunk will be
    extended with 0s if it is less than 4 values.
    '''
    viter = iter(values)
    while True:
        try:
            v1 = float(next(viter) or 0.0)
        except StopIteration:
            return
        try:
            v2 = float(next(viter) or 0.0)
        except StopIteration:
            yield (v1, 0.0, 0.0, 0.0)
            return
        try:
            v3 = float(next(viter) or 0.0)
        except StopIteration:
            yield (v1, v2, 0.0, 0.0)
            return
        try:
            v4 = float(next(viter) or 0.0)
        except StopIteration:
            yield (v1, v2, v3, 0.0)
            return
        yield (v1, v2, v3, v4)


def chunk4i(values: Iterable[IntScalar|None], /) -> Iterable[tuple[int, int, int, int]]:
    '''
    Chunk an iterable of int values into groups of 4. The last group  will be
    extended with 0s if it is less than 4 values.
    '''
    viter = iter(values)
    while True:
        try:
            v1 = int(next(viter) or 0)
        except StopIteration:
            return
        try:
            v2 = int(next(viter) or 0)
        except StopIteration:
            yield (v1, 0, 0, 0)
            return
        try:
            v3 = int(next(viter) or 0)
        except StopIteration:
            yield (v1, v2, 0, 0)
            return
        try:
            v4 = int(next(viter) or 0)
        except StopIteration:
            yield (v1, v2, v3, 0)
            return
        yield (v1, v2, v3, v4)

@overload
def weight(args: Iterable[Scalar|None]|_NPVector, /, *,
           precision: Literal[1],
           ) -> tuple[_Weight8]: ...
@overload
def weight(args: Iterable[Scalar|None]|_NPVector, /, *,
           precision: Literal[2],
           ) -> tuple[_Weight16]: ...
@overload
def weight(args: Iterable[Scalar|None]|_NPVector, /, *,
           precision: Literal[0, 4],
           ) -> tuple[_Weightf]: ...
@overload
def weight(args: Iterable[Scalar|None]|_NPVector, /, *,
           precision: int=0,
           ) -> tuple[Weight]: ...
@overload
def weight(*args: Scalar|None) -> tuple[_Weightf, ...]: ...
@overload
def weight(*args: Scalar|None, precision: Literal[1]) -> tuple[_Weight8, ...]: ...
@overload
def weight(*args: Scalar|None, precision: Literal[2]) -> tuple[_Weight16, ...]: ...
@overload
def weight(*args: Scalar|None, precision: Literal[0, 4]) -> tuple[_Weightf, ...]: ...
@overload
def weight(*args: Scalar|None, precision: int=0) -> tuple[_Weightf, ...]|tuple[_Weight8, ...]|tuple[_Weight16, ...]: ...
def weight(*args: Iterable[Scalar|None]|_NPVector|Scalar|None,
           precision: Literal[0, 1, 2, 4]|int=0
        ) -> tuple[_Weightf, ...]|tuple[_Weight8, ...]|tuple[_Weight16, ...]|tuple[Weight,...]:
    '''
    Validate and return a set of canonicalized weight objects. These will be interpreted
    according to the _precision_ parameter. Each weight object will hold weights for up to 4
    joints.

    precision=0
    precision=4
        32-bit floating numbers normalized between 0.0 and 1.0, inclusive. They will be
        reweighted to sum to 1.0
    precision=1
        8-bit integers normalized and weighted to sum to 255
    precision=2
        16-bit integers normalized and weighted to sum to 65535
    '''
    match args:
        case ()|((),):
            raise ValueError('Invalid weight')
        case (np.ndarray()|Iterable(),):
            nargs = cast(Iterable[Scalar|None], args[0])
        case _:
            nargs = cast(Iterable[Scalar|None], args)
    match precision:
        case 0|4:
            return _weightf(nargs)
        case 1:
            return _weighti(nargs, 255, _Weight8)
        case 2:
            return _weighti(nargs, 65535, _Weight16)
        case _:
            raise ValueError(f'Invalid {precision=}')


def _weightf(arg: Iterable[Scalar|None]) -> tuple[_Weightf, ...]:
    '''
    Validate and return a set of canonicalized weight objects based on
    float32 weights. The weights are normalized to sum to 1.0.
    '''
    values = arg
    if isinstance(arg, np.ndarray):
        total = float(arg.sum())
    else:
        total = sum(
            float(v or 0)
            for v in arg
        )
    if abs(total) < EPSILON:
        raise ValueError('No meaningfully non-zero weightw')
    return tuple(_Weightf(*(c/total for c in chunk)) for chunk in chunk4(values))


def _weighti(arg: Iterable[Scalar|None],
             limit: int,
             fn: Callable[[int, int, int, int], Any]) -> tuple[Any, ...]:
    match arg:
        case np.ndarray():
            values = arg
            total = float(arg.sum())
        case Sequence():
            values = arg
            total = sum(
                float(v or 0)
                for v in values
            )
        case Iterable():
            values = list(arg)
            total = sum(
                float(v or 0)
                for v in values
            )
    if abs(total) < EPSILON:
        raise ValueError('No meaningfully non-zero weights')
    ivalues = [round((i or 0) * limit / total) for i in values]
    dt = find_dtype(ivalues)
    avalues = np.fromiter(ivalues, dtype=dt)
    return tuple(fn(*chunk) for chunk in chunk4i(avalues))


def find_dtype(values: Iterable[IntScalar]) -> np.dtype[np.uint8]|np.dtype[np.uint16]|np.dtype[np.float32]:
    '''
    Find the smallest numpy dtype that can hold the values.
    '''
    assert isinstance(values, (Iterable, Generator))
    dt = np.dtype(np.uint8)
    for v in values:
        if isinstance(v, float):
            return np.dtype(np.float32)
        try:
            v = int(v)
        except TypeError:
            raise ValueError("Values must be convertable to int")
        if v < 0:
            raise ValueError('values must be non-negative')
        if v > 255:
            dt = np.dtype(np.uint16)
        if v > 65535:
            raise ValueError('values must be less than 65536')
    return dt


AttributeDataList: TypeAlias = (
    list[int]
    |list[float]
    |list[tuple[int, ...]]
    |list[tuple[float, ...]]
    |list[np.ndarray[tuple[int, ...], Any]]
)
'''
List of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''

BType: TypeAlias = type[AttributeDataItem]|type[type[AttributeDataItem]]|int|type[Scalar]|float
BTypeType: TypeAlias = type[BType]

BTYPE = TypeVar('BTYPE', bound=BType)