'''
Types describing attribute values, as well as node properties such as scale.

For example, node translation uses a Vector3 for translation.

In general, these functions take 4 types of parameters:
- The float or integer values of the components of the type.
- A tuple of float or integer values.
- A numpy array of float or integer values.
- An object of the type being created.
'''

from typing import NamedTuple, TypeAlias, Literal, TypeVar, overload, Optional, Any, Self
from math import sqrt
from collections.abc import Generator, Iterable, Callable, Sequence
from itertools import islice

import numpy as np

from gltf_builder.core_types import ByteSize, ByteSizeAuto

EPSILON = 1e-12
'''
A small value for floating point comparisons.
'''
EPSILON2 = EPSILON * EPSILON
'''
The square of `EPSILON` for efficient near-zero length tests.
'''

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


class VectorLike(NamedTuple):
    '''
    Types that directly support vector operations such as length, addition, and dot products.
    '''
    def __bool__(self) -> bool:
        return sum(v*v for v in self) >= EPSILON2
    
    @property
    def length(self):
        return sqrt(sum(v*v for v in self))
    
    def __add__(self, other: Self) -> Self:
        if type(self) is not type(other):
            raise ValueError('Invalid vector addition')
        return type(self)(*(a+b for a,b in zip(self, other)))
    
    def __sub__(self, other: Self) -> Self:
        if type(self) is not type(other):
            raise ValueError('Invalid vector subtraction')
        return type(self)(*(a-b for a,b in zip(self, other)))   

                                                                                                                                                                                                                                                                                                                                                                                                                        
    def __neg__(self) -> Self:
        return type(self)(*(-a for a in self))
    
    def __mul__(self, other: float|Self) -> Self:
        match other:
            case float():
                return type(self)(*(a*other for a in self))
            case VectorLike() if len(self) == len(other):
               return sum(a*b for a,b in zip(self, other))
            case _:
                raise ValueError('Invalid vector multiplication')

    def __rmul__(self, other: float|Self) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> 'VectorLike':
        return type(self)(*(a/other for a in self))
    
    def dot(self, other: 'VectorLike') -> float:
        return sum(a*b for a,b in zip(self, other))


class PointLike(NamedTuple):
    '''
    Pointlike quantities have meaningful scalar distances
    '''
    def __sub__(self, other: Self) -> float:
        if type(self) is not type(other):
            raise ValueError('Invalid point subtraction')
        return sqrt(sum((a-b)**2 for a,b in zip(self, other)))


class _Vector2(_Floats2, VectorLike):
    '''
    A 2D vector, x and y.
    '''
    x: float
    y: float


class _Vector3(_Floats3, VectorLike):
    '''
    A 3D vector, x, y, and z. 3D vectors support cross products.
    '''
    x: float
    y: float
    z: float
    
    def cross(self, other: Self) -> Self:
        '''
        Return the cross product of this vector and another.
        '''
        return _Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    __matmul__ = cross


class _Vector4(_Floats4, VectorLike):
    '''
    A 4D vector, x, y, z, and w.
    '''
    x: float
    y: float
    z: float
    w: float


class _Point(_Floats3, PointLike):
    '''
    A point in 3D space. Required for any `Vertex`.
    '''
    x: float
    y: float
    z: float


class _Scale(_Floats3):
    '''
    Scale factors for a 3D object node.
    '''
    x: float
    y: float
    z: float


class _Tangent(_Floats4, VectorLike):
    '''
    A tangent vector. The w value is -1 or 1, indicating the direction of the bitangent.
    '''
    x: float
    y: float
    z: float
    w: Literal[-1, 1]

    def __bool__(self) -> bool:
        '''
        Ignore w, which is always -1 0r 1
        '''
        x, y, z, _ = self
        return (x*x + y*y + z*z) > EPSILON*EPSILON
    
    def __mul__(self, other: 'float|VectorLike') -> 'VectorLike':
        match other:
            case float():
                return type(self)(*(a*other for a in self))
            case VectorLike() if len(self) == len(other):
               return self.x * other.x + self.y * other.y + self.z * other.z
            case _Tangent():
                return self.x * other.x + self.y * other.y + self.z * other.z
            case _:
                raise ValueError('Invalid vector multiplication')
    
    @property
    def length(self):
        x, y, z, _ = self
        return sqrt(x*x + y*y + z*z)
    
    def cross(self, other: 'Vector3|_Tangent') -> 'Vector3':
        '''
        Return the cross product of this vector and another. The cross product involving
        tangent is not tangent.
        '''
        match other:
            case _Tangent():
                sign = 1 if self.w == other.w else -1
            case _:
                sign = 1
        return  _Vector3(
            sign * (self.y * other.z - self.z * other.y),
            sign(self.z * other.x - self.x * other.z),
            sign(self.x * other.y - self.y * other.x)
        )
    
    __matmul__ = cross

class _Uvf_(NamedTuple):
    '''
    A 2D texture coordinate (in U and V) in floating point.
    '''
    u: float
    v: float

class _Uvf(_Uvf_, PointLike):
    u: float
    v: float

class _UvX(NamedTuple):
    '''
    A 2D texture coordinate (in U and V) in normalied ints.
    '''
    u: int
    v: int

class _Uv8(_UvX, PointLike):
    '''
    A 2D texture coordinate (in U and V) in 8-bit integers.
    '''
    u: int
    v: int


class _Uv16(_UvX, PointLike):
    '''
    A 2D texture coordinate (in U and V) in 16-bit integers.
    '''
    u: int
    v: int

_Uv: TypeAlias = _Uvf|_Uv8|_Uv16
'''
A 2D texture coordinate (in U and V) in floating point, 8-bit, or 16-bit integers.
'''

class RGB(NamedTuple):
    '''
    A RGB color with floating point values between 0.0 and 1.0, inclusive.
    '''
    r: float
    g: float
    b: float

class RGBA(NamedTuple):
    '''
    A RGBA color with floating point values between 0.0 and 1.0, inclusive.
    '''
    r: float
    g: float
    b: float
    a: float


class _RGBI(NamedTuple):
    r: int
    g: int
    b: int

class _RGBAI(NamedTuple):
    r: int
    g: int
    b: int
    a: int

class RGB8(_RGBI):
    '''
    An RGB color with 8-bit integer values between 0 and 255, inclusive.
    '''
    pass

class RGBA8(_RGBAI):
    '''
    An RGAB color with 8-bit integer values between 0 and 255, inclusive.
    '''
    pass

class RGB16(_RGBI):
    '''
    An RGB color with 16-bit integer values between 0 and 255, inclusive.
    '''

class RGBA16(_RGBAI):
    '''
    An RGBA color with 16-bit integer values between 0 and 255, inclusive.
    '''

_Colorf: TypeAlias = RGB|RGBA
_Color8: TypeAlias = RGB8|RGBA8
_Color16: TypeAlias = RGB16|RGBA16
_Color: TypeAlias = _Colorf|_Color8|_Color16

class _Joint(NamedTuple):
    '''
    A tuple of four integers representing a joint index.
    '''
    j1: int
    j2: int
    j3: int
    j4: int

class _Joint8(_Joint):
    '''
    A tuple of four 8-bit integers representing a joint index
    '''
    pass

class _Joint16(_Joint):
    '''
    A tuple of four 16-bit integers representing a joint index.
    '''
    pass

class _Weightf(NamedTuple):
    '''
    A tuple of four floats representing a morph target weight.
    '''
    w1: float
    w2: float
    w3: float
    w4: float

class _WeightX(NamedTuple):
    '''
    A tuple of four floats representing a morph target weight.
    '''
    w1: int
    w2: int
    w3: int
    w4: int

class _Weight8(_WeightX):
    '''
    A tuple of four 8-bit ints representing a morph target weight.
    '''
    pass

class _Weight16(_WeightX):
    '''
    A tuple of four 16-bit ints representing a morph target weight.
    '''
    pass

_Weight: TypeAlias = _Weightf|_Weight8|_Weight16
'''
A tuple of four floats or integers representing a morph target weight.
'''

NP2Vector: TypeAlias = np.ndarray[tuple[Literal[2]], np.float32]
NP3Vector: TypeAlias = np.ndarray[tuple[Literal[3]], np.float32]
NP4Vector: TypeAlias = np.ndarray[tuple[Literal[4]], np.float32]

NP4IVector32: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint32]
NP4IVector16: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint16]
NP4IVector8: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint8]
NP4IVector16s: TypeAlias = np.ndarray[tuple[Literal[4]], np.int16]
NP4IVector8s: TypeAlias = np.ndarray[tuple[Literal[4]], np.int8]

NP3IVector32: TypeAlias = np.ndarray[tuple[Literal[3]], np.uint32]
NP3IVector16: TypeAlias = np.ndarray[tuple[Literal[3]], np.uint16]
NP3IVector8: TypeAlias = np.ndarray[tuple[Literal[3]], np.uint8]
NP3IVector16s: TypeAlias = np.ndarray[tuple[Literal[3]], np.int16]
NP3IVector8s: TypeAlias = np.ndarray[tuple[Literal[3]], np.int8]

NP2IVector32: TypeAlias = np.ndarray[tuple[Literal[2]], np.uint32]
NP2IVector16: TypeAlias = np.ndarray[tuple[Literal[2]], np.uint16]
NP2IVector8: TypeAlias = np.ndarray[tuple[Literal[2]], np.uint8]
NP2IVector16s: TypeAlias = np.ndarray[tuple[Literal[2]], np.int16]
NP2IVector8s: TypeAlias = np.ndarray[tuple[Literal[2]], np.int8]

Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
Vec4: TypeAlias = tuple[float, float, float, float]

IVec4: TypeAlias = tuple[int, int, int, int]

Vector2: TypeAlias = Vec2|_Vector2|NP2Vector
Vector3: TypeAlias = Vec3|_Vector3|NP3Vector
Vector4: TypeAlias = Vec4|_Vector4|NP4Vector

_Vector: TypeAlias = _Vector2|_Vector3|_Vector4
Vector: TypeAlias = Vector2|Vector3|Vector4

Matrix2: TypeAlias = tuple[
    float, float,
    float, float,
]
Matrix3: TypeAlias = tuple[
    float, float, float,
    float, float, float,
    float, float, float
]
Matrix4: TypeAlias = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]

Scalar: TypeAlias = float
Point: TypeAlias = Vector3|_Point
Tangent: TypeAlias = Vector4|_Tangent
Uv: TypeAlias = Vector2|_Uv
'''
A texture coordinate object
'''
Normal: TypeAlias = Vector3
Scale: TypeAlias = Vector3|Scalar|_Scale
Color: TypeAlias = _Color|NP3Vector|Vec3|NP4Vector|Vec4
Joint: TypeAlias = _Joint|IVec4|NP4IVector8|NP4IVector16
Weight: TypeAlias = _Weight|Vec4|NP4Vector|IVec4|NP4IVector8|NP4IVector16

AttributeDataItem: TypeAlias = (
    Point
    |Normal
    |Uv
    |Tangent
    |Color
    |Joint
    |Weight
    |Vector2
    |Vector3
    |Vector4
    |tuple[int, ...]
    |tuple[float, ...]
    |np.ndarray[tuple[int, ...], np.float32]
    |np.ndarray[tuple[int, ...], np.uint8]
    |np.ndarray[tuple[int, ...], np.uint16]
    |np.ndarray[tuple[int, ...], np.uint32]
    |np.ndarray[tuple[int, ...], np.int8]
    |np.ndarray[tuple[int, ...], np.int16]
)
'''
Valid types for an attribute data item.
'''

float01: TypeAlias = float|Literal[0,1]
'''
A float value between 0 and 1, or the literals 0 or 1.
'''

@overload
def point() -> _Point: ...
@overload
def point(p: Point, /) -> _Point: ...
@overload
def point(p: np.ndarray, /) -> _Point: ...
@overload
def point(p: tuple[float,float|int,float|int], /) -> _Point: ...
@overload
def point(x: float|int, y: float|int, z: float|int) -> _Point: ...
def point(x: Optional[float|Point|np.ndarray|tuple[float|int,float|int,float|int]]=None,
        y: Optional[float|int]=None,
        z: Optional[float|int]=None) -> _Point:
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
            return _Point(0.0, 0.0, 0.0)
        case _Point(), None, None:
            return x
        case float()|int(), float()|int(), float()|int():
            return _Point(float(x), float(y), float(z))
        case (float()|int(), float()|int(), float()|int()), None, None:
            return _Point(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return _Point(float(x[0]), float(x[1]), float(x[2]))
        case _:
            raise ValueError('Invalid point')
        

@overload
def uv() -> _Uv: ...
@overload
def uv(v: Uv, /) -> _Uv: ...
@overload
def uv(u: float, v: float) -> _Uv: ...
def uv(u: Optional[float|Point]=None,
        v: Optional[float]=None,
        size: ByteSizeAuto=4) -> _Uv:
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
            _uv  = _Uv8
            def scale(v):
                v = min(1.0, max(0.0, float(v)))
                return round(v * 255)
        case 2:
            _uv  = _Uv16
            def scale(v):
                v = min(1.0, max(0.0, float(v)))
                return round(v * 65535)
        case 4|'inf':
            _uv = _Uvf
            def scale(v):
                return min(1.0, max(0.0, float(v)))
        case _:
            raise ValueError(f'Invalid size for uv = {size}')
    def unscale(v):
        match v:
            case _Uvf():
                return v
            case _Uv8():
                return _Uvf(float(v.u) / 255, float(v.v) / 255)
            case _Uv16():
                return _Uvf(float(v.u) / 65535, float(v.v) / 65535)
    match u, v:
        case None, None:
            return _uv(scale(0.0), scale(0.0))
        case _Uvf()|_Uv8()|_Uv16()|_UvX(), None:
            if type(u) is _uv:
               return u
            u = unscale(u)
            return _uv(scale(u.u), scale(u.v))
        case (float()|int(), float()|int()), None:
            return _uv(scale(u[0]), scale(u[1]))
        case np.ndarray(), None if u.shape == (2,):
            return _uv(scale(u[0]), scale(u[1]))
        case float()|int(), float()|int():
            return _uv(scale(u), scale(v))
        case _:
            raise ValueError('Invalid uv')     


@overload
def vector2() -> _Vector2: ...
@overload
def vector2(v: Vector2, /) -> _Vector2: ...
@overload
def vector2(x: float, y: float) -> _Vector2: ...
def vector2(x: Optional[float|Point]=None,
            y: Optional[float]=None) -> _Vector2:
    match x,y:
        case None, None:
            return _Vector2(0.0, 0.0)
        case _Vector2(), None:
            return x 
        case (float()|int(), float()|int()), None:
            return _Vector2(float(x[0]), float(x[1]))
        case np.ndarray(), None if x.shape == (2,):
            return _Vector2(float(x[0]), float(x[1]))
        case float()|int(), float()|int():
            return _Vector2(float(x), float(y))
        case _:
            raise ValueError('Invalid vector2')  


@overload
def vector3() -> _Vector3: ...
@overload
def vector3(v: Vector3, /) -> _Vector3: ...
@overload
def vector3(x: float, y: float, z: float) -> _Vector3: ...
def vector3(x: Optional[float|Point]=None,
            y: Optional[float]=None,
            z: Optional[float]=None) -> _Vector3:
    match x, y, z:
        case None, None, None:
            return _Vector3(0.0, 0.0, 0.0)
        case _Vector3(), None, None:
            return x
        case (float()|int(), float()|int(), float()|int()), None, None:
            return _Vector3(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return _Vector3(float(x[0]), float(x[1]), float(x[2]))
        case float()|int(), float()|int(), float()|int():
            return _Vector3(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid vector3')   


@overload
def vector4() -> _Vector4: ...
@overload
def vector4(v: Vector4, /) -> _Vector4: ...
@overload
def vector4(x: float, y: float, z: float) -> _Vector4: ...
def vector4(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None,
        w: Optional[float]=None
    ) -> _Vector4:
    match x, y, z, w:
        case None, None, None, None:
            return _Vector4(0.0, 0.0, 0.0, 0.0)
        case _Vector4(), None, None, None:
            return x
        case (float()|int(), float()|int(), float()|int(), float()|int()), None, None, None:
            return _Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case np.ndarray(), None, None, None if x.shape == (4,):
            return _Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case float()|int(), float()|int(), float()|int(), float()|int():
            return _Vector4(float(x), float(y), float(z), float(w))
        case _:
            raise ValueError('Invalid vector4')


@overload
def scale() -> _Scale: ...
@overload
def scale(p: Scale, /) -> _Scale: ...
@overload
def scale(x: float, y: float, z: float) -> _Scale: ...
def scale(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None) -> _Point:
    match x, y, z:
        case None, None, None:
            return _Scale(1.0, 1.0, 1.0)
        case float()|int(), None, None if y is None and z is None:
            x = float(x)
            return _Scale(x, x, x)
        case _Scale(), None, None:
            return x
        case (float()|int(), float()|int(), float()|int()), None, None:
            return _Scale(float(x[0]), float(x[1]), float(x[2]))
        case np.ndarray(), None, None if x.shape == (3,):
            return _Scale(float(x[0]), float(x[1]), float(x[2]))
        case float()|int(), float()|int(), float()|int():
            return _Scale(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid scale')


@overload
def tangent( t: Tangent) -> _Tangent: ...
@overload
def tangent(x: float|Tangent,
            y: Optional[float]=None,
            z: Optional[float]=None,
            w: Optional[Literal[-1, 1]] = None,
        ) -> _Tangent: ...
def tangent(x: float|Tangent,
            y: Optional[float]=None,
            z: Optional[float]=None,
            w: Optional[Literal[-1, 1]] = 1,
        ) -> _Tangent:
    w = w or 1
    match x, y, z, w:
        case _Tangent(), None, None, -1|1:
            return x
        case (float()|int(), float()|int(), float()|int(), float()|int()), None, None, -1|1:
            return _Tangent(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case np.ndarray(), None, None, -1|1 if x.shape == (4,) and x[3] in (-1, 1):
            return _Tangent(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case float()|int(), float()|int(), float()|int(), -1|1:
            return _Tangent(float(x), float(y), float(z), float(w))
        case _:
            raise ValueError('Invalid tangent')

@overload
def color() -> RGB: ...
@overload
def color(c: RGB|NP3Vector|Vec3) -> RGB: ...
@overload
def color(c: RGBA|NP4Vector|Vec4) -> RGBA: ...
@overload
def color(c: RGB8|NP3IVector8) -> RGB8: ...
@overload
def color(c: RGBA8|NP4IVector8) -> RGBA8: ...
@overload
def color(c: RGB8|NP3IVector16) -> RGB16: ...
@overload
def color(c: RGBA8|NP4IVector16) -> RGBA16: ...
@overload
def color(c: Color, /) ->  Color: ...
@overload
def color(r: float, g: float, b: float, a: float) -> RGBA: ...
@overload
def color(r: float, g: float, b: float) -> RGB: ...
@overload
def color(c: Color, /, size: Literal[8]) ->  RGB|RGBA: ...
@overload
def color(r: float, g: float, b: float, a: float, size: Literal[8]) -> RGBA8: ...
@overload
def color(r: float, g: float, b: float, size: Literal[8]) -> RGB8: ...
@overload
def color(c: Color, /, size: Literal[16]) ->  RGB16|RGBA16: ...
@overload
def color(r: float, g: float, b: float, a: float, size: Literal[16]) -> RGBA16: ...
@overload
def color(r: float, g: float, b: float, size: Literal[16]) -> RGB16: ...
@overload
def color(c: Color, /, size: Literal[32]) ->  RGB|RGBA: ...
@overload
def color(r: float, g: float, b: float, a: float, size: Literal[32]) -> RGBA: ...
@overload
def color(r: float, g: float, b: float, size: Literal[32]) -> RGB: ...
def color(r: Optional[float01|Color]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None,
         size: ByteSize=4,
    ) -> RGB|RGBA|RGB8|RGBA8|RGB16|RGBA16:
    limit, rgb, rgba = {
        1: (255, RGB8, RGBA8),
        2: (65535, RGB16, RGBA16),
        4: (1.0, RGB, RGBA),
    }[size]
    def scale(v: float) -> int:
        return int(max(0, min(limit, round(v * limit))))
    def clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))
    rescale = clamp if size == 4 else scale
    match r, g, b, a:
        case None, None, None, None:
            return rgb(0, 0, 0)
        case (xr, xg, xb), None, None, None:
            return rgb(rescale(xr), rescale(xg), rescale(xb))
        case (xr, xg, xb, xa), None, None, None:
            return rgba(rescale(xr), rescale(xg), rescale(xb), rescale(xa))
        case rgb()|rgba(), None, None, None:
            return r
        case RGB()|RGBA(), None, None, None:
            return rgb(rescale(r.r), rescale(r.g), rescale(r.b))
        case RGB8()|RGBA8(), None, None, None:
            return color(r.r / 255, r.g / 255, r.b / 255, r.a / 255,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
                          )
        case RGB16()|RGBA16(), None, None, None:
            return color(r.r / 65535, r.g / 65535, r.b / 65535, r.a / 65535,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
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


def rgb8(r: int, g: int, b: int, a: Optional[int]=None) -> RGB8:
    '''
    Create a RGB8 or RGBA8 color object directly from 8-bit integer values.
    '''
    def clamp(v: int) -> int:
        return max(0, min(255, v))
    if a is None:
        return RGB8(clamp(r), clamp(g), clamp(b))
    return RGBA8(clamp(r), clamp(g), clamp(b), clamp(a))


def rgb16(r: int, g: int, b: int, a: Optional[int]=None) -> RGB16:
    '''
    Create a RGB16 or RGBA16 color object directly from 16-bit integer values.
    '''
    def clamp(v: int) -> int:
        return max(0, min(65536, v))
    if a is None:
        return RGB16(clamp(r), clamp(g), clamp(b))
    return RGBA16(clamp(r), clamp(g), clamp(b), clamp(a))


def joints(weights: dict[int, int|float], /,
           size: Literal[0, 1, 2]=0,
           precision: Literal[0, 1, 2, 4]=0) -> tuple[tuple[_Joint, ...], tuple['Weight', ...]]:
    '''
    Validate and return tuples of joint objects and weight objects, in groups of four.
    '''
    return joint(*weights.keys(), size=size), weight(list(weights.values()), precision=precision)

JOINT_RANGES = [
        None,
        (_Joint8, 255, (np.uint8, np.uint16),),
        (_Joint16, 65535, (np.uint8, np.uint16,)),
    ]

@overload
def joint(ids: tuple[int,...]|np.ndarray[tuple[int],int], /,
        size: int=0,
        ) -> tuple[_Joint]: ...
@overload
def joint(*ids: int,
          size: ByteSizeAuto=0) -> tuple[_Joint,...]: ...
def joint(*ids: int|tuple[int, ...]|np.ndarray[tuple[int], int],
          size: ByteSizeAuto=0) -> tuple[_Joint, ...]:
    '''
    Validate and return a tuple of joint objects, in groups of four.

    Parameters
    ----------
        ids: A tuple of joint indices, or a numpy array of joint indices.
        size: The byte size of the joint indices. 0 for unspecified, 1 for 8-bit, 2 for 16.
    '''
    if size == 0:
        match ids:
            case _ if all(isinstance(i, int) for i in ids):
                size = 1 if all(i <= 255 for i in ids) else 2
            case _ if all(isinstance(i, Sequence) for i in ids):
                size = 1 if all(v <= 255 for s in ids for v in s) else 2
            case _  if all(isinstance(i, np.ndarray) for i in ids):
                size = 1 if all(i.dtype == np.uint8 for i in ids) else 2
            case _:
                raise ValueError('Invalid joints')
    jtype, lim, np_dtype = JOINT_RANGES[size]
    match ids:
        case (_Joint(),) if isinstance(ids[0], jtype):
            return ids
        case tuple() if all(isinstance(i, int) and i <= lim for i in ids):
            return tuple(jtype(*chunk) for chunk in chunk4i(ids))
        case (tuple(), *_) if all(
                isinstance(i, int) and i <= lim
                for d in ids
                for i in d
            ):
            num_values = sum(len(a) for a in ids)
            flat = (i for a in ids for i in a)
            return tuple(jtype(*chunk)
                         for chunk in chunk4i(ids[0], num_values=num_values)
                        )
        case _ if all(
                    isinstance(i, np.ndarray) and i.dtype in np_dtype
                    for i in ids
                ):
            num_values = sum(len(a) for a in ids)
            flat = (i for a in ids for i in a)
            return tuple(jtype(*chunk)
                         for chunk in chunk4i(flat, num_values=num_values)
                        )
        case _:
            raise ValueError('Invalid joints')    


def chunk4(values: Iterable[float|int|None],
           num_values: Optional[int]=None) -> Iterable[tuple[float, float, float, float]]:
    '''
    Chunk an iterable of float values into groups of 4. The last chuunk will be
    extended with 0s if it is less than 4 values.
    '''
    num_values = num_values or len(values)
    count = num_values // 4
    more = num_values % 4
    viter = iter(values)
    for i in range(count):
        yield (
            float(next(viter) or 0.0),
            float(next(viter) or 0.0),
            float(next(viter) or 0.0),
            float(next(viter) or 0.0),
        )
    match more:
        case 0:
            return
        case 1:
            yield (
                float(next(viter) or 0.0),
                0.0,
                0.0,
                0.0,
            )
        case 2:
            yield (
                float(next(viter) or 0.0),
                float(next(viter) or 0.0),
                0.0,
                0.0,
            )
        case 3:
            yield (
                float(next(viter) or 0.0),
                float(next(viter) or 0.0),
                float(next(viter) or 0.0),
                0.0,
            )


def chunk4i(values: Iterable[int|None], num_values: Optional[int]=None) -> Iterable[tuple[int, int, int, int]]:
    '''
    Chunk an iterable of int values into groups of 4. The last group  will be
    extended with 0s if it is less than 4 values.
    '''
    num_values = num_values or len(values)
    count = num_values // 4
    more = num_values % 4
    viter = iter(values)
    for i in range(count):
        yield (
            int(next(viter) or 0),
            int(next(viter) or 0),
            int(next(viter) or 0),
            int(next(viter) or 0),
        )
    match more:
        case 0:
            return
        case 1:
            yield (
                int(next(viter) or 0),
                0,
                0,
                0,
            )
        case 2:
            yield (
                int(next(viter) or 0),
                int(next(viter) or 0),
                0,
                0,
            )
        case 3:
            yield (
                int(next(viter) or 0),
                int(next(viter) or 0),
                int(next(viter) or 0),
                0,
            )
 
W  = TypeVar('W', bound=Weight)

@overload
def weight(v: W, /) -> tuple[W, ...]: ...
@overload
def weight(*args: float01|None) -> tuple[_Weightf, ...]: ...
@overload
def weight(*args: int|None, precision: Literal[1]) -> tuple[_Weight8, ...]: ...
@overload
def weight(*args: int|None, precision: Literal[16]) -> tuple[_Weight16, ...]: ...
def weight(*args: float01|None, precision: Literal[0, 1, 2, 4]=0) -> tuple[_Weightf, ...]:
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
    match precision:
        case 0|4:
            return _weightf(args)
        case 1:
            return _weighti(args, 255, _Weight8)
        case 2:
            return _weighti(args, 65535, _Weight16)
        case _:
            raise ValueError(f'Invalid {precision=}')


def _weightf(args: float01|None):
    '''
    Validate and return a set of canonicalized weight objects based on float32 weights.
    The weights are normalized to sum to 1.0.
    '''
    def reweigh(values: tuple[float|int|None]|np.ndarray) -> tuple[_Weightf, ...]:
        if isinstance(values, np.ndarray):
            total = values.sum()
        else:
            total = sum(
                v or 0 
                for v in values
            )
        if abs(total) < EPSILON:
            return (_Weightf(0.0, 0.0, 0.0, 0.0,),)
        return tuple(_Weightf(*(c/total for c in chunk)) for chunk in chunk4(values))
    match args:
        case (_Weightf(), *more) if all(isinstance(v, _Weightf) for v in more):
            return args
        case ()|((),):
            raise ValueError('Invalid weight')
        case _ if len(args) > 0 and all(v is None or isinstance(v, (float, int)) for v in args):
            return reweigh(args)
        case _ if all(v is None or isinstance(v, (float, int))
                    for a in args
                    for v in a
                ):
            return reweigh([v for a in args for v in a])
        case _ if all(
                        isinstance(a, np.ndarray) and a.dtype == np.float32
                        for a in args
                    ):
            return reweigh(args[0])
        case _:
            raise ValueError('Invalid weight') 

@overload
def weight8() -> tuple: ...
@overload
def weight8(v: Weight, /) -> tuple[_Weight8]: ...
@overload
def weight8(*args: float|int|None) -> tuple[_Weight8, ...]: ...
def weight8(*args: float|int|None) -> tuple[_Weight8, ...]:
    return _weighti(args, 255, _Weight8)


@overload
def weight16() -> tuple: ...
@overload
def weight16(v: Weight, /) -> tuple[_Weight16]: ...
@overload
def weight16(*args: int|float|None) -> tuple[_Weight16, ...]: ...
def weight16(*args: int|float|None) -> tuple[_Weight16, ...]:
    '''
    
    '''
    return _weighti(args, 65535, _Weight16)


def _weighti(args: tuple[np.ndarray|float|int|None],
             limit: int,
             fn: Callable[[tuple[int, int, int, int]], Any]) -> tuple[Any, ...]:
    match args:
        case ()|((),):
            raise ValueError('Invalid weight, zero-length')
        case _ if (limit == 255
                    and all(
                        isinstance(v, _Weight8)
                        for v in args
                    )
                ):
            return args
        case _ if (limit == 65535
                    and all(
                        isinstance(v, _Weight16)
                        for v in args
                    )
                ):
            return args
        case _ if all(
                        isinstance(v, int)
                        for v in args
                    ):
            dt = find_dtype(args)
            values = np.fromiter(args, dtype=dt)
        case _ if all(isinstance(v, np.ndarray) for v in args):
            dt = find_dtype(v for a in args for v in a)
            values = np.fromiter((v for a in args for v in a), dt)
        case _ if all(isinstance(v, Iterable) for v in args):
            dt = find_dtype(v for a in args for v in a)
            values = np.fromiter((v for a in args for v in a), dtype=dt)
        case _ if (
                    len(args) > 0
                    and all(isinstance(v, Sequence) for v in args)
                    and all(
                            v is None or isinstance(v, int)
                            for v in args[0]
                        )
                ):
            dt = find_dtype(v for a in args for v in a)
            viter = (
                    v or 0
                    for a in args
                    for v in a
                )
            values = np.fromiter(viter, dtype=dt)

    total = values.sum()
    if abs(total) < EPSILON:
        return (fn(0, 0, 0, 0),)
    results = np.zeros(len(values), dtype=values.dtype)
    for i in range(len(values)):
        results[i] = _map_range(values[i]/total, limit)
    s = results.sum()
    delta = limit - int(s)
    if delta > 0:
        adj = 1
    elif delta < 0:
        adj = -1
    if delta != 0:
        errs = sorted(
            (
                (i, abs(float(r)/limit - v))
                for i, (r,v) in enumerate(zip(results, values))
                if results[i] != 0
            ),
            key=lambda a: a[1])
        for i, _ in islice(errs, 0, abs(delta)):
            results[i] = int(results[i]) + adj
    return tuple(fn(*chunk) for chunk in chunk4i(results, num_values=len(results)))


def _map_range(value: float|int, limit: int) -> int:
    '''
    Map a value from 0..1.0. The value is clamped to the input range.
    '''
    value = max(0.0, min(1.0, value))
    return round(float(value) * limit)


def find_dtype(values: Iterable[int]) -> np.dtype:
    '''
    Find the smallest numpy dtype that can hold the values.
    '''
    if not isinstance(values, (Iterable, Generator)):
        raise TypeError('values must be an iterable')
    dt = np.uint8
    for v in values:
        if isinstance(v, float):
            return np.float32
        try:
            v = int(v)
        except TypeError:
            raise ValueError("Values must be convertable to int")
        if v < 0:
            raise ValueError('values must be non-negative')
        if v > 255:
            dt = np.uint16
        if v > 65535:
            return np.uint32
    return dt


AttributeDataItem: TypeAlias = int|float|tuple[int, ...]|tuple[float, ...]|np.ndarray[tuple[int, ...], Any] 

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

AttributeDataSequence: TypeAlias = (
    Sequence[int]
    |Sequence[float]
    |Sequence[tuple[int, ...]]
    |Sequence[tuple[float, ...]]
    |Sequence[np.ndarray[tuple[int, ...], Any]]
)
'''
Sequence of attribute data in various formats. Lists of:
- integers
- floats
- tuples of integers
- tuples of floats
- numpy arrays of integers
- numpy arrays of floats
'''
