'''
Types describing attribute values.
'''

from typing import NamedTuple, TypeAlias, Literal, overload, Optional, Any
from math import sqrt
from collections.abc import Iterable, Callable, Sequence
from itertools import islice

import numpy as np

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
        return sum(v*v for v in self) > EPSILON*EPSILON
    
    @property
    def length(self):
        return sqrt(sum(v*v for v in self))
    
    def __add__(self, other: 'VectorLike') -> 'VectorLike':
        return type(self)(*(a+b for a,b in zip(self, other)))
    
    def __sub__(self, other: 'VectorLike') -> 'VectorLike':
        return type(self)(*(a-b for a,b in zip(self, other)))
    
    def __mul__(self, other: 'float|VectorLike') -> 'VectorLike':
        match other:
            case float():
                return type(self)(*(a*other for a in self))
            case VectorLike() if len(self) == len(other):
               return sum(a*b for a,b in zip(self, other))
            case _:
                raise ValueError('Invalid vector multiplication')

    def __rmul__(self, other: float) -> 'VectorLike':
        return type(self)(*(a*other for a in self))

    def __truediv__(self, other: float) -> 'VectorLike':
        return type(self)(*(a/other for a in self))
    
    def dot(self, other: 'VectorLike') -> float:
        return sum(a*b for a,b in zip(self, other))


class PointLike(NamedTuple):
    def distance(self, other: 'PointLike') -> float:
        return sqrt(sum((a-b)**2 for a,b in zip(self, other)))


class _Vector2(_Floats2, VectorLike):
    '''
    A 2D vector, x and y.
    '''
    x: float
    y: float


class _CrossProduct(NamedTuple):
    '''
    A cross product of two vectors.
    '''
    x: float
    y: float
    z: float

class _Vector3(_Floats3, VectorLike):
    '''
    A 3D vector, x, y, and z.
    '''
    x: float
    y: float
    z: float


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
    
    def cross(self, other: 'Vector3') -> 'Vector3':
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
    Scale factors for a 3D object.
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
        return (x*x + y+y + z*z) > EPSILON*EPSILON
    
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

class _Uvf(VectorLike, PointLike):
    '''
    A 2D texture coordinate (in U and V) in floating point.
    '''
    u: float
    v: float

class _UvX(VectorLike, PointLike):
    '''
    A 2D texture coordinate (in U and V) in normalied ints.
    '''
    u: int
    v: int

class _Uv8(_UvX):
    '''
    A 2D texture coordinate (in U and V) in 8-bit integers.
    '''
    pass


class _Uv16(_UvX):
    '''
    A 2D texture coordinate (in U and V) in 16-bit integers.
    '''
    pass

_Uv: TypeAlias = _Uvf|_Uv8|_Uv16
'''
A 2D texture coordinate (in U and V) in floating point, 8-bit, or 16-bit integers.
'''

class _RGBF(NamedTuple):
    '''
    A RGB color with floating point values between 0.0 and 1.0, inclusive.
    '''
    r: float
    g: float
    b: float

class _RGBAF(_RGBF):
    '''
    A RGBA color with floating point values between 0.0 and 1.0, inclusive.
    '''
    a: float

class RGB(_RGBF):
    '''
    A RGB color with floating point values between 0.0 and 1.0, inclusive.
    '''
    pass

class RGBA(_RGBAF):
    '''
    A RGBA color with floating point values between 0.0 and 1.0, inclusive.
    '''
    pass


class _RGBI(NamedTuple):
    r: int
    g: int
    b: int

class _RGBAI(_RGBI):
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
    pass

class RGBA16(_RGBAI):
    '''
    An RGBA color with 16-bit integer values between 0 and 255, inclusive.
    '''
    pass

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

NPIVector32: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint32]
NPIVector16: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint16]
NPIVector8: TypeAlias = np.ndarray[tuple[Literal[4]], np.uint8]
NPIVector16s: TypeAlias = np.ndarray[tuple[Literal[4]], np.int16]
NPIVector8s: TypeAlias = np.ndarray[tuple[Literal[4]], np.int8]

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
Joint: TypeAlias = _Joint|IVec4|NPIVector8|NPIVector16
Weight: TypeAlias = _Weight|Vec4|NP4Vector|IVec4|NPIVector8|NPIVector16

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
def point(x: float, y: float, z: float) -> _Point: ...
def point(x: Optional[float|Point]=None,
        y: Optional[float]=None,
        z: Optional[float]=None) -> _Point:
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
    match x:
        case None:
            return _Point(0.0, 0.0, 0.0)
        case _Point():
            return x
        case x, y, z:
            return _Point(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Point(float(x[0]), float(x[1]), float(x[2]))
        case float():
            return _Point(float(x), float(y), float(z))
        case _:
            raise ValueError('Invalid point')
        

@overload
def uv() -> _Uv: ...
@overload
def uv(v: Uv, /) -> _Uv: ...
@overload
def uv(u: float, v: float) -> _Uv: ...
def uv(u: Optional[float|Point]=None,
        v: Optional[float]=None) -> _Uv:
    '''
    Return a canonicalized Uv texture coordinate object.

    Parameters
    ----------
        u: The x value of the texture coordinate, or a Uv object of some type.
        v: The y value of the texture coordinate.

    Returns
    -------
        A `_Uv` object (a `NamedTuple`)
    '''
    match u, v:
        case None, None:
            return _Uv(0.0, 0.0)
        case _Uv(), None:
            return u
        case (float()|int(), float()|int()), None:
            return _Uv(float(u[0]), float(u[1]))
        case np.ndarray(), None if u.shape == (2,):
            return _Uv(float(u[0]), float(u[1]))
        case float()|int(), float()|int():
            return _Uv(float(u), float(v))
        case _:
            raise ValueError('Invalid vector2')     


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
        case float(), None, None if y is None and z is None:
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
        case _Tangent(), None, None, None:
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
def color(c: Color, /) -> RGB|RGBA: ...
@overload
def color(r: float, g: float, b: float, a: float) -> RGBA: ...
@overload
def color(r: float, g: float, b: float) -> RGB:
    ...
def color(r: Optional[float01|Color]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None,
    ) -> RGB|RGBA:
    def clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))
    match r, g, b, a:
        case None, None, None, None:
            return RGB(0.0, 0.0, 0.0)
        case RGB(), None, None, None:
            return r
        case RGBA(), None, None, None:
            return r
        case RGB8(), None, None, None:
            return RGB(clamp(r.r / 255), clamp(r.g / 255), clamp(r.b / 255))
        case RGBA8(), None, None, None:
            return RGBA(clamp(r.r / 255), clamp(r.g / 255), clamp(r.b / 255), clamp(r.a / 255))
        case RGB16(), None, None, None:
            return RGB(clamp(r.r / 65535), clamp(r.g / 65535), clamp(r.b / 65535))
        case RGBA16(), None, None, None:
            return RGBA(clamp(r.r / 65535), clamp(r.g / 65535), clamp(r.b / 65535), clamp(r.a / 65535))
        case (r, g, b), None, None, None:
            return RGB(clamp(r), clamp(g), clamp(b))
        case (r, g, b, a), None, None, None:
            return RGBA(clamp(r), clamp(g), clamp(b), clamp(a))
        case np.ndarray(), None, None, None, if r.shape == (3,):
            return RGB(clamp(r[0]), clamp(r[1]), clamp(r[2]))
        case np.ndarray(), None, None, None if r.shape == (4,):
            return RGBA(clamp(r[0]), clamp(r[1]), clamp(r[2]), clamp(r[3]))
        case _:
            raise ValueError('Invalid color')
        

def _color(r: Optional[float01]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None, /,
         limit: int = 255,
         rgb: type[RGB8|RGB16] = RGB8,
         rgba: type[RGBA8|RGBA16] = RGBA8,
    ) -> RGB8|RGBA8|RGB16|RGBA16:
    def scale(v: float) -> int:
        return int(max(0, min(limit, round(v * limit))))
    match r, g, b, a:
        case None, None, None, None:
            return rgb(0, 0, 0)
        case (r, g, b), None, None, None:
            return rgb(scale(r), scale(g), scale(b))
        case (r, g, b, a), None, None, None:
            return rgba(scale(r), scale(g), scale(b), scale(a))
        case rgb()|rgba(), None, None, None:
            return r
        case RGB()|RGBA(), None, None, None:
            return rgb(scale(r.r), scale(r.g), scale(r.b))
        case RGB8()|RGBA8(), None, None, None:
            return _color(r.r / 255, r.g / 255, r.b / 255, r.a / 255,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
                          )
        case RGB16()|RGBA16(), None, None, None:
            return _color(r.r / 65535, r.g / 65535, r.b / 65535, r.a / 65535,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
                          )
        case float()|0|1, float()|0|1, float()|0|1, float()|0|1:
            return rgba(scale(r), scale(g), scale(b), scale(a))
        case float()|0|1, float()|0|1, float()|0|1:
            return rgb(scale(r), scale(g), scale(b))
        case np.ndarray(), None, None, None if r.shape == (3,):
            return rgb(scale(r[0]), scale(r[1]), scale(r[2]))
        case np.ndarray(), None, None, None if r.shape == (4,):
            return rgb(scale(r[0]), scale(r[1]), scale(r[2]), scale(r[3]))
        case _:
            raise ValueError('Invalid color')


@overload
def color8() -> RGB8: ...
@overload
def color8(c: RGB|NP3Vector|Vec3) -> RGB8: ...
@overload
def color8(c: RGBA|NP4Vector|Vec4) -> RGBA8: ...
@overload
def color8(c: Color, /) -> RGB8|RGBA8: ...
@overload
def color8(r: float01, g: float01, b: float01, a: float01) -> RGBA8: ...
@overload
def color8(r: float01, g: float01, b: float01) -> RGB8: ...
def color8(r: Optional[float01|Color]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None,
    ) -> RGB8|RGBA8:
    return _color(r, g, b, a,
                  limit=255,
                  rgb=RGB8,
                  rgba=RGBA8,
                  )


@overload
def color16() -> RGB16: ...
@overload
def color16(c: RGB|NP3Vector|Vec3) -> RGB16: ...
@overload
def color16(c: RGBA|NP4Vector|Vec4) -> RGBA16: ...
@overload
def color16(c: Color, /) -> RGB16|RGBA16: ...
@overload
def color16(r: float01, g: float01, b: float01, a: float01) -> RGBA16: ...
@overload
def color16(r: float01, g: float01, b: float01) -> RGB16: ...
def color16(r: Optional[float|Literal[0,1]|Color]=None,
         g: Optional[float01]=None,
         b: Optional[float01]=None,
         a: Optional[float01]=None,
    ) -> RGB16|RGBA16:
    return _color(r, g, b, a,
                  limit=65535,
                  rgb=RGB16,
                  rgba=RGBA16,
                  )

def rgb8(r: int, g: int, b: int) -> RGB8:
    def clamp(v: int) -> int:
        return max(0, min(255, v))
    return RGB8(clamp(r), clamp(g), clamp(b))


def rgb16(r: int, g: int, b: int) -> RGB8:
    def clamp(v: int) -> int:
        return max(0, min(65536, v))
    return RGB8(clamp(r), clamp(g), clamp(b))


@overload
def joint() -> _Joint: ...
@overload
def joint(v: Joint, /) -> _Joint: ...
@overload
def joint(x: int) -> _Joint: ...
@overload
def joint(x: int, y: int) -> _Joint: ...
@overload
def joint(x: int, y: int, z: int) -> _Joint: ...
@overload
def joint(x: int, y: int, z: int, w: int) -> _Joint: ...
def joint(x: Optional[int|tuple[int]|tuple[int,int]|tuple[int,int,int]|tuple[int,int,int,int]]=None,
        y: Optional[int]=None,
        z: Optional[int]=None,
        w: Optional[int]=None
    ) -> _Joint:
    match x, y, z, w:
        case _Joint(), None, None, None:
            return x
        case tuple():
            match x:
                case (int(), int()|None, int()|None, int()|None):
                    return _Joint(int(x), int(x[1] or 0), int(x[2] or 0), int(x[3] or 0))
                case (int(), int()|None, int()|None):
                    return _Joint(int(x), int(x[1] or 0), int(x[2] or 0), 0)
                case (int(), int()|None, int()|None):
                    return _Joint(int(x), int(x[1] or 0), 0, 0)
                case (int(),):
                    return _Joint(int(x[0]), 0, 0, 0)
        case np.ndarray(), None, None, None if x.shape == (4,):
            return _Joint(int(x[0]), int(x[1]), int(x[2]), int(x[3]))
        case np.ndarray(), None, None, None if x.shape == (43):
            return _Joint(int(x[0]), int(x[1]), int(x[2]), 0)
        case np.ndarray(), None, None, None if x.shape == (2,):
            return _Joint(int(x[0]), int(x[1]), 0, 0)
        case np.ndarray(), None, None, None if x.shape == (1,):
            return _Joint(int(x[0]), 0, 0, 0)
        case float(), float()|None, float()|None, float()|None:
            return _Joint(int(x), int(y or 0), int(z or 0), int(w or 0))
        case _:
            raise ValueError('Invalid vector4')    


def chunk4(values: Iterable[float|int|None]) -> Iterable[tuple[float, float, float, float]]:
    '''
    Chunk an iterable of values into groups of 4.
    '''
    count = len(values) // 4
    more = len(values) % 4
    for i in range(count):
        yield (
            float(values[i*4] or 0.0),
            float(values[i*4+1] or 0.0),
            float(values[i*4+2] or 0.0),
            float(values[i*4+3] or 0.0),
        )
    match more:
        case 0:
            return
        case 1:
            yield (
                float(values[-1] or 0.0),
                0.0,
                0.0,
                0.0,
            )
        case 2:
            yield (
                float(values[-2] or 0.0),
                float(values[-1] or 0.0),
                0.0,
                0.0,
            )
        case 3:
            yield (
                float(values[-3] or 0.0),
                float(values[-2] or 0.0),
                float(values[-1] or 0.0),
                0.0,
            )


def chunk4i(values: Iterable[int|None]) -> Iterable[tuple[int, int, int, int]]:
    '''
    Chunk an iterable of values into groups of 4.
    '''
    count = len(values) // 4
    more = len(values) % 4
    for i in range(count):
        yield (
            int(values[i*4] or 0),
            int(values[i*4+1] or 0),
            int(values[i*4+2] or 0),
            int(values[i*4+3] or 0),
        )
    match more:
        case 0:
            return
        case 1:
            yield (
                int(values[-1] or 0),
                0,
                0,
                0,
            )
        case 2:
            yield (
                int(values[-2] or 0),
                int(values[-1] or 0),
                0,
                0,
            )
        case 3:
            yield (
                int(values[-3] or 0),
                int(values[-2] or 0),
                int(values[-1] or 0),
                0,
            )


@overload
def weight() -> tuple: ...
@overload
def weight(v: Weight, /) -> tuple[_Weightf, ...]: ...
@overload
def weight(*args: float01|None) -> tuple[_Weightf, ...]: ...
def weight(*args: float01|None) -> tuple[_Weightf, ...]:
    def reweigh(values: tuple[float|int|None]|np.ndarray) -> tuple[_Weightf, ...]:
        if isinstance(values, np.ndarray):
            total = values.sum()
        else:
            total = sum(
                v or 0 
                for v in values
            )
        if abs(total) < EPSILON:
            return (0.0,) * len(args)
        return tuple(_Weightf(*(c/total for c in chunk)) for chunk in chunk4(values))
    match args:
        case (_Weightf(), *more) if all(isinstance(v, _Weightf) for v in more):
            return args
        case tuple(values) if all(v is None or isinstance(v, (float, int)) for v in values):
            return reweigh(values)
        case (tuple(values),) if all(v is None or isinstance(v, (float, int)) for v in values):
            return reweigh(values)
        case (np.ndarray(),) if args[0].dtype == np.float32:
            return reweigh(values[0])
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


def _weighti(args: tuple[float|int|None],
             limit: int,
             fn: Callable[[tuple[int, int, int, int]], Any]) -> tuple[Any, ...]:
    def reweigh(values: tuple[float|int|None]|np.ndarray) -> tuple[Any, ...]:
        if isinstance(values, np.ndarray):
            total = values.sum()
        else:
            total = sum(v or 0 for v in values)
        if abs(total) < EPSILON:
            return (0,) * len(args)
        results = [
            _map_range((v or 0)/total, limit)
            for v in values
        ]
        s = sum(results)
        delta = limit - s
        if delta > 0:
            adj = 1
        elif delta < 0:
            adj = -1
        if delta != 0:
            errs = sorted(((i, abs(float(r)/limit - v)) for i, (r,v) in enumerate(zip(results, values))), key=lambda a: a[1])
            for i, _ in islice(errs, 0, abs(delta)):
                results[i] += adj
        return tuple(fn(*chunk) for chunk in chunk4i(results))
    match args:
        case tuple(values) if all(isinstance(v, _Weight8) for v in values) and limit == 255:
            return args
        case tuple(values) if all(isinstance(v, _Weight16) for v in values) and limit == 65535:
            return args
        # We don't permit interconverting between integer formats because of loss of precision.
        # Converting to an integer format is a one-way ticket.
        case tuple(values) if all(v is None or isinstance(v, (float, int)) for v in values):
            return reweigh(values)
        case (tuple(values),) if all(v is None or isinstance(v, (float, int)) for v in values):
            return reweigh(values)
        case (np.ndarray(),) if args[0].dtype == np.dtype(np.float32):
            return reweigh(values[0])
        case _:
            raise ValueError('Invalid weight') 


def _map_range(value: float|int, limit: int) -> int:
    '''
    Map a value from 0..1.0. The value is clamped to the input range.
    '''
    value = max(0.0, min(1.0, value))
    return round(float(value) * limit)


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
