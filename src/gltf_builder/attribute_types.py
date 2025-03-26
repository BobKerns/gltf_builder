'''
Types describing attribute values.
'''

from typing import NamedTuple, TypeAlias, Literal, overload, Optional, Any
from math import sqrt
from collections.abc import Sequence

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
def uv(x: float, y: float) -> _Uv: ...
def uv(x: Optional[float|Point]=None,
            y: Optional[float]=None) -> _Uv:
    '''
    Return a canonicalized Uv texture coordinate object.

    Parameters
    ----------
        x: The x value of the texture coordinate, or a Uv object of some type.
        y: The y value of the texture coordinate.

    Returns
    -------
        A `_Uv` object (a `NamedTuple`)
    '''
    match x:
        case None:
            return _Uv(0.0, 0.0)
        case _Uv():
            return x
        case x, y:
            return _Uv(float(x), float(y))
        case np.ndarray() if x.shape == (2,):
            return _Uv(float(x[0]), float(x[1]))
        case float():
            return _Uv(float(x), float(y))
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
    match x:
        case None:
            return _Vector2(0.0, 0.0)
        case _Vector2():
            return x 
        case x, y:
            return _Vector2(float(x), float(y))
        case np.ndarray() if x.shape == (2,):
            return _Vector2(float(x[0]), float(x[1]))
        case float():
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
    match x:
        case None:
            return _Vector3(0.0, 0.0, 0.0)
        case _Vector3():
            return x
        case x, y, z:
            return _Vector3(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Vector3(float(x[0]), float(x[1]), float(x[2]))
        case float():
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
    match x:
        case None:
            return _Vector4(0.0, 0.0, 0.0, 0.0)
        case _Vector4():
            return x
        case x, y, z, w:
            return _Vector4(float(x), float(y), float(z), float(w))
        case np.ndarray() if x.shape == (4,):
            return _Vector4(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        case float():
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
    match x:
        case None:
            return _Scale(1.0, 1.0, 1.0)
        case float() if y is None and z is None:
            return _Scale(x, x, x)
        case _Scale():
            return x
        case x, y, z:
            return _Scale(float(x), float(y), float(z))
        case np.ndarray() if x.shape == (3,):
            return _Scale(float(x[0]), float(x[1]), float(x[2]))
        case _ if y is not None and z is not None:
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
    match x:
        case _Tangent():
            return x
        case (x, y, z, w):
            if w not in (-1, 1):
                raise ValueError(f'Invalid tangent w value {w=} must be -1 or 1')
            return _Tangent(float(x), float(y), float(z), float(w))
        case np.ndarray() if x.shape == (4,):
            return _Tangent(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
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
    match r:
        case None:
            return RGB(0.0, 0.0, 0.0)
        case RGB():
            return r
        case RGBA():
            return r
        case RGB8():
            return RGB(clamp(r.r / 255), clamp(r.g / 255), clamp(r.b / 255))
        case RGBA8():
            return RGBA(clamp(r.r / 255), clamp(r.g / 255), clamp(r.b / 255), clamp(r.a / 255))
        case RGB16():
            return RGB(clamp(r.r / 65535), clamp(r.g / 65535), clamp(r.b / 65535))
        case RGBA16():
            return RGBA(clamp(r.r / 65535), clamp(r.g / 65535), clamp(r.b / 65535), clamp(r.a / 65535))
        case r, g, b:
            return RGB(clamp(r), clamp(g), clamp(b))
        case r, g, b, a:
            return RGBA(clamp(r), clamp(g), clamp(b), clamp(a))
        case np.ndarray() if r.shape == (3,):
            return RGB(clamp(r[0]), clamp(r[1]), clamp(r[2]))
        case np.ndarray() if r.shape == (4,):
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
    match r:
        case None:
            return rgb(0, 0, 0)
        case (r, g, b):
            return rgb(scale(r), scale(g), scale(b))
        case (r, g, b, a):
            return rgba(scale(r), scale(g), scale(b), scale(a))
        case rgb()|rgba():
            return r
        case RGB()|RGBA():
            return rgb(scale(r.r), scale(r.g), scale(r.b))
        case RGB8()|RGBA8():
            return _color(r.r / 255, r.g / 255, r.b / 255, r.a / 255,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
                          )
        case RGB16()|RGBA16():
            return _color(r.r / 65535, r.g / 65535, r.b / 65535, r.a / 65535,
                          limit=limit,
                          rgb=rgb,
                          rgba=rgba,
                          )
        case float()|0|1 if a is not None:
            return rgba(scale(r), scale(g), scale(b), scale(a))
        case float()|0|1:
            return rgb(scale(r), scale(g), scale(b))
        case np.ndarray() if r.shape == (3,):
            return rgb(scale(r[0]), scale(r[1]), scale(r[2]))
        case np.ndarray() if r.shape == (4,):
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
def joint(x: int, y: int, z: int) -> _Joint: ...
def joint(x: Optional[int|Point]=None,
        y: Optional[int]=None,
        z: Optional[int]=None,
        w: Optional[int]=None
    ) -> _Joint:
    match x:
        case None:
            return _Joint(0, 0, 0, 0)
        case _Joint():
            return x
        case x, y, z, w:
            return _Joint(int(x), int(y), int(z), int(w))
        case np.ndarray() if x.shape == (4,):
            return _Joint(int(x[0]), int(x[1]), int(x[2]), int(x[3]))
        case float():
            return _Joint(int(x), int(y), int(z), int(w))
        case _:
            raise ValueError('Invalid vector4')    



@overload
def weight() -> _Weightf: ...
@overload
def weight(v: Weight, /) -> _Weightf: ...
@overload
def weight(x: float|int, y: float|int, z: float|int, w: float|int) -> _Weightf: ...
def weight(x: Optional[float|int]=None,
        y: Optional[float|int]=None,
        z: Optional[float|int]=None,
        w: Optional[float|int]=None
    ) -> _Weightf:
    def normalize(x, y, z, w):
        total = x + y + z + w
        if total == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        return x / total, y / total, z / total, w / total
    match x:
        case None:
            return _Weightf(0.0, 0.0, 0.0, 0.0)
        case _Weightf():
            return x
        case x, y, z, w :
            return _Weightf(*normalize(x, y, z, w))
        case np.ndarray() if x.shape == (4,):    
            return _Weightf(*normalize(x[0], x[1], x[2], x[3]))
        case _:
            raise ValueError('Invalid weight') 

@overload
def weight8() -> _Weight8: ...
@overload
def weight8(v: Weight, /) -> _Weight8: ...
@overload
def weight8(x: float|int, y: float|int, z: float|int, w: float|int) -> _Weight8: ...
def weight8(x: Optional[float|int]=None,
        y: Optional[float|int]=None,
        z: Optional[float|int]=None,
        w: Optional[float|int]=None
    ) -> _Weight8:
    def normalize(x, y, z, w):
        total = x + y + z + w
        if total == 0.0:
            return 0, 0, 0, 0
        x, y, z, w = (
            int(x * 255),
            int(y * 255),
            int(z * 255),
            int(w * 255)
        )
        total = x + y + z + w
        diff = 255 - total
        if diff > 0:
            x += 1
        if diff > 1:
            y += 1
        if diff > 2:
            z += 1
        if diff > 3:
            w += 1
        return x, y, z, w
    match x:
        case None:
            return _Weight8(0, 0, 0, 0)
        case _Weight8():
            return x
        case x, y, z, w:
            return _Weight8(*normalize(x, y, z, w))
        case np.ndarray() if x.shape == (4,):    
            return _Weight8(*normalize(x[0], x[1], x[2], x[3]))
        case _:
            raise ValueError('Invalid weight') 

@overload
def weight16() -> _Weight16: ...
@overload
def weight16(v: Weight, /) -> _Weight16: ...
@overload
def weight16(x: float|int, y: float|int, z: float|int, w: float|int) -> _Weight16: ...
def weight16(x: Optional[float|int]=None,
        y: Optional[float|int]=None,
        z: Optional[float|int]=None,
        w: Optional[float|int]=None
    ) -> _Weight16:
    '''
    
    '''
    def normalize(x, y, z, w):
        total = x + y + z + w
        if total == 0.0:
            return 0, 0, 0, 0
        x, y, z, w = (
            int(x * 65535),
            int(y * 65535),
            int(z * 65535),
            int(w * 65535)
        )
        total = x + y + z + w
        diff = 65535 - total
        if diff > 0:
            x += 1
        if diff > 1:
            y += 1
        if diff > 2:
            z += 1
        if diff > 3:
            w += 1
        return x, y, z, w
    match x:
        case None:
            return _Weight16(0, 0, 0, 0)
        case _Weight16():
            return x
        case x, y, z, w:
            return _Weight16(*normalize(x, y, z, w))
        case np.ndarray() if x.shape == (4,):    
            return _Weight16(*normalize(x[0], x[1], x[2], x[3]))
        case _:
            raise ValueError('Invalid weight') 


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
