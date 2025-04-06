'''
Tests for type constructors and types in attribute_types.py.
'''

from collections.abc import Callable
from typing import Optional, Literal, Any
from functools import wraps
from inspect import signature
from math import sqrt, cos, pi


import numpy as np

from pytest import approx, mark, raises, fixture

from gltf_builder.core_types import ByteSize
from gltf_builder.attribute_types import (
    joints, vector2, vector3, vector4, tangent, scale, point, uv, joint,
    weight, weight8, weight16,
    color, rgb8,  rgb16, RGB, RGBA, RGB8, RGBA8, RGB16, RGBA16, 
    Vector2, Vector3, Vector4,
    _Weightf, _Weight8, _Weight16,
    Tangent, Scale, Point, PointLike, UvfFloat, Uv16, Uv8,
    Joint, _Joint8, _Joint16,
    EPSILON,
)

def case_tuple(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a tuple of data.
    '''
    return tuple(data)


def case_obj(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a tuple of data.
    '''
    return cnst(*data)

def tuple_of(c: Callable[[Callable[..., Any]], Callable[..., Any]],
             size: Optional[ByteSize]=None):
    '''
    Test a type constructor with a tuple of data.
    '''
    @wraps(c)
    def tupleobj(cnst: Callable[..., Any], data: tuple, /,
            size: Optional[ByteSize]=size):
        kwargs = {}
        if size is not None and signature(c).parameters.get('size') is not None:
            kwargs['size'] = size
        return (c(cnst, data, **kwargs),)
    tupleobj.__name__ = f'{c.__name__}_tuple'
    tupleobj.__qualname__ = tupleobj.__name__
    return tupleobj


def case_weight(cnst: Callable[..., Any], data: tuple, /,
                size: ByteSize):
    '''
    Test a type constructor with a tuple of data.
    '''
    total = sum(data)
    match size:
        case 1:
            def scale(v: float):
                return round((float(v)/total)*255)
            zero = 0
        case 2:
            def scale(v: float):
                return round((float(v)/total)*65535)
            zero = 0
        case 4:
            def scale(v):
                return float(v)/total
            zero = 0.0
    if abs(total) < EPSILON:
        return cnst(*(zero for _ in data))
    return cnst(*(scale(d) for d in data))

def case_numpy(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a numpy array.
    '''
    return (np.array(data, np.float32),)

def case_numpy8(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a numpy array.
    '''
    return (np.array(data, np.uint8),)


def case_numpy16(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a numpy array.
    '''
    return (np.array(data, np.uint16),)

case_numpy_tuple = tuple_of(case_numpy)
case_tuple_tuple = tuple_of(case_tuple)
case_obj_tuple = tuple_of(case_obj)
case_weight_tuple = tuple_of(case_weight)

def case_uvf(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a _UvF instance.
    '''
    return (UvfFloat(*(float(d) for d in data)),)


def case_uv8(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a _UvF instance.
    '''
    return (Uv8(*(round(float(d) * 255) for d in data)),)


def case_uv16(cnst: Callable[..., Any], data: tuple,):
    '''
    Test a type constructor with a _UvF instance.
    '''
    return (Uv16(*(round(float(d) * 65535) for d in data)),)


def validator_fn(tcase: Callable[[Callable[..., Any], tuple], tuple]):
    @wraps(tcase)
    def validator(
            t: type,
            cnst: Callable[..., Any],
            min_data: int,
            max_data: int,
            ndata: int,
            data: tuple,
            exc: type[Exception] | None,
            size: ByteSize|Literal['inf']=4,
            to_int: bool=False,
            epsilon: Optional[float]=None,
    ):
        '''
        Validate the type constructor against the supplied data.
        The data is passed to the constructor and the result is compare
        to the expected type and value.

        The expected data is constructed by one of the following functions:
        - case_tuple: The data is passed as a tuple to the constructor.
        - case_obj: The data is passed as an object of type _t_.
        - case_numpy: The data is passed as a numpy array to the constructor.

        If the constructor in cnst takes a _size_ keyword argument, it is passed.

        An exception is expected if ndata is outside the range of
        min_data and max_data, or if exc is supplied.
        
        Parameters
        ----------
        t : type
            The expected type of the result.
        cnst : Callable[..., Any]
            The constructor function to test.
        min_data : int
            The minimum number of data elements accepted by the constructor.
        max_data : int
            The maximum number of data elements accepted by the constructor.
        ndata : int
            The number of data elements to pass to the constructor.
        data : tuple
            The data to pass to the constructor.
        exc : type[Exception] | None
            The expected exception type, or None if no exception is expected.
        size : ByteSize|Literal['inf'], optional keyword argument
            The size of the data in bytes, by default 4.
            The values are:
            - 1: 1 byte integer
            - 2: 2 bytes integer
            - 4: 4 bytes float32 between 0 and 1
            - 'inf': unlimited range float
        to_int : bool, optional keyword argument
            If True, the data is converted to int, by default False.
            This is used for the color types, where the data is
            converted to int in the range of 0 to 255 or 0 to 65535, based
            on the _size_ parameter.
        '''
        match size:
            case 1:
                def elt_type(x):
                    x = min(1.0, max(0.0, float(x)))
                    return round(x*255)
            case 2:
                def elt_type(x):
                    x = min(1.0, max(0.0, float(x)))
                    return round(x*65535)
            case 4|'inf':
                elt_type = float
            case _:
                raise ValueError(f"Invalid size: {size}")
        # Override epsilon for cases where the conversion loses resolution.
        # This is the case for uv8 and uv16, where the conversion to int
        # loses resolution.
        match size, tcase:
            case 2,  tc if tc is case_uv8:
                epsilon = 65535.0/255
            case 4|'inf', tc if tc is case_uv16:
                epsilon = 1.0/65535
            case 4|'inf', tc if tc is case_uv8:
                epsilon = 1.0/255
            case 1|2, tc if tc in (case_numpy, case_numpy_tuple):
                # Numpy arrays are not exact, so we use a larger epsilon.
                epsilon = 1.0
            case 4|'inf', tc if tc in (case_numpy, case_numpy_tuple):
                epsilon = 2*(float(np.float32(0.3)) - 0.3)
        kwargs = {}
        if signature(cnst).parameters.get('size') is not None:
            kwargs['size'] = size
        if isinstance(exc, type) and issubclass(exc, Exception):
            with raises((exc, TypeError)):
                r = cnst(*data, **kwargs)
            return
        data = data[0:ndata]
        if 0 < len(data) < min_data or len(data) > max_data:
            with raises((ValueError, TypeError)):
                r = cnst(*data, **kwargs)
            return
        expected = t(*[elt_type(float(n)) for n in data])
        match expected:
            case (tuple()|np.ndarray(),):
                # If the constructor under test was given a tuple w/ data, it is
                # expected to return it unwrapped.
                expected = expected[0]
        if to_int:
            data = tuple(elt_type(n) for n in data)
        data = tcase(t, data)
        r = cnst(*data, **kwargs)
        assert tuple(r) == approx(tuple(expected), abs=epsilon)
        assert type(r) is t
        match size:
            case 1:
                assert all(isinstance(v, int) for v in r)
                assert all(0 <= v <= 255 for v in r)
            case 2:
                assert all(isinstance(v, int) for v in r)
                assert all(0 <= v <= 65535 for v in r)
            case 4:
                assert all(isinstance(v, float) for v in r)
                assert all(0 <= v <= 1.0 for v in r)
            case 'inf':
                assert all(isinstance(v, float) for v in r)
            case _:
                raise ValueError(f"Invalid size: {size}")
    validator.__name__ = f'validator_{tcase.__name__}'
    validator.__qualname__ = validator.__name__
    return validator
# Input cases

@fixture(params=[
        case_tuple,
        case_tuple_tuple,
        case_obj,
        case_obj_tuple,
        case_numpy,
    ])
def validator_nested(request):
    '''
    Validator for types and constructor functions.

    '''
    return validator_fn(request.param)
    


@fixture(params=[case_tuple,
                         case_obj,
                         ])
def validator_flat(request):
    '''
    Validator for types and constructor functions.

    '''
    return validator_fn(request.param)


    
@fixture(params=[case_tuple,
                         case_uvf,
                         case_uv8,
                         case_uv16,
                         case_numpy,
                         ])
def validator_uv(request):
    '''
    Validator for types and constructor functions.

    '''
    return validator_fn(request.param)
    

SCALED_PARAMS = [
    ((0.4, 0.3, 0.2, 0.1), None),
    ((0.0, 0.0, 0.0, 0.0), None),
    ((0, "foo", 0, 0,), ValueError),
    ((0, 0, 0, 0, 0), ValueError),
]

VEC_PARAMS = [
    ((1.0, 2.0, 3.0, 4.0), None),
    *SCALED_PARAMS,
]


# Constructors and constructed types, floating point unlimited range.
@mark.parametrize('cnst, ndata, t', [
    (vector2, 2, Vector2),
    (vector3, 3, Vector3),
    (vector4, 4, Vector4),
    (scale, 3, Scale),
    (point, 3, Point),
])
# Data and expected exceptions.
@mark.parametrize('data, exc', VEC_PARAMS)
def test_type_constructors(
                        validator_nested,
                        t ,
                        ndata,
                        cnst,
                        data,
                        exc,
                    ):
    '''
    Test the type constructors
    '''
    validator_nested(t, cnst, ndata, ndata, ndata, data, exc, size='inf')


def test_origin():
    p = point()
    assert p == Point(0.0, 0.0, 0.0)


@mark.parametrize('cnst, data, err', [
    (vector2, (1.0,), ValueError),
    (vector2, (1.0, 2.0, 3.0), TypeError),
    (vector2, (1.0, "foo"), ValueError),
    (vector3, (1.0, 2.0), ValueError),
    (vector3, (1.0, 2.0, 3.0, 4.0), TypeError),
    (vector3, (1.0, 2.0, "foo"), ValueError),
    (vector4, (1.0, 2.0, 3.0), ValueError),
    (vector4, (1.0, 2.0, 3.0, 4.0, 5.0), TypeError),
    (vector4, (1.0, 2.0, 3.0, "foo"), ValueError),
    (uv, (1.0,), ValueError),
    (uv, (1.0, "foo"), ValueError),
    (uv, (1.0, 2.0, 3.0), TypeError),
    (scale, ("foo"), ValueError),
    (scale, (1.0, 2.0, 3.0, 4.0), TypeError),
    (point, (1.0,), ValueError),
    (point, (1.0, 2.0), ValueError),
    (point, (1.0, 2.0, 3.0, 4.0), TypeError),
    (point, (1.0, 2.0, "foo"), ValueError),
])
def test_type_constructor_exceptions(
                        cnst,
                        data,
                        err,
                    ):
    '''
    Test the type constructors
    '''
    with raises(err):
        cnst(*data)


# Constructors and constructed types, floating point unlimited range.
@mark.parametrize('cnst, ndata, t, size', [
    (uv, 2, UvfFloat, 'inf',),
    (uv, 2, Uv16, 2,),
    (uv, 2, Uv8, 1,),
])
# Data and expected exceptions.
@mark.parametrize('data, exc', SCALED_PARAMS)
def test_uv(
            validator_uv,
            t ,
            ndata,
            cnst,
            data,
            exc,
            size,
        ):
    '''
    Test the type constructors.
    '''
    validator_uv(t, cnst, ndata, ndata, ndata, data, exc,
                size=size,
        )


@mark.parametrize('cnst, t, vals', [
    (vector2, Vector2, (0.0, 0.0)),
    (vector3, Vector3, (0.0, 0.0, 0.0)),
    (vector4, Vector4, (0.0, 0.0, 0.0, 0.0)),
    (uv, UvfFloat, (0.0, 0.0)),
    (scale, Scale, (1.0, 1.0, 1.0)),
    (point, Point, (0.0, 0.0, 0.0)),
])
def test_emvty(cnst, t, vals):
    '''
    Test the empty type constructor.
    '''
    r = cnst()
    assert r == t(*vals)
    assert isinstance(r, t)


@mark.parametrize('cnst, min_data, max_data, ndata, t, size, to_int', [
    (color, 3, 4, 3, RGB, 4, False),
    (color, 3, 4, 4, RGBA, 4, False),
])
# Data and expected exceptions.
@mark.parametrize('data, exc', [
    ((1.0, 0.9, 0.3, 0.1), None),
    ((1, "foo", 1, 1,), ValueError),
    ((1, 1, 1, 1, 1), ValueError),
])
def test_color(
                validator_nested,
                t,
                min_data,
                max_data,
                ndata,
                cnst,
                data,
                exc,
                size,
                to_int,
            ):
    validator_nested(t, cnst, min_data, max_data, ndata, data, exc,
               size=size,
               to_int=to_int,
            )


@mark.parametrize('size', [1, 2, 4])
def test_empty_color(size):
    '''
    Test the empty color type constructor.
    '''
    match size:
        case 1:
            expected = RGB8(0, 0, 0)
        case 2:
            expected = RGB16(0, 0, 0)
        case 4:
            expected = RGB(0.0, 0.0, 0.0)
    r = color(size=size)
    assert r == expected

@mark.parametrize('cnst, min_data, max_data, ndata, t, size, to_int', [
    (rgb8, 3, 4, 3, RGB8, 1, True),
    (rgb8, 3, 4, 4, RGBA8, 1, True),
    (rgb16, 3, 4, 3, RGB16, 2, True),
    (rgb16, 3, 4, 4, RGBA16, 2, True),
])
# Data and expected exceptions.
@mark.parametrize('data, exc', [
    ((1.0, 0.9, 0.3, 0.1), None),
    ((1, "foo", 1, 1,), ValueError),
    ((1, 1, 1, 1, 1), ValueError),
])
def test_color_flat(
                validator_flat,
                t,
                min_data,
                max_data,
                ndata,
                cnst,
                data,
                exc,
                size,
                to_int,
            ):
    validator_flat(t, cnst, min_data, max_data, ndata, data, exc,
               size=size,
               to_int=to_int,
            )

@mark.parametrize('data', [
    (1, 2, 3, 1),
    (0.4, 0.3, 0.2, 1),
    (1, 2, 3, -1),
    (0.4, 0.3, 0.2, -1),
])
@mark.parametrize('tcase', [
    case_tuple,
    case_tuple_tuple,
    case_obj,
    case_obj_tuple,
    case_numpy,
])
def test_tangent(data, tcase):
    '''
    Test the tangent type constructor.
    '''
    expected = Tangent(*(float(n) for n in data))
    args = tcase(Tangent, tuple(float(d) for d in data))
    r = tangent(*args)
    assert tuple(r) == approx(tuple(expected))
    assert type(r) is Tangent
    assert all(isinstance(v, float) for v in r)

@mark.parametrize('size', [0, 1, 2])
@mark.parametrize('data', [
    (1,),
    (1, 2,),
    (1, 2, 3,),
    (1, 2, 3, 4,),
    (1, 2, 3, 4, 5,),
    (1, 2, 3, 4, 5, 6,),
    (1, 2, 3, 4, 5, 6, 7,),
])
@mark.parametrize('tcase', [
    case_tuple,
    case_tuple_tuple,
    case_obj,
    #case_obj_tuple,
    case_numpy8,
    case_numpy16,
])
def test_joint(tcase, data, size):
    '''
    Test the joint type constructor.
    '''
    jtype = [Joint, _Joint8, _Joint16][size]
    def extend(d):
        return d + (0,) * (4 - len(d))
    expected = [
        jtype(*extend(data[i*4:i*4+4]))
        for i in range((len(data)+3)//4)
    ]
    tdata = tcase(jtype, extend(data[:4]))
    r = joint(*tdata, size=size)
    for v, e in zip(r, expected):
        assert isinstance(v, jtype)
        assert tuple(v) == approx(tuple(e))

@mark.parametrize('data, size, expected', [
    ((0.4, 0.3, 0.2, 0.1), 4, ((0.4, 0.3, 0.2, 0.1),)),
    ((0.2, 0.2, 0.3, 0.2, 0.1), 4, ((0.2, 0.2, 0.3, 0.2), (0.1, 0.0, 0.0, 0.0))),
    ((0.1, 0.2, 0.3, 0.2, 0.1, 0.1,), 4, ((0.1, 0.2, 0.3, 0.2), (0.1, 0.1, 0.0, 0.0))),
    ((1, 2, 3, 4), 4, ((0.1, 0.2, 0.3, 0.4),)),
    ((0.2, 0, 0, 0), 4, ((1.0, 0, 0, 0),)),
    ((0.5, 0.25, 0.125, 0.125), 1, ((128, 64, 31, 32),)),
    ((2, 2, 4, 8), 1, ((31, 32, 64, 128),)),
    ((0.2, 0, 0, 0), 1, ((255, 0, 0, 0),)),
    ((0.5, 0.25, 0.125, 0.125), 2, ((32768, 16384, 8191, 8192),)),
    ((2, 2, 4, 8), 2, ((8191, 8192, 16384, 32768),)),
    ((0.2, 0, 0, 0), 2, ((65535, 0, 0, 0),)),
    ((0.2,), 2, ((65535, 0, 0, 0),)),
    ((0,), 1, ((0, 0, 0, 0),)),
    ((0,), 2, ((0, 0, 0, 0),)),
    ((0.0,), 4, ((0.0, 0.0, 0.0, 0.0),)),
    ((), 4, ValueError),
    ((), 1, ValueError),
    ((), 2, ValueError),
])
@mark.parametrize('tcase, zeropad', [
    (case_tuple, False),
    (case_tuple_tuple, False),
    (case_weight_tuple, True),
    (case_numpy, False)   
])
def test_weight(tcase,
                zeropad,
                data,
                size,
                expected):
    '''
    Test the weight type constructor.
    '''
    c, t, rnd, lim  = {
        4: (weight, _Weightf, float, 1.0),
        1: (weight8, _Weight8, round, 255),
        2: (weight16, _Weight16, round, 65535),
    }[size]
    zero = rnd(0)
    if isinstance(expected, tuple):
        expected = tuple(tuple(rnd(v) for v in e)
                         for e in expected)
    data = [rnd(d * lim) for d in data]
    kwargs, t_kwargs = {}, {}
    if signature(tcase).parameters.get('size') is not None:
        t_kwargs['size'] = size
    if signature(c).parameters.get('size') is not None:
        kwargs['size'] = size
    argdata = data
    if zeropad:
        # Zeropad means we pad with zeros and construct the expected data
        # from the input data, and ignore the declared expected data.
        if len(data) != 4:
            argdata = (*(
                float(d)
                for d in data
                ),
                zero, zero, zero, zero
            )[:4]
    if isinstance(expected, type) and issubclass(expected, Exception):
        arg = tcase(t, argdata, **t_kwargs)
        with raises(expected):
            c(arg, **kwargs)
        return
    arg =  tcase(t, argdata, **t_kwargs)
    if zeropad:
        expected = tuple(v
                        for e in expected
                        for v in tcase(t, e, **t_kwargs)
                        )
    else:
        expected = tuple(t(*[*e, zero, zero, zero][:4]) for e in expected)
    r = c(*arg, **kwargs)
    for v, e in zip(r, expected):
        assert type(v) is type(e)
        for vx, ex in zip(v, e):
            assert isinstance(vx, (type(ex), np.float32))
            if isinstance(vx, float):
                assert vx == approx(ex)
            else:
                assert ex-1 <= vx <= ex+1


@mark.parametrize('precision, weight', [
    (0, _Weightf),
    (1, _Weight8),
    (2, _Weight16),
    (4, _Weightf),
])

@mark.parametrize('size, joint', [
    (0, None),
    (1, _Joint8),
    (2, _Joint16),
])
@mark.parametrize('data, e_weights',[
    ({1: 0.3}, ((1.0, 0.0, 0.0, 0.0),)),
    ({1: 0.3}, (((1.0, 0.0, 0.0, 0.0),))),
    ({1: 0.3}, (((1.0, 0.0, 0.0, 0.0),))),
    ({1: 0.3}, (((255, 0.0, 0.0, 0.0),))),
    ({1: 0.3}, (((65535, 0.0, 0.0, 0.0),))),
    ({127: 0.3},  (((65535, 0.0, 0.0, 0.0),))),
    ({128: 0.3},  (((65535, 0.0, 0.0, 0.0),))),
    ({65535: 0.3},  (((65535, 0.0, 0.0, 0.0),))),
    ({65536: 0.3},  (((65536, 0.0, 0.0, 0.0),))),
])
def test_joints_weights(data, e_weights, size, joint, precision, weight):
    kwargs = {
        "size": size,
        "precision": precision,
    }
    e_joints = tuple(data.keys())
    # Handle out-of-range joint indexes.
    match size:
        case 1 if any(v > 255 for v in e_joints):
            with raises(ValueError):
                joints(data, **kwargs)
            return
        case 0|2 if any(v > 65535 for v in e_joints):
            with raises(ValueError):
                joints(data, **kwargs)
            return
    if joint is None:
        if all(v <= 255 for v in e_joints):
            joint = _Joint8
        else:
            joint = _Joint16
    def subst(v: int|float):
        match v, precision:
            case 0|0.0, 0|4:
                return 0.0
            case 0|0.0, 1|2:
                return 0
            case _, 1:
                return 255
            case _, 2:
                return 65535
            case _, 0|4:
                return 1.0
            case _, _:
                raise ValueError(f"Invalid test data {v=} {precision=}")
    pe_joints = tuple(joint(*(*v, 0, 0, 0, 0)[:4])
                      for i in range((len(e_joints) + 3) // 4)
                      for v in (e_joints[i*4:i*4+4],)
                      )
    pe_weights = tuple(weight(*(subst(v) for v in w)) for w in e_weights)
    r_joints, r_weights = joints(data, **kwargs)
    assert r_joints == pe_joints
    assert r_weights == pe_weights


@mark.parametrize('data, expected', [
    ((), (1.0, 1.0, 1.0)),
    ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    ((1.0, 2.0), ValueError),
    ((1.0,), (1.0, 1.0, 1.0)),
    ((1,), (1.0, 1.0, 1.0)),
    ((-1,), (-1.0, -1.0, -1.0)),
])
def test_scale(data, expected):
    '''
    Test the scale type constructor.
    '''
    if isinstance(expected, type) and issubclass(expected, Exception):
        with raises(expected):
            scale(*data)
        return
    expected = Scale(*(float(n) for n in expected))
    r = scale(*data)
    assert tuple(r) == approx(tuple(expected))
    assert type(r) is Scale
    assert all(isinstance(v, float) for v in r)


def via_vector(p1: PointLike, p2: PointLike):
    return (p1 - p2).length


@mark.parametrize('via', [
    via_vector,
    PointLike.distance,
])
@mark.parametrize('fn, n', [
    (point, 3),
    (uv, 2),
])
@mark.parametrize('p1, p2', [
    ((0, 0, 0), (0, 0, 0),),
    ((0, 0, 0), (1, 0, 0),),
    ((0, 0, 0), (0, 1, 0),),
    ((0, 0, 0), (0, 0, 1),),
    ((0, 0, 0), (0, 1, 1),),
    ((1, 1, 1), (2, 2, 2),),
    ((-1, -1, -1), (2, -2, -2),),
    ((-1.5, -1.5, -1.5), (2.5, -2.5, -2.5),),
    ((-1.5, -1.5, -1.5), (-2.5, -2.5, -2.5),),
    ((0, 0, 0), (1, 1, 1),),
])
def test_point_difference(fn, n, p1, p2, via):
    '''
    Test the distance function.
    '''
    p1 = p1[:n]
    p2 = p2[:n]
    p1 = fn(p1)
    p2 = fn(p2)
    # Use the constructed values to allow for scaling or clamping.
    d = sqrt(sum((a-b)*(a-b) for a,b in zip(p1, p2)))
    assert (p1 - p2).length == d
    assert (p2 - p1).length == d

def tanplus(x, y, z):
    return tangent(x, y, z, 1)
def tanminus(x, y, z):
    return tangent(x, y, z, -1)

@mark.parametrize('fn, dims', [
    (tanplus, 3),
    (tanminus, 3),
    (vector2, 2),
    (vector3, 3),
    (vector4, 4),
])
@mark.parametrize('v1, v2, r', [
    ((1, 1, 1, 1), (1, 1, 1, 1), (2, 3, 4)),
    ((0, 0, 1, 0), (0, 0, 1, 0), (0, 1, 1)),
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0)),
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0)),
    ((1, 0, 0, 0), (1, 1, 0, 0), (sqrt(2)*cos(pi/4), sqrt(2)*cos(pi/4), sqrt(2)*cos(pi/4))),
])
def test_dot(v1, v2, r, fn, dims):
    '''
    Test the dot product function.
    '''
    v1 = fn(*v1[:dims])
    v2 = fn(*v2[:dims])
    expected = r[dims-2]
    assert v1.dot(v2) == approx(expected)
    assert v2.dot(v1) == approx(expected)
    assert v1 * v2 == approx(expected)
    assert v2 * v1 == approx(expected)

@mark.parametrize('v1, v2, expect', [
    ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
    ((0, 0, 1), (0, 0, 1), (0, 0, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 1, 0), (0, 0, 1)),
])
def test_cross(v1, v2, expect):
    '''
    Test the cross product function.
    '''
    v1 = vector3(v1)
    v2 = vector3(v2)
    expect = vector3(expect)
    r = v1 @ v2
    rr = v2 @ v1
    assert tuple(r) == approx(tuple(expect))
    assert tuple(rr) == approx(tuple(-expect))

@mark.parametrize('fn, n, extra', [
    (vector2, 2, ()),
    (vector3, 3, ()),
    (vector4, 4, ()),
    (tangent, 3, (1,)),
    (tangent, 3, (-1,)),
])
@mark.parametrize('data, expect', [
    ((0, 0, 0, 0), False),
    ((1e-13, 1e-13, 1e-13, 1e-13), False),
    ((0.1, 0.1, 0.1, 0.1), True),
])
def test_bool(fn, n, extra, expect, data):
    '''
    Test the bool function.
    '''
    data = data[:n] + extra
    v = fn(data)
    assert bool(v) == expect
    assert bool(-v) == expect


@mark.parametrize('v1, v2, expected', [
    (vector2(1, 2), vector2(3, 4), vector2(4, 6)),
    (vector3(1, 2, 3), vector3(4, 5, 6), vector3(5, 7, 9)),
    (vector4(1, 2, 3, 4), vector4(5, 6, 7, 8), vector4(6, 8, 10, 12)),
    (tangent(1, 2, 3, 1), tangent(4, 5, 6, 1), tangent(5, 7, 9, 1)),
    (tangent(1, 2, 3, -1), tangent(4, 5, 6, -1), tangent(5, 7, 9, -1)),
])
def test_vector_add(v1, v2, expected):
    '''
    Test the vector addition function.
    '''
    r = v1 + v2
    rr = v2 + v1
    assert tuple(r) == approx(tuple(expected))
    assert tuple(rr) == approx(tuple(expected))
    
