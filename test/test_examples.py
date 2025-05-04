import math

from gltf_builder import (
    PrimitiveMode, Quaternion as Q,
    mesh, node,
)

def test_example1(test_builder):

    CUBE = (
        (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
        (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
    )
    CUBE_FACE1 = (0, 1, 2, 3)
    CUBE_FACE2 = (4, 5, 6, 7)
    CUBE_FACE3 = (0, 4, 5, 1)
    CUBE_FACE4 = (0, 4, 7, 3)
    CUBE_FACE5 = (1, 2, 6, 5)
    CUBE_FACE6 = (1, 5, 6, 2)

    msh = mesh('CUBE')
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
    msh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])
    with test_builder() as tb:
        top = tb.node('TOP')
        cube = node('CUBE',
                    mesh=msh,
                    translation=(-0.5, -0.5, -0.5),
        )
        # Instantiate it at the origin
        top.instantiate(cube)
        # Instantiate it translated, scaled, and rotated.
        top.instantiate(cube,
                        translation=(2, 0, 0),
                        scale=(1, 2, 2),
                        rotation=Q.from_axis_angle((1, 1, 0.5), math.pi/4)
                    )
