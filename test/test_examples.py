from gltf_builder import Builder, PrimitiveMode

def test_example1(outdir):
    builder = Builder()

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

    mesh = builder.add_mesh('CUBE')
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
    mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])
    top = builder.add_node(name='TOP')
    cube = builder.add_node(name='CUBE',
                            mesh=mesh,
                            translation=(-0.5, -0.5, -0.5),
                            detached=True, # Don't make it part of the scene
    )
    # Instantiate it at the origin
    top.instantiate(cube)
    # Instantiate it translated, scaled, and rotated.
    top.instantiate(cube,
                    translation=(2, 0, 0),
                    scale=(1, 2, 2),
                    rotation=(0.47415988, -0.40342268,  0.73846026,  0.25903472)
                )
    gltf = builder.build()
    gltf.save_binary(outdir / 'example1.glb')