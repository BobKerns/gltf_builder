# GTLF Builder

This library wraps the `pygltflib` library to handle the low-level details of managing buffers, buffer views, and accessors.

## Usage

Install via your usual tool (I recommend `uv` as the modern upgrade from `pip` and others).

```python
from gtlf_builder import Builder

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

builder = Builder()

mesh = builder.add_mesh('CUBE')
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE1])
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE2])
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE3])
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE4])
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE5])
mesh.add_primitive(PrimitiveMode.LINE_LOOP, *[CUBE[i] for i in CUBE_FACE6])

node = builder.add_node(name='TOP', mesh=mesh)

gltf = builder.build()
gltf.save_binary('cube.glb')
```

The `builder.build()` method produces a regular `pygltflib.GLTF2` instance.

To create hierarchy, use the `add_node() method on a parent node.

Note that referencing the same tuple for a point treats it as the same vertex, while a copy will create a separate vertex.

## Still Needed

* Tangents and other attributes (Normals are almost supported.)
* Materials
* Skins
* Textures
* …
