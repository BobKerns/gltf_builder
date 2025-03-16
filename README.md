# GTLF Builder

This library wraps the `pygltflib` library to handle the low-level details of managing buffers, buffer views, and accessors.

In this document, we will generalliy refer to  the `pygltflib` library with the `gltf` orefix.

You start by creating a `Builder` instance. There are classes corresponding to the major classes from the `pygltflib` library, with names prepended with 'B'. For example, this library supplies a `BNode` class that plays the same role as `gltf.Node`. These classe are compiled to the corresponding `gltf` clases with the `compile(Builder)` method.

Compilation and collection of the pieces is performed by the `Builder.build()` method.

## Usage

Install via your usual tool (I recommend `uv` as the modern upgrade from `pip` and others).

```python
from gltf_builder import Builder, PrimitiveMode

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
gltf.save_binary('cube.glb')
```

The `builder.build()` method produces a regular `pygltflib.GLTF2` instance.

To create hierarchy, use the `add_node()` method on a parent node.

Note that referencing the same tuple for a point treats it as the same vertex, while a copy will create a separate vertex.

## Instancing

Simple instancing can be done by simply using the same mesh for multiple nodes.

You can also instance a node hierarchy with the `instantiate` method. This takes a node and copies it, optionally supplying a transformation.

The node can be an existing node in the scene, or it cn be created with the `detached=True` option to `add_node()`, which creates the a node that is not added to the scene. You can then use this as the root of an instancable tree, and add child nodes and meshes.

You can access existing nodes by name by the `builder[`_name_`]` syntax. If nodes with the same name appear in different places, you may need to first access a parent that holds only one of the duplicates. Alternatively, you can loop over all nodes like this:

```python
builder = Builder()
# Add a bunch of nodes
...
# Print the names of every node in the tree
for node in builder:
    print(f'node={node.name}')

# Get a list of all nodes named 'Fred'
fred = [n for n in builder if n.name == 'Fred']
```

## Still Needed

* Tangents and other attributes (implemented but no tested)
* Materials
* Skins
* Textures
* …
