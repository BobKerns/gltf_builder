# glLF Builder

[![glTF Logo](docs/img/glTF_100px_June16.png)](https://www.khronos.org/gltf/)
[![WebGL Logo](docs/img/WebGL_100px_June16.png)](https://www.khronos.org/webgl/)

This library wraps the [`pygltflib`](https://github.com/lukas-shawford/gltflib) library to handle the low-level details of managing buffers, buffer views, and accessors.

In this document, we will generally refer to  the `pygltflib` library with the `gltf` prefix.

For more information about the underlying glTF model, see [our page on glTF](docs/glTF.md).

## Project Status

As of 2025-04-26: Under active development, with extensive tests. All files produced by the tests are validated with the [official validator](https://github.khronos.org/glTF-Validator/).

Working:

- Nodes, meshes, sufficient for wireframes
- Attributes (color is the only useful attribute in wireframe)

Incomplete:

- Images
- Textures
- Materials
- Vertex management
- Skins
- Documentation

Not started:

- Animation

## Usage

You start by creating a `Builder` instance. There are abstract types corresponding to the major classes from the `pygltflib` library, with names prepended with 'B'. For example, this library supplies a `BNode` class that plays the same role as `gltf.Node`. These classes are compiled to the corresponding `gltf` classes with the `compile()` method, which returns a `pygltflib.GLTF2` instance.

The `BXxxxx` names are abstract; the implementation classes bear names like `_Xxxxx`.

Compilation and collection of the pieces is performed by the `Builder.build()` method.

Install via your usual tool (I recommend [`uv`](https://docs.astral.sh/uv/) as the modern upgrade from `pip` and others).

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
top = builder.node('TOP')
cube = node('CUBE',
    mesh=mesh,
    translation=(-0.5, -0.5, -0.5),
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

Notes:

- The `builder.build()` method produces a regular `pygltflib.GLTF2` instance.
- To create hierarchy, use the `add_node()` method on a parent node.
- Using `Builder.add_mesh()` rather than `mesh` associates it with the builder immediately
  - This allows you to retrieve it with `builder.meshes[`_name_`]`.
  - `mesh()` produces a detached mesh, which can used for one or more nodes.
- Using `Builder.node()` rather than `node()` associates it with the builder immediately.
  - This allows you to retrieve it with `builder.nodes[`_name_`]`.
  - `node()` produces a detached node, which can instantiated or added later.
- Objects do not need to be added to the builder explicitly if they are referenced by other objects.
- Using `Builder.add_mesh()` or `Builder.node()` adds the resulting mesh to the file, even if not referenced. They are otherwise equivalent to `mesh()` or `node()`.

## Instancing

Simple instancing can be done by simply using the same mesh for multiple nodes.

You can also instance a node hierarchy with the `instantiate` method. This takes a node and copies it, optionally supplying a transformation.

The node can be an existing node in the scene, or it cn be created with `node()`, which creates the a node that is not added to the scene. You can then use this as the root of an instancable tree, and add child nodes and meshes.

You can access existing nodes by name by the `builder.nodes[`_name_`]` syntax. If nodes with the same name appear in different places, you may need to first access a parent that holds only one of the duplicates. Alternatively, you can loop over all nodes like this:

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

## Naming

`name_policy` is a configuration option on the `Builder` that determines how names are applied to elements within the GLTF builder. For each naming scope, it can be set to different modes to control the naming behavior:

- `explicit`: Names are only applied to elements that you explicitly name.
- `auto`: Names are automatically generated and applied to all elements.
- `mixed`: A combination of explicit and automatic naming is used.

Adjusting the `name_policy` allows for greater flexibility and control over the naming conventions used in your GLTF files.

## Matrices, Vector types, colors, etc

This includes types and functions for creating vectors, colors, etc. These are useful for providing attribute values, but if you provide the appropriate `tuple` or `np.ndarray` of values, they will be converted with the appropriate constructor function.

Using the provided functions gives error and range checking, and may inform the library of the intended data type to use, and allows you to use operations like matrix or vector operations.

### Constructor Functions

The main user objects have functions to construct them, rather than using the classes directly.

The functions are declared to return abstract types; the concrete implementation classes are hidden. The public interface is through these abstract types.

| Constructor       | Type                | Description                                     |
|-------------------|---------------------|-------------------------------------------------|
| `point`           | `Point`             ||
| `vector2`         | `Vector2`           ||
| `vector3`         | `Vector3`           ||
| `vector4`         | `Vector4`           ||
| `uv`              | `UvPoint`           | (comes in 8-bit, 16bit, and floating versions)  |
| `tangent`         | `Tangent`           ||
| `scale`           | `Scale`             ||
| `quaternion`      | `Quaternion`        ||
| `matrix2`         | `Matrix2`           ||
| `matrix3`         | `Matrix3`           ||
| `matrix4`         | `Matrix4`           ||
| `color`           | `Color`             | (comes in 8-bit, 16-bit, and floating versions below) |
| `rgb`, `rgba`     | `RGB` and `RGBA`    | Floating point |
| `rgb8`, `rgba8`   | `RGB8` and `RGBA8`  | 8-Bit |
| `rgb16`, `rgba16` | `RGB16`and `RGBA16` | 16 Bit |

These take the expected values, with the following notes:

- All functions, if provided an instance of their constructed type, will return it unchanged.
- `color` takes values between 0..1 inclusive. The `size=` keyword argument specifies the data format used, 1, 2, or 4 bytes. 4 bytes uses `np.float32` format.
- `rgb` and `rgba` takes values between 0..1 inclusive, like color, and always uses the `np.float32` format.
- `rgb8`, `rgba8` use integer values between 0..255 inclusive and the 1-byte format.
- `rgb16` and `rgba16` use integer values between 0..65535 inclusive and the 2-byte format.
- `matrix2`, `matrix3`, and `matrix4` will accept tuples of 4, 9, or 16 values, or tuples of tuples in 2x2, 3x3, or 4x4 form.
- `scale` will accept 3 values, a tuple or `np.ndarray` of 3 values, or a single value to be applied to all three dimensions.
- The `tangent` function constructs `Tangent` values for use as the `TANGENT` vertex attribute. As such, it takes the _X_, _Y_, and _Z_ values like a `Vector3` and a fourth value, either -1 or 1, to indicate its orientation. The `Tangent` value can be treated like a `Vector3`, ie. it supports cross product via the `@` operator.
- `uv` returns texture coordinates. The `size` argument specifies 1, 2, or 4-byte formats, with the 4-byte format being `np.float32`
- Points (including uv texture coordinates) and vectors are not the same, so have different types:
  - They behave differently under transformation.
  - Vectors add, points do not.
    - But you can subtract points to get a vector.
    - You can add a point and a vector to get a new point.

## Graphic Elements: Nodes, Meshes, Primitives,and more

| Constructor               | Type          | Description           |
|---------------------------|---------------|-----------------------|
| [node](#node)            | `BNode`       | Hierarchical structure |
| [mesh](#mesh)           | `BMesh`       | Holds a list of primitives |
| [primitive](#primitive) | `BPrimitive`  | One drawing operation |
| [camera](#camera)         | `BCamera`     | A camera description  |
|                           | `BPerspectiveCamera`  ||
|                           | `BOrthographicCamera` ||

See [Geometry model](docs/glTF.md#geometry-model)

## Node

Nodes represent hierarchical geometry.

## Mesh

...

## Primitive

...

## Camera

...

## Documentation

Documentation is an ongoing effort, and presently disorganized, but improvement and examples are a priority once key functionality is working.

Read the [documentation](docs/README.md), and file a [documentation issue](https://github.com/BobKerns/gltf_builder/issues/new?template=documentation.md).

## Development

There are several pages relating to [development](DEVELOPMENT.md), [release](RELEASE.md), and [internals](docs/README.md#internals)

### Still Needed

- Animation
- Material support is not yet integrated
- Scoping of indexing and vertex management.
- Lots more documentation and examples.
