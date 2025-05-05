# Documentation Overview

[![glTF Logo](img/glTF_100px_June16.png)](https://www.khronos.org/gltf/)
[![WebGL Logo](img/WebGL_100px_June16.png)](https://www.khronos.org/webgl/)

- [Quaternions](quaternions.md)
- [glTF Model](glTF.md)
- [Compiler](compiler.md)

## Quaternions (Rotations)

Because `pygltflib` uses quaternions of the form (X, Y, Z, W) instead of the form (W, X, Y, Z) used by `scipy`, and to avoid introducing heavyweight and potentially incompatible libraries, we provide (courtesy of ChatGPT to my specifications) an implementation of various quaternion routines relating to rotations.

The alternative would be to introduce another dependency and a wrapper.

Basic usage

```python
import gltf_builder.quaternion as Q

# Rotate around Z axis by pi/4
rotation = Q.from_axis_angle((0, 0, 1), math.pi / 4)
# Instantiate a geometry, rotated.
root_node.instantiate(cube, rotation=rotation)
```

See [quaternions.md](quaternions.md) or [quaternions.py](src/gltf_builder/quaternions.py) for more information.

## Internals

- [glTF Model](glTF.md)
- [Compiler](compiler.md)
