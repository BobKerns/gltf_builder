# Change log

## Release 0.2.0

### Public changes in Release 0.2.0

- Introduced new functions for constructing attribute types (e.g., color, joins, weights).
- Restricted to a single buffer due to library limitations.
- Renamed several API methods for clarity and consistency.
- Enhanced `BNode.instantiate()` to accept both `BMesh` and `BNode`.
- Add functions for constructing attribute types, such as `color`, `joins`, `weights`, etc. Handles ranges and normalization

### Internal changes in Release 0.2.0

- Overhauled the compilation protocol for greater optimization control and efficiency.
  - Compilation now proceeds over a series of phases, that collect the objects used, enumerate them, calculate sizes, allocate space in the buffer(s), and finally produce the final structure.
- Disallow multiple buffers, as the underlying `pygltflib` library does not fully support them.
- Avoid repeatedly copying data, by writing data directly into the buffer once sizing and allocation is complete.

### Documentation in Release 0.2.0

- Various documentation improvements.

### Build/Test in Release 0.2.0

- Unit tests now output both `.glb` and `.gltf` files.
- Unit tests validate `.glb` files with the official validator.

## Release 0.1.8

- Clean up extras.
- Add predefined geometries (cube, so far)
- Fix some bugs that came to light in testing predefined geometries.

## Release 0.1.7

- How and when object names are used is now under control of a flag.
- Added a quaternion library with help from ChatGPT, for convenience and to avoid compatibility issues with other fine libraries.
