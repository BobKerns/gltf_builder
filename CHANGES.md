# Change log.

## Release 0.2.0

* Major changes to underlying compilation protocol to enable more control over optimizations. Compilation now proceeds over a series of phases, that collect the objects used, enumerate them, calculate sizes, allocate space in the buffer(s), and finally produce the final structure.
* Disallow multiple buffers, as the underlying `pygltflib` library does not fully support them.
* Avoid repeatedly copying data, by writing data directly into the buffer once sizing and allocation is complete.
* Various renamings, for clarity and for public/private distinction.
  * `add_node()` becomes `create_node()`.
  * `add_mesh()` becomes `create_mesh()`.
* Supplying `detached=True` to `create_xxx()` methods creates detached opbjects without adding, for later use.
  * The `detached` attribute will be set for detached objects.
* The `BNode.instantiate()` will accept a `BMesh` as well as a `BNode`.
* The unit tests write out both `.glb` and `.gltf` files for examination.
* Minor documentation improvements.


## Release 0.1.8

* Clean up extras.
* Add predefined geometries (cube, so far)
* Fix some bugs that came to light in testing predefined geometries.

## Release 0.1.7

* How and when object names are used is now under control of a flag.
* Added a quaternion library with help from ChatGPT, for convenience and to avoid compatibility issues with other fine libraries.
