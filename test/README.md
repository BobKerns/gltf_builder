# Testing gltf builder

There is a large and growing suite of test cases.  Generally, the test modules are parallel to the source files they test.

Documentation examples are tested in [test_examples.py](test_examples.py), and should be kept in sync.

Tests which produce output files do so in the `test/out` heirarchy, which is arranged like this:

```text
test/
  out/
    <module>/
      <test name>/
        <test name>.glb
        <test name>.gltf
        <test name>.json
```

This happens automatically when the `test_builder` fixture is used, and its `build()` method is called. The `build()` method takes additional arguments to pass to the official [gltf validator](https://github.com/KhronosGroup/glTF-Validator).

## Requirements

To run the tests, you need node.js and npm.

```bash
npm install
```

Will install the validator. The file [gltf-validator.js](gltf-validator.js) is a node script that loads and runs the validator from the command line.

The python function `validate_gltf()` in [conftest.py](conftest.py) invokes this script in a subprocess. The `test_builder` uses this to write the validation report to a JSON file.

The validation report is returned, and is saved in the returned `GLTF2` instance's extras field under:

```python
GLTF2.extras['gltf_builder']['validation']
```

This is only in the returned value, not the `.gltf` or `glb` files. It is there to potentially allow tests to introspect further in the report, such as the number of items saved.

It is also saved in `test_builder.report`.

## Debugging

With the recommended VSCode plugins, `.glb` files can be viewed directly within VSCode.

* [glTF Tools](https://marketplace.visualstudio.com/items/?itemName=cesium.gltf-vscode)
  * This provides 4 different libraries for previewing models
* [glTF Model Viewer](https://marketplace.visualstudio.com/items/?itemName=cloudedcat.vscode-model-viewer)
  * This uses google's model viewer

These provide inline validation of the `.gltf` JSON data, mouseover hover information, navigation by clicking on element IDs, and more.
