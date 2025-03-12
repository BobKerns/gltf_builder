from gltf_builder import Builder

def test_empty_builder():
    b = Builder()
    g = b.build()
    blob = g.binary_blob()
    assert len(blob) == 0
