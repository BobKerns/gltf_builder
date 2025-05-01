'''
Test cases for the cameras module.
'''

import pytest

from gltf_builder import (
    camera, CameraType,
)

from conftest import TEST_EXTRAS, TEST_EXTENSIONS


def test_camera_perspective(test_builder):
    '''
    Test that a perspective camera is correctly built.
    '''
    cam = camera(CameraType.PERSPECTIVE, 'Camera',
                    yfov=45.0,
                    znear=0.01,
                    zfar=1000.0,
                    aspectRatio=1.0,
                    extras=TEST_EXTRAS,
                    extensions=TEST_EXTENSIONS,
                    )
    assert cam.name == 'Camera'
    assert cam._scope_name == 'camera'
    assert cam.aspectRatio == 1.0
    assert cam.yfov == 45.0
    assert cam.znear == 0.01
    assert cam.zfar == 1000.0
    assert cam.extras == TEST_EXTRAS
    assert cam.extensions == TEST_EXTENSIONS
    assert cam.perspective is not None
    assert cam.orthographic is None
    assert cam.perspective.aspectRatio == 1.0
    assert cam.perspective.yfov == 45.0
    assert cam.perspective.znear == 0.01
    assert cam.perspective.zfar == 1000.0
    assert cam.perspective.extras == {}
    assert cam.perspective.extensions == {}

def test_camera_orthographic():
    '''
    Test that an orthographic camera is correctly built.
    '''
    cam = camera(CameraType.ORTHOGRAPHIC, 'Camera',
                    xmag=1.0,
                    ymag=1.0,
                    znear=0.01,
                    zfar=1000.0,
                    extras=TEST_EXTRAS,
                    extensions=TEST_EXTENSIONS,
                    )
    assert cam.name == 'Camera'
    assert cam._scope_name == 'camera'
    assert cam.xmag == 1.0
    assert cam.ymag == 1.0
    assert cam.znear == 0.01
    assert cam.zfar == 1000.0
    assert cam.extras == TEST_EXTRAS
    assert cam.extensions == TEST_EXTENSIONS
    assert cam.perspective is None
    assert cam.orthographic is not None
    assert cam.orthographic.xmag == 1.0
    assert cam.orthographic.ymag == 1.0
    assert cam.orthographic.znear == 0.01
    assert cam.orthographic.zfar == 1000.0
    assert cam.orthographic.extras == {}
    assert cam.orthographic.extensions == {}


@pytest.mark.parametrize('test_camera', [
    camera(CameraType.PERSPECTIVE),
    camera(CameraType.ORTHOGRAPHIC),
])
def test_camera_node(cube, index_sizes, test_camera, test_builder):
    '''
    Test that a camera node is correctly built.
    '''
    index_size, idx_bytes, idx_views = index_sizes
    tb = test_builder
    tb.create_node('CameraNode',
                   translation=(0, 0, -10),
                   camera=test_camera)
    tb.instantiate(cube.nodes['TOP'],)
    g = tb.build(index_size=index_size)
    assert len(tuple(
        n
        for n in g.nodes
        if n.camera is not None
    )) == 1
