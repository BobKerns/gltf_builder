'''
Test cases for the cameras module.
'''

from gltf_builder.cameras import camera
from gltf_builder.core_types import CameraType


def test_camera_perspective(test_builder):
    '''
    Test that a perspective camera is correctly built.
    '''
    cam = camera(CameraType.PERSPECTIVE, 'Camera',
                    yfov=45.0,
                    znear=0.01,
                    zfar=1000.0,
                    aspectRatio=1.0,
                    extras={'test': 'extras'},
                    extensions={'test': 'extensions'},
                    )
    assert cam.name == 'Camera'
    assert cam._index == -1
    assert cam._scope_name == 'camera'
    assert cam.aspectRatio == 1.0
    assert cam.yfov == 45.0
    assert cam.znear == 0.01
    assert cam.zfar == 1000.0
    assert cam.extras == {'test': 'extras'}
    assert cam.extensions == {'test': 'extensions'}
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
                    extras={'test': 'extras'},
                    extensions={'test': 'extensions'},
                    )
    assert cam.name == 'Camera'
    assert cam._index == -1
    assert cam._scope_name == 'camera'
    assert cam.xmag == 1.0
    assert cam.ymag == 1.0
    assert cam.znear == 0.01
    assert cam.zfar == 1000.0
    assert cam.extras == {'test': 'extras'}
    assert cam.extensions == {'test': 'extensions'}
    assert cam.perspective is None
    assert cam.orthographic is not None
    assert cam.orthographic.xmag == 1.0
    assert cam.orthographic.ymag == 1.0
    assert cam.orthographic.znear == 0.01
    assert cam.orthographic.zfar == 1000.0
    assert cam.orthographic.extras == {}
    assert cam.orthographic.extensions == {}