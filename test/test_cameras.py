'''
Test cases for the cameras module.
'''

import math
from typing import cast
import pytest

import pygltflib as gltf

from gltf_builder import (
    camera, CameraType,
)

from conftest import TEST_EXTRAS, TEST_EXTENSIONS, BuilderContext, GeometryData
from gltf_builder.core_types import IndexSize
from gltf_builder.entities import BOrthographicCamera, BPerspectiveCamera
from gltf_builder.utils import count_iter, first, index_of


def test_camera_perspective(test_builder: BuilderContext):
    '''
    Test that a perspective camera is correctly built.
    '''
    cam = camera(CameraType.PERSPECTIVE, 'Camera',
                    yfov=math.pi / 6,
                    znear=0.01,
                    zfar=1000.0,
                    aspectRatio=1.0,
                    extras=TEST_EXTRAS,
                    extensions=TEST_EXTENSIONS,
                    )
    assert cam.name == 'Camera'
    assert cam._scope_name == 'camera'
    assert cam.aspectRatio == 1.0
    assert cam.yfov == math.pi / 6
    assert cam.znear == 0.01
    assert cam.zfar == 1000.0
    assert cam.extras == TEST_EXTRAS
    assert cam.extensions == TEST_EXTENSIONS
    assert cam.perspective is not None
    assert cam.orthographic is None
    assert cam.perspective.aspectRatio == 1.0
    assert cam.perspective.yfov == math.pi / 6
    assert cam.perspective.znear == 0.01
    assert cam.perspective.zfar == 1000.0
    assert cam.perspective.extras == {}
    assert cam.perspective.extensions == {}
    with test_builder() as tb:
        tb.add(cam)


def test_camera_orthographic(test_builder: BuilderContext):
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
    with test_builder() as tb:
        tb.add(cam)


@pytest.mark.parametrize('test_camera', [
    camera(CameraType.PERSPECTIVE, 'perspective'),
    camera(CameraType.ORTHOGRAPHIC, 'orthographic'),
])
def test_camera_node(cube: GeometryData,
                     index_sizes: tuple[IndexSize, int, int],
                     test_camera: BPerspectiveCamera | BOrthographicCamera,
                     test_builder: BuilderContext):
    '''
    Test that a camera node is correctly built.
    '''
    index_size, idx_bytes, idx_views = index_sizes
    with test_builder() as tb:
        tb.node('CameraNode',
                    translation=(0, 0, -10),
                    camera=test_camera,
                    )
        tb.instantiate(cube.nodes['TOP'],)
        g: gltf.GLTF2 = tb.build(index_size=index_size)
        assert count_iter(
            n
            for n in g.nodes
            if n.camera is not None
        ) == 1

        cameras: list[gltf.Camera] = g.cameras
        cam_idx = index_of(
            cast(list[gltf.Camera], cameras),
            value_eq=test_camera.name,
            attribute='name',
        )
        cam = cameras[cam_idx]
        assert cam.name == test_camera.name
        node = first(n for n in g.nodes if n.name == 'CameraNode')
        assert node.camera == cam_idx
        assert node.translation == [0, 0, -10]