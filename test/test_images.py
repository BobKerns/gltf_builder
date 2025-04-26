'''
Tests for the images module.
'''

import numpy as np

from gltf_builder.core_types import ImageType
from gltf_builder.images import image

def test_image_uri():
    '''
    Test that an image can be created.
    '''
    img = image('test_image', uri='test_image.png')
    assert img.name == 'test_image'
    assert img.uri == 'test_image.png'
    assert img.blob is None
    assert img.imageType is ImageType.PNG
    assert img.mimeType == 'image/png'
    assert img.extras == {}
    assert img.extensions == {}


def test_image_blob():
    '''
    Test that an image can be created.
    '''
    blob = b'\x00\x01\x02\x03'
    img = image('test_image', imageType=ImageType.JPEG, blob=blob)
    assert img.name == 'test_image'
    assert img.blob == blob
    assert img.uri is None
    assert img.imageType == ImageType.JPEG
    assert img.mimeType == 'image/jpeg'
    assert img.extras == {}
    assert img.extensions == {}


def test_image_ndarray():
    '''
    Test that an image can be created.
    '''
    blob = b'\x00\x01\x02\x03'
    array = np.frombuffer(blob, dtype=np.uint8)
    img = image('test_image', imageType=ImageType.JPEG, blob=array)
    assert img.name == 'test_image'
    assert img.blob == blob
    assert img.uri is None
    assert img.imageType == ImageType.JPEG
    assert img.mimeType == 'image/jpeg'
    assert img.extras == {}
    assert img.extensions == {}