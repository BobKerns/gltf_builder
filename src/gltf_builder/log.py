'''
Log setup for the glTF Builder package.
'''

from logging import getLogger, basicConfig, DEBUG

if False:
    basicConfig(level=DEBUG, format='%(message)s')
GLTF_LOG = getLogger('gltf_builder')

