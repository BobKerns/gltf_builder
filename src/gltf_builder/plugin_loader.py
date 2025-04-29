'''
Plugin loader for glTF Builder.

This handles plugin discovery and loading for glTF Builder.
How and when they are loaded, activated, and used is up to the
caller.
'''

from collections import defaultdict
from contextlib import suppress
from importlib.metadata import PackageMetadata, PackageNotFoundError, entry_points, metadata
import re
import sys
from typing import Callable, Optional, cast
from warnings import warn

from semver import Version

from gltf_builder.accessors import std_repr

_PluginGroup = dict[str, 'Plugin']
_PLUGINS: dict[str, _PluginGroup] = defaultdict(dict)

class Plugin:
    '''
    Base class for all plugins.
    '''
    name: str
    version: str = '0.0.0'
    semver: Optional[Version] = None
    author: str = ''
    summary: str = ''
    requires_python: Optional[Version] = None

    def __init__(self, name: str, /, *,
                author: str='',
                version: str='0.0.0',
                semver: Optional[Version]=None,
                summary: str='',
                requires_python: Optional[Version]=None,
            ) -> None:
        self.name = name
        self.author = author
        self.version = version
        self.semver = semver
        self.summary = summary
        self.requires_python = requires_python

    def __repr__(self) -> str:
        return std_repr(self, (
            'name',
            'version',
            'author',
            'summary',
        ))

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plugin):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

RE_EMAIL =  re.compile(
                       r'^\s*(\S[^\<]*)\s*\<.*$'
                       + r'|\s*[<]([^@]*)@[^>]+[>].*$'
                       + r'|\s*([^@]+)@.*$')
def human_email(email: str) -> str:
    '''
    Extract the most human portion of an email address.
    '''
    match = RE_EMAIL.match(email)
    if match is None:
        return email
    return (
        match.group(1) or
        match.group(3) or
        match.group(2) or
        email
    ).strip()


def load_plugins(group: str, /, *,
                 validator: Optional[Callable[[Plugin], bool]]=None,
                 ) -> list[Plugin]:
    '''
    Load the plugins from their metadata.

    Parameters
    ----------
    group : str
        The group of the plugin.
    validator : Optional[Callable[[Plugin], bool]]
        A function that validates the plugin.
        If the plugin is not valid, it will be skipped.
        The function should return True if the plugin is valid,
        False otherwise.
        If None, all plugins will be assumed valid.

    Returns
    -------
    list[Plugin]
        The list of plugins.

    Raises
    ------
    ValueError
        If the group is not found.
    TypeError
        If the validator is not callable.
    '''
    def find_package(cls: type) -> PackageMetadata|None:
        '''
        Find the package that contains the extension plugin.
        This is used to access the plugin metadata.
        '''
        name = cls.__module__
        if not name:
            return None
        sep = '.'
        while sep:
            with suppress(PackageNotFoundError):
                return metadata(name)
            name, sep, _ = name.rpartition('.')

    group_plugins = _PLUGINS[group]

    for entry_point in entry_points(group='gltf_builder.extensions'):
        plugin_class = cast(type[Plugin], entry_point.load())
        pkg = find_package(plugin_class)
        if pkg is None:
            warn(f'Plugin {entry_point.name} has no metadata')
            continue
        if not issubclass(plugin_class, Plugin):
            warn(f'Plugin {entry_point.name} is not a subclass of Plugin')
            continue
        try:
            author = pkg.get('Author-email', '') # type: ignore
            version = pkg.get('Version', '') # type: ignore
            semver: Version|None = None
            with suppress(ValueError):
                semver = Version.parse(version,
                                        optional_minor_and_patch=True)
            doc = plugin_class.__doc__ or ''
            summary= (doc.strip()
                      .split('\n', 1)[0]
                      .strip()
            )
            requires_python: Version|None = None
            python = pkg.get('Requires-Python', '').strip() # type: ignore
            python = python.split(' ', 1)[0]
            with suppress(ValueError):
                if python.startswith('>='):
                    python = python[2:].strip()
                requires_python = Version.parse(python,
                                                optional_minor_and_patch=True)
                if requires_python > sys.version_info[:3]:
                    warn(f'Plugin {entry_point.name} requires Python {requires_python}')
                    continue

            plugin = plugin_class(entry_point.name,
                                version=version,
                                semver=semver,
                                author=human_email(author),
                                summary=summary,
                                requires_python=requires_python,
                            )
            if validator and not validator(plugin):
                warn(f'Plugin {entry_point.name} is not valid')
                continue
            group_plugins[entry_point.name] = plugin
        except Exception as e:
            warn(f'Plugin {entry_point.name} failed to load: {e}')
            continue
    return list(group_plugins.values())

def get_plugin(group: str, name: str) -> Plugin|None:
    '''
    Get a plugin by name.

    Parameters
    ----------
    group : str
        The group of the plugin.
    name : str
        The name of the plugin.
    Returns
    -------
    Plugin|None
        The plugin if found, None otherwise.
    '''
    if group not in _PLUGINS:
        load_plugins(group)
    return _PLUGINS[group].get(name)
