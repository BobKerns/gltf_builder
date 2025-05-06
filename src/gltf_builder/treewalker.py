'''
Walk the tree of nodes in various ways.
This module provides functions to traverse the tree of nodes in a glTF
file, allowing for different traversal strategies such as which nodes
to visit, the order of traversal, and how the results are aggregated.

This module is used by the compiler to traverse the tree of nodes in a glTF
file and collect the data needed to build the final glTF file.
'''

from abc import abstractmethod
from typing import Generic, Protocol, TypeVar, Optional, TYPE_CHECKING
from collections.abc import Generator, Iterable


from gltf_builder.core_types import (
    Phase,
    EntityType,
)
from gltf_builder.global_state import GlobalState


if TYPE_CHECKING:
    from gltf_builder.entities import Entity
    from gltf_builder.builder import Builder
    from gltf_builder.compiler import _DoCompileReturn


_T_Return = TypeVar('_T_Return')
'''
Return type of the `TreeWalker` class. This determines the return type
of the `walk` method, and the outermost call to the `aggregate` function.
'''



class VisitFunc(Protocol):
    '''
    Function to call for each node visited.

    Parameters:
    -----------
    node: Entity
        The node being visited.
    entity_type: EntityType
        The type to which the node is being visited.
    phase: Phase
        The phase of the compilation process.

    Returns:
    -------
    Varies
    '''

    @abstractmethod
    def __call__(self,
                globl: 'GlobalState',
                node: 'Entity',
                to_type: EntityType,
                phase: Phase,
            /):
        ...


class OrderFunc(Protocol):
    '''
    Function to determine the order of traversal.

    Parameters:
    -----------
    node: Entity
        The node being visited.
    to_type: EntityType
        The type to which the node is being visited.
    phase: Phase
        The phase of the compilation process.
    Returns:
    -------
    Generator[Entity, None, None]
        An iteration over the nodes to be processed.
    '''

    def __call__(self,
                globl: 'GlobalState',
                node: 'Entity',
                to_type: EntityType,
                phase: Phase,
                /) -> Generator['Entity', None, None]:
        '''
        Return the order of traversal for the node.
        '''
        if False:
            yield


class AggregateFunc(Protocol):
    '''
    Function to aggregate the results of the traversal.
    '''

    def __call__(self,
                globl: 'GlobalState',
                entity: 'Entity',
                to_type: EntityType,
                phase: Phase,
                this_value: _DoCompileReturn,
                values: Iterable[_DoCompileReturn],
                /) -> None:
        '''
        Aggregate the results of the traversal for the node.
        '''
        raise NotImplementedError('Aggregate function not implemented.')


class TreeWalker(Generic[_T_Return]):
    '''
    Class to walk the tree of nodes in a glTF file.

    This class is used by the compiler to traverse the tree of nodes in a glTF
    file and collect the data needed to build the final glTF file.

    A `TreeWalker` encapsulates a traversal policy, and does not itself hold any state.

    The traversal policy is defined by the `visit`, `order`, and `aggregate` functions.
    The `visit` function is called for each node visited, the `order` function
    determines the nodes visited and their order of traversal, and the `aggregate`
    function aggregates the results of the traversal.

    The walk method is called to start the traversal of the tree of nodes.

    The `phase` argument indicates what phase of the compilation process is being
    performed. It is possible to check this for phase-specific behavior, but a
    separate `TreeWalker` should be created for each phase even if they share code,
    as the `phase` is associated with the `TreeWalker` instance.

    A `TreeWalker` returns a `GlobalState` object that contains the results of the traversal.
    The `GlobalState` object may then be passed to another treewalker, to further collect or
    modify the state of the glTF file.
    '''

    visit: VisitFunc
    '''
    Function to call for each node visited.
    '''
    order: OrderFunc
    '''
    Function to determine the order of traversal.
    '''
    aggregate: AggregateFunc
    '''
    Function to aggregate the results of the traversal.
    '''
    phase: Phase
    '''
    Phase of the compilation process.
    '''

    def __init__(self, /, *,
                 visit: VisitFunc,
                 order: OrderFunc,
                 aggregate: AggregateFunc,
                 phase: Phase,
                 ) -> None:
        '''A short docstring'''
        self.visit = visit
        self.order = order
        self.aggregate = aggregate
        self.phase = phase

    def walk_entity(self,
                    globl: 'GlobalState',
                    entity: 'Entity',\
                    /) -> _DoCompileReturn:
        '''
        Walk the node and return the result of the traversal.
        '''
        phase, to_type = globl.phase, EntityType.BUILDER
        assert phase is not None
        state = globl.state(entity)
        this_value = self.visit(
            globl,
            entity,
            to_type,
            phase)
        order = self.order(
            globl,
            entity,
            to_type,
            phase)
        values = (
            self.walk_entity(globl, n)
            for n in order
        )
        if self.aggregate:
            result = self.aggregate(
                globl,
                entity,
                to_type,
                phase,
                this_value,
                values)
        else:
            result = this_value
        setattr(state, phase.name, result)
        return result


    def walk(self, builder: 'Builder', globl: Optional[GlobalState]=None) -> GlobalState:
        '''
        Walk the tree of nodes in the glTF file.
        '''
        globl = GlobalState(builder)
        globl.treewalker = self
        globl.phase = self.phase
        # The order function should just return the root node itself.
        # ordered = self.order(globl, globl, ScopeName.BUILDER, self.phase)
        # But globl stands in for the builder root.
        this_value = self.walk_entity(globl,globl.entity)
        if self.aggregate:
            ret = self.aggregate(
                globl,
                globl.entity,
                EntityType.BUILDER,
                self.phase,
                this_value,
                (),
            )
        else:
            ret = this_value
        globl.returns[self.phase] = ret
        return globl

