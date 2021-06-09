from typing import Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, List, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class AnonymousWalker(RandomWalker):
    """Anonymous walking strategy which transforms each vertex name other than
    the root node, into positional information, in order to anonymize the
    randomly extracted walks.

    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        random_state: The random state to use to keep random determinism with
            the walking strategy.
            Defaults to None.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        with_reverse: True to extracts parents and children hops from an
            entity, creating (max_walks * max_walks) more walks of 2 * depth,
            allowing also to centralize this entity in the walks. False
            otherwise.
            Defaults to False.

    """

    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts random walks for an entity based on a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            A dictionary having the entity as key and a list of tuples as value
            corresponding to the extracted walks.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, entity):
            vertex_names = [vertex.name for vertex in walk]
            canonical_walk: List[str] = [
                vertex.name
                if vertex.name == entity.name
                else str(vertex_names.index(vertex.name))
                for vertex in walk
            ]
            canonical_walks.add(tuple(canonical_walk))
        return {entity.name: list(canonical_walks)}
