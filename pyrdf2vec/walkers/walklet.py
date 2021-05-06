from typing import Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WalkletWalker(RandomWalker):
    """Walklets walking strategy which transforms randomly extracted walks into
    walklets which are walks of size one or two including the root node and
    potentially another vertex that can be a predicate or an object.

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
            entity, creating (max_walks * max_walks) walks of 2 * depth,
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
            if len(walk) == 1:
                canonical_walks.add((walk[0].name,))
            for i in range(1, len(walk)):
                if self.with_reverse:
                    canonical_walks.add((walk[i].name, entity.name))
                else:
                    canonical_walks.add((entity.name, walk[i].name))
        return {entity.name: list(canonical_walks)}
