from typing import Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class AnonymousWalker(RandomWalker):
    """Walker that transforms label information into positional information in
    order to anonymize the random walks.

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
        with_reverse: True to extracts children's and parents' walks from the
            root, creating (max_walks * max_walks) more walks of 2 * depth,
            False otherwise.
            Defaults to False.

    """

    def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided entities which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, instance):
            canonical_walk = []
            str_walk = [hop.name for hop in walk]
            for i, hop in enumerate(walk):
                if i == 0:
                    canonical_walk.append(hop.name)
                else:
                    canonical_walk.append(str(str_walk.index(hop.name)))
            canonical_walks.add(tuple(canonical_walk))
        return {instance.name: tuple(canonical_walks)}
