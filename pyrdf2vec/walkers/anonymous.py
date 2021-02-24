from typing import Dict, List, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker


@attr.s
class AnonymousWalker(RandomWalker):
    """Walker that transforms label information into positional information in
    order to anonymize the random walks.

    Args:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        with_reverse: extracts children's and parents' walks from the root,
            creating (max_walks * max_walks) more walks of 2 * depth.
            Defaults to False.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    def _extract(
        self, kg: KG, instance: Vertex
    ) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks: Set[Tuple[str, ...]] = set()
        for walk in self.extract_walks(kg, instance):
            canonical_walk: List[str] = []
            str_walk = [hop.name for hop in walk]
            for i, hop in enumerate(walk):
                if i == 0:
                    canonical_walk.append(hop.name)
                else:
                    canonical_walk.append(str(str_walk.index(hop.name)))
            canonical_walks.add(tuple(canonical_walk))
        return {instance.name: tuple(canonical_walks)}
