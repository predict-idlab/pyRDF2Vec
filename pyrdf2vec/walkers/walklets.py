from typing import Any, Dict, Optional, Tuple

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class WalkletWalker(RandomWalker):
    """Defines the walklet walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: Optional[int] = None,
        sampler: Sampler = UniformSampler(),
        n_jobs: int = 1,
    ):
        super().__init__(depth, walks_per_graph, sampler, n_jobs)

    def _extract(
        self, kg: KG, instance: rdflib.URIRef
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
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
        canonical_walks = set()
        walks = self.extract_random_walks(kg, str(instance))
        for walk in walks:
            if len(walk) == 1:  # type:ignore
                canonical_walks.add((str(walk[0]),))  # type:ignore
            for n in range(1, len(walk)):  # type:ignore
                canonical_walks.add(
                    (str(walk[0]), str(walk[n]))  # type: ignore
                )
        return {instance: tuple(canonical_walks)}
