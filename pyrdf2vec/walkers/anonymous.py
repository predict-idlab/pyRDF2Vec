from typing import Any, Dict, Optional, Tuple

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class AnonymousWalker(RandomWalker):
    """Defines the anonymous walking strategy.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        seed: The seed to use to ensure ensure random determinism to generate
            the same walks for entities.
            Defaults to None.

    """

    def __init__(
        self,
        depth: int,
        max_walks: Optional[int] = None,
        sampler: Sampler = UniformSampler(),
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(depth, max_walks, sampler, n_jobs, seed)

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
        for walk in self.extract_walks(kg, instance):
            canonical_walk = []
            str_walk = [str(x) for x in walk]  # type: ignore
            for i, hop in enumerate(walk):  # type: ignore
                if i == 0:
                    canonical_walk.append(str(hop))
                else:
                    canonical_walk.append(str(str_walk.index(str(hop))))
            canonical_walks.add(tuple(canonical_walk))
        return {instance: tuple(canonical_walks)}
