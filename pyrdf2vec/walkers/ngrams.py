import itertools
from typing import Any, Dict, List, Optional, Tuple

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class NGramWalker(RandomWalker):
    """Defines the N-Grams walking strategy.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        grams: The number of grams.
            Defaults to 3.
        wildcards: the wild cards.
            Defaults to None.
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
        grams: int = 3,
        wildcards: list = None,
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(depth, max_walks, sampler, n_jobs, seed)
        self.grams = grams
        self.n_gram_map: Dict[Tuple, str] = {}
        self.wildcards = wildcards

    def _take_n_grams(self, walks: List[Vertex]) -> List[str]:
        """Takes the N-Grams.

        Args:
            walks: The walks.

        Returns:
            The N-Grams.

        """
        n_gram_walk = []
        for i, hop in enumerate(walks):
            if i == 0 or i % 2 == 1 or i < self.grams:
                n_gram_walk.append(str(hop))
            else:
                n_gram = tuple(
                    str(walks[j])
                    for j in range(max(0, i - (self.grams - 1)), i + 1)
                )
                if n_gram not in self.n_gram_map:
                    self.n_gram_map[n_gram] = str(len(self.n_gram_map))
                n_gram_walk.append(self.n_gram_map[n_gram])
        return n_gram_walk

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
        for walk in self.extract_walks(kg, str(instance)):
            canonical_walks.add(
                tuple(self._take_n_grams(walk))  # type:ignore
            )

            # Introduce wild-cards and re-calculate n-grams
            if self.wildcards is None:
                continue

            for wildcard in self.wildcards:
                for idx in itertools.combinations(
                    range(1, len(walk)), wildcard  # type: ignore
                ):
                    new_walk = list(walk).copy()  # type: ignore
                    for ix in idx:
                        new_walk[ix] = Vertex("*")
                    canonical_walks.add(tuple(self._take_n_grams(new_walk)))
        return {instance: tuple(canonical_walks)}
