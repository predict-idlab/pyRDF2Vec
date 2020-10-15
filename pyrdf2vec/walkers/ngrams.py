import itertools
from typing import Any, Dict, List, Set, Tuple

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class NGramWalker(RandomWalker):
    """Defines the N-Grams walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().
        grams: The number of grams.
            Defaults to 3.
        wildcards: the wild cards.
            Defaults to None.

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = UniformSampler(),
        grams: int = 3,
        wildcards: list = None,
    ):
        super().__init__(depth, walks_per_graph, sampler)
        self.grams = grams
        self.n_gram_map = {}  # type: Dict[Tuple, str]
        self.wildcards = wildcards

    def _take_n_grams(self, walks: List[Vertex]) -> List[Dict[Tuple, str]]:
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
        return n_gram_walk  # type:ignore

    def _extract(
        self, graph: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Dict[Tuple[Any, ...], str], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, str(instance))
            for walk in walks:
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
                        canonical_walks.add(
                            tuple(self._take_n_grams(new_walk))
                        )
        return canonical_walks
