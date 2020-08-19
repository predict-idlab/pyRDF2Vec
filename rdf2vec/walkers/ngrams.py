import itertools
from hashlib import md5

import numpy as np

from rdf2vec.graph import Vertex
from rdf2vec.walkers import RandomWalker


class NGramWalker(RandomWalker):
    """Defines the N-Grams walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.
        n (int): The number of grams.
            Defaults to 3.
        wildcards (list): the wild cards.
            Defaults to None.

    """

    def __init__(self, depth, walks_per_graph, n=3, wildcards=None):
        super().__init__(depth, walks_per_graph)
        self.n = n
        self.wildcards = wildcards
        self.n_gram_map = {}

    def _take_n_grams(self, walk):
        """Takes the N-Grams.

        Args:
            walk (list): The walk.

        Returns:
            list: The N-Grams.

        """
        n_gram_walk = []
        for i, hop in enumerate(walk):
            if i == 0 or i % 2 == 1 or i < self.n:
                n_gram_walk.append(hop.name)
            else:
                n_gram = tuple(
                    walk[j].name for j in range(max(0, i - (self.n - 1)), i + 1)
                )
                if n_gram not in self.n_gram_map:
                    self.n_gram_map[n_gram] = str(len(self.n_gram_map))
                n_gram_walk.append(self.n_gram_map[n_gram])

        return n_gram_walk

    def extract(self, graph, instances):
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (array-like): The instances to extract the knowledge graph.

        Returns:
            list: The 2D matrix with its:
                number of rows equal to the number of provided instances;
                number of column equal to the embedding size.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walks.add(tuple(self._take_n_grams(walk)))

                # Introduce wild-cards and re-calculate n-grams
                if self.wildcards is None:
                    continue

                for wildcard in self.wildcards:
                    for idx in itertools.combinations(
                        range(1, len(walk)), wildcard
                    ):
                        new_walk = list(walk).copy()
                        for ix in idx:
                            new_walk[ix] = Vertex("*")
                        canonical_walks.add(tuple(self._take_n_grams(new_walk)))
        return canonical_walks
