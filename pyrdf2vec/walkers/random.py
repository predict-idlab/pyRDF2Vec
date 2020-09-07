from hashlib import md5
from typing import Any, Set, Tuple

import numpy as np

from pyrdf2vec.graph import KnowledgeGraph, Vertex
from pyrdf2vec.walkers import Walker


class RandomWalker(Walker):
    """Defines the random walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth, walks_per_graph):
        super().__init__(depth, walks_per_graph)

    def extract_random_walks(
        self, graph: KnowledgeGraph, root: Vertex
    ) -> list:
        """Extracts random walks of depth - 1 hops rooted in root.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root.

        Returns:
            The array of the walks.

        """
        # Initialize one walk of length 1 (the root)
        walks = {(root,)}

        for i in range(self.depth):
            # In each iteration, iterate over the walks, grab the
            # last hop, get all its neighbors and extend the walks
            walks_copy = walks.copy()
            for walk in walks_copy:
                node = walk[-1]
                neighbors = graph.get_neighbors(node)
                if len(neighbors) > 0:
                    walks.remove(walk)

                for neighbor in neighbors:
                    walks.add(walk + (neighbor,))  # type: ignore

            # TODO: Should we prune in every iteration?
            if self.walks_per_graph is not None:
                n_walks = min(len(walks), self.walks_per_graph)
                walks_ix = np.random.choice(
                    range(len(walks)), replace=False, size=n_walks
                )
                if len(walks_ix) > 0:
                    walks_list = list(walks)
                    walks = {walks_list[ix] for ix in walks_ix}
        return list(walks)

    def extract(self, graph: KnowledgeGraph, instances: list) -> set:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its:
              number of rows equal to the number of provided instances;
              number of column equal to the embedding size.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(hop.name)
                    else:
                        digest = md5(hop.name.encode()).digest()[:8]
                        canonical_walk.append(str(digest))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
