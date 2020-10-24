from hashlib import md5
from typing import Any, List, Set, Tuple

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import Walker


class RandomWalker(Walker):
    """Defines the random walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph,
        sampler: Sampler = UniformSampler(),
    ):
        super().__init__(depth, walks_per_graph, sampler)

    def extract_random_walks_bfs(self, graph, root):
        """Breadth-first search to extract all possible walks."""
        walks = {(root,)}
        for i in range(self.depth):
            walks_copy = walks.copy()
            for walk in walks_copy:
                hops = graph.get_hops(walk[-1])
                if len(hops) > 0:
                    walks.remove(walk)
                for (pred, obj) in hops:
                    walks.add(walk + (pred, obj))
        return list(walks)

    def extract_random_walks_dfs(self, graph, root):
        """Depth-first search to extract a limited number of walks."""
        # TODO: Currently we are allowing duplicate walks in order
        # TODO: to avoid infinite loops. Can we do this better?

        self.sampler.initialize()

        walks = []
        while len(walks) < self.walks_per_graph:
            new = (root,)
            d = 1
            while d // 2 < self.depth:
                last = d // 2 == self.depth - 1
                hop = self.sampler.sample_neighbor(graph, new, last)
                if hop is None:
                    break
                new = new + (hop[0], hop[1])
                d = len(new) - 1
            walks.append(new)
        return list(set(walks))

    def extract_random_walks(self, kg: KG, root: str) -> List[Vertex]:
        """Breadth-first search to extract all possible walks.

        Args:
            kg: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root.

        Returns:
            The list of the walks.

        """
        if self.walks_per_graph is None:
            return self.extract_random_walks_bfs(kg, root)
        return self.extract_random_walks_dfs(kg, root)

    def _extract(
        self, kg: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Extracts the walks and processes them for the embedding model.

        Args:
            kg: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks = set()
        for i, instance in enumerate(instances):
            walks = self.extract_random_walks(kg, instance)
            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):  # type: ignore
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(str(hop))
                    else:
                        digest = md5(str(hop).encode()).digest()[:8]
                        canonical_walk.append(str(digest))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
