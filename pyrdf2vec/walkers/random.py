from hashlib import md5
from typing import Any, Dict, List, Tuple

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
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        is_support_remote: If true, indicate that the walking strategy can be
            used to retrieve walks via a SPARQL endpoint server.
            Defaults to False.
    """

    def __init__(
        self,
        depth: int,
        walks_per_graph,
        sampler: Sampler = UniformSampler(),
        n_jobs: int = 1,
        is_support_remote=True,
    ):
        super().__init__(
            depth, walks_per_graph, sampler, n_jobs, is_support_remote
        )

    def extract_random_walks_bfs(self, kg: KG, root: str):
        """Breadth-first search to extract all possible walks.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            root: The root node.

        Returns:
            The list of the walks.

        """
        walks = {(root,)}
        for i in range(self.depth):
            for walk in walks.copy():
                hops = kg.get_hops(walk[-1])
                if len(hops) > 0:
                    walks.remove(walk)
                for (pred, obj) in hops:
                    walks.add(walk + (pred, obj))  # type: ignore
        return list(walks)

    def extract_random_walks_dfs(self, kg: KG, root: str):
        """Depth-first search to extract a limited number of walks."""
        # TODO: Currently we are allowing duplicate walks in order
        # TODO: to avoid infinite loops. Can we do this better?

        self.sampler.initialize()

        walks: List[Tuple[Any, ...]] = []
        while len(walks) < self.walks_per_graph:
            new = (root,)
            d = 1  # type: ignore
            while d // 2 < self.depth:
                last = d // 2 == self.depth - 1
                hop = self.sampler.sample_neighbor(kg, new, last)
                if hop is None:
                    break
                new = new + (hop[0], hop[1])  # type:ignore
                d = len(new) - 1
            walks.append(new)
        return list(set(walks))

    def extract_random_walks(self, kg: KG, root: str) -> List[Vertex]:
        """Extract all possible walks.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root node.

        Returns:
            The list of the walks.

        """
        if self.walks_per_graph is None:
            return self.extract_random_walks_bfs(kg, root)
        return self.extract_random_walks_dfs(kg, root)

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
        for walk in self.extract_random_walks(kg, instance):
            canonical_walk = []
            for i, hop in enumerate(walk):  # type: ignore
                if i == 0 or i % 2 == 1:
                    canonical_walk.append(str(hop))
                else:
                    canonical_walk.append(
                        str(md5(str(hop).encode()).digest()[:8])
                    )
            canonical_walks.add(tuple(canonical_walk))
        return {instance: tuple(canonical_walks)}
