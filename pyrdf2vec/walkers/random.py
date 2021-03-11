import asyncio
from hashlib import md5
from typing import Dict, List, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import Walker


@attr.s
class RandomWalker(Walker):
    """Defines the random walking strategy.

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
        random_state: The random state to use to ensure random determinism to
            generate the same walks for entities.
            Defaults to None.

    """

    def _bfs(
        self, kg: KG, root: Vertex, is_reverse: bool = False
    ) -> List[Tuple[Vertex, ...]]:
        """Extracts random walks with Breadth-first search.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            root: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors. Otherwise False.
                Defaults to False

        Returns:
            The list of walks for the root node.

        """
        walks: Set[Tuple[Vertex, ...]] = {(root,)}
        for i in range(self.depth):
            for walk in walks.copy():
                if is_reverse:
                    hops = kg.get_hops(walk[0], True)
                    for pred, obj in hops:
                        walks.add((obj, pred) + walk)
                else:
                    hops = kg.get_hops(walk[-1])
                    for pred, obj in hops:
                        walks.add(walk + (pred, obj))

                if len(hops) > 0:
                    walks.remove(walk)
        return list(walks)

    def _dfs(
        self, kg: KG, root: Vertex, is_reverse: bool = False
    ) -> List[Tuple[Vertex, ...]]:
        """Extracts a random limited number of walks with Depth-first search.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            root: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors. Otherwise False.
                Defaults to False

        Returns:
            The list of walks for the root node according to the depth and
            max_walks.

        """
        self.sampler.visited = set()
        walks: List[Tuple[Vertex, ...]] = []
        assert self.max_walks is not None
        while len(walks) < self.max_walks:
            sub_walk: Tuple[Vertex, ...] = (root,)
            d = 1
            while d // 2 < self.depth:
                pred_obj = self.sampler.sample_neighbor(
                    kg, sub_walk, d // 2 == self.depth - 1, is_reverse
                )
                if pred_obj is None:
                    break
                if is_reverse:
                    sub_walk = (pred_obj[1], pred_obj[0]) + sub_walk
                else:
                    sub_walk += (pred_obj[0], pred_obj[1])
                d = len(sub_walk) - 1
            walks.append(sub_walk)
        return list(set(walks))

    async def extract_walks(
        self, kg: KG, root: Vertex
    ) -> List[Tuple[Vertex, ...]]:
        """Extracts all possible walks.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root node to extract walks.

        Returns:
            The list of the walks.

        """
        if self.max_walks is None:
            fct_search = self._bfs
        else:
            fct_search = self._dfs
        if self.with_reverse:
            return [
                r_walk[:-1] + walk
                for walk in fct_search(kg, root)
                for r_walk in fct_search(kg, root, is_reverse=True)
            ]
        return [walk for walk in fct_search(kg, root)]

    async def _extract(
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
        literals = []
        walks = await asyncio.create_task(self.extract_walks(kg, instance))
        if not kg.mul_req:
            literals = await asyncio.create_task(
                kg.get_literals(instance.name)
            )

        canonical_walks: Set[Tuple[str, ...]] = set()
        for walk in walks:
            canonical_walk: List[str] = []
            for i, hop in enumerate(walk):
                if i == 0 or i % 2 == 1:
                    canonical_walk.append(hop.name)
                else:
                    # Use a hash to reduce memory usage of long texts by using
                    # 8 bytes per hop, except for the first hop and odd
                    # hops (predicates).
                    canonical_walk.append(
                        str(md5(hop.name.encode()).digest()[:8])
                    )
            canonical_walks.add(tuple(canonical_walk))
        return {instance.name: [tuple(canonical_walks), literals]}
