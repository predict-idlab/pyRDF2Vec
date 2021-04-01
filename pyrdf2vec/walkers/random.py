from hashlib import md5
from typing import List, Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import Walker


@attr.s
class RandomWalker(Walker):
    """Defines the random walking strategy.

    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        md5_bytes: The number of bytes to keep after hashing objects in
            MD5. Hasher allows to reduce the memory occupied by a long
            text. If md5_bytes is None, no hash is applied.
            Defaults to 8.
        random_state: The random state to use to keep random determinism with
            the walking strategy.
            Defaults to None.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        with_reverse: True to extracts children's and parents' walks from the
            root, creating (max_walks * max_walks) more walks of 2 * depth,
            False otherwise.
            Defaults to False.
    """

    md5_bytes = attr.ib(kw_only=True, default=8, type=int, repr=False)

    def _bfs(
        self, kg: KG, root: Vertex, is_reverse: bool = False
    ) -> List[Walk]:
        """Extracts random walks with Breadth-first search.

        Args:
            kg: The Knowledge Graph.
            root: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False.

        Returns:
            The list of walks for the root node.

        """
        walks: Set[Walk] = {(root,)}
        for i in range(self.max_depth):
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
    ) -> List[Walk]:
        """Extracts a random limited number of walks with Depth-first search.

        Args:
            kg: The Knowledge Graph.
            root: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False.

        Returns:
            The list of walks for the root node according to the depth and
            max_walks.

        """
        self.sampler.visited = set()
        walks: List[Walk] = []
        assert self.max_walks is not None
        while len(walks) < self.max_walks:
            sub_walk: Walk = (root,)
            d = 1
            while d // 2 < self.max_depth:
                pred_obj = self.sampler.sample_hop(
                    kg, sub_walk, d // 2 == self.max_depth - 1, is_reverse
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

    def extract_walks(self, kg: KG, root: Vertex) -> List[Walk]:
        """Extracts all possible walks.

        Args:
            kg: The Knowledge Graph.
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

    def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided entities which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, instance):
            canonical_walk: List[str] = []
            for i, hop in enumerate(walk):
                if i == 0 or i % 2 == 1 or self.md5_bytes is None:
                    canonical_walk.append(hop.name)
                elif self.md5_bytes is not None:
                    canonical_walk.append(
                        str(md5(hop.name.encode()).digest()[: self.md5_bytes])
                    )
            canonical_walks.add(tuple(canonical_walk))
        return {instance.name: list(canonical_walks)}
