from hashlib import md5
from typing import List, Optional, Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import Walker


@attr.s
class RandomWalker(Walker):
    """Random walking strategy which extracts walks from a root node using the
    Depth First Search (DFS) algorithm if a maximum number of walks is
    specified, otherwise the Breadth First Search (BFS) algorithm is used.

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
        with_reverse: True to extracts parents and children hops from an
            entity, creating (max_walks * max_walks) walks of 2 * depth,
            allowing also to centralize this entity in the walks. False
            otherwise.
            Defaults to False.

    """

    md5_bytes = attr.ib(
        kw_only=True,
        type=Optional[int],
        default=8,
        repr=False,
    )

    def _bfs(
        self, kg: KG, entity: Vertex, is_reverse: bool = False
    ) -> List[Walk]:
        """Extracts random walks for an entity based on Knowledge Graph using
        the Breadth First Search (BFS) algorithm.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False.

        Returns:
            The list of unique walks for the provided entity.

        """
        walks: Set[Walk] = {(entity,)}
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
        self, kg: KG, entity: Vertex, is_reverse: bool = False
    ) -> List[Walk]:
        """Extracts random walks for an entity based on Knowledge Graph using
        the Depth First Search (DFS) algorithm.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False.

        Returns:
            The list of unique walks for the provided entity.

        """
        self.sampler.visited = set()
        walks: List[Walk] = []
        assert self.max_walks is not None
        while len(walks) < self.max_walks:
            sub_walk: Walk = (entity,)
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
        return list(walks)

    def extract_walks(self, kg: KG, entity: Vertex) -> List[Walk]:
        """Extracts random walks for an entity based on Knowledge Graph using
        the Depth First Search (DFS) algorithm if a maximum number of walks is
        specified, otherwise the Breadth First Search (BFS) algorithm is used.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            The list of unique walks for the provided entity.

        """
        fct_search = self._bfs if self.max_walks is None else self._dfs
        if self.with_reverse:
            return [
                r_walk[:-1] + walk
                for walk in fct_search(kg, entity)
                for r_walk in fct_search(kg, entity, is_reverse=True)
            ]
        return [walk for walk in fct_search(kg, entity)]

    def _map_vertex(self, entity: Vertex, pos: int) -> str:
        """Maps certain vertices to MD5 hashes to save memory. For entities of
        interest (provided by the user to the extract function) and predicates,
        the string representation is kept.

        Args:
            entity: The entity to be mapped.
            pos: The position of the entity in the walk.

        Returns:
            A hash (string) or original string representation.

        """
        if (
            entity.name in self._entities
            or pos % 2 == 1
            or self.md5_bytes is None
        ):
            return entity.name
        else:
            ent_hash = md5(entity.name.encode()).digest()
            return str(ent_hash[: self.md5_bytes])

    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts random walks for an entity based on a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            A dictionary having the entity as key and a list of tuples as value
            corresponding to the extracted walks.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, entity):
            canonical_walk: List[str] = [
                self._map_vertex(vertex, i) for i, vertex in enumerate(walk)
            ]
            canonical_walks.add(tuple(canonical_walk))
        return {entity.name: list(canonical_walks)}
