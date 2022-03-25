import itertools
import math
from collections import defaultdict
from hashlib import md5
from typing import List, Optional, Set

import attr
import community
import networkx as nx
import numpy as np

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import Entities, EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import Walker


def check_random_state(seed):
    return np.random


community.community_louvain.check_random_state = check_random_state


def sample_from_iterable(x):
    perms = itertools.permutations(x)
    length = math.factorial(len(x))
    rand_ix = np.random.randint(min(length, 10000))
    for _ in range(rand_ix):
        _ = next(perms)
    return next(perms)


np.random.permutation = lambda x: next(itertools.permutations(x))


@attr.s
class CommunityWalker(Walker):
    """Community walking strategy which groups vertices with similar properties
    through probabilities and relations that are not explicitly modeled in a
    Knowledge Graph. Similar to the Random walking strategy, the Depth First
    Search (DFS) algorithm is used if a maximum number of walks is specified.
    Otherwise, the Breadth First Search (BFS) algorithm is chosen.

    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise.
            Defaults to True.
        hop_prob: The probability to hop.
            Defaults to 0.1.
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
        resolution: The resolution to use.
            Defaults to The resolution to use.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        with_reverse: True to extracts parents and children hops from an
            entity, creating (max_walks * max_walks) walks of 2 * depth,
            allowing also to centralize this entity in the walks. False
            otherwise.
            Defaults to False.

    """

    hop_prob = attr.ib(
        kw_only=True,
        default=0.1,
        type=float,
        validator=attr.validators.instance_of(float),
    )

    md5_bytes = attr.ib(
        kw_only=True,
        type=Optional[int],
        default=8,
        repr=False,
    )

    resolution = attr.ib(
        kw_only=True,
        default=1,
        type=int,
        validator=attr.validators.instance_of(int),
    )

    _is_support_remote = attr.ib(
        init=False, repr=False, type=bool, default=False
    )

    def _community_detection(self, kg: KG) -> None:
        """Converts the knowledge graph to a networkX graph.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            kg: The Knowledge Graph.

        """
        nx_graph = nx.Graph()

        for vertex in kg._vertices:
            if not vertex.predicate:
                nx_graph.add_node(str(vertex), vertex=vertex)

        for vertex in kg._vertices:
            if not vertex.predicate:
                # Neighbors are predicates
                for pred in kg.get_neighbors(vertex):
                    for obj in kg.get_neighbors(pred):
                        nx_graph.add_edge(
                            str(vertex), str(obj), name=str(pred)
                        )

        # Create a dictionary that maps the URI on a community
        partition = community.best_partition(
            nx_graph, resolution=self.resolution
        )
        self.labels_per_community = defaultdict(list)

        self.communities = {}
        vertices = nx.get_node_attributes(nx_graph, "vertex")
        for node in partition:
            if node in vertices:
                self.communities[vertices[node]] = partition[node]

        for node in self.communities:
            self.labels_per_community[self.communities[node]].append(node)

    def _bfs(
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
        walks: Set[Walk] = {(entity,)}
        rng = np.random.RandomState(self.random_state)
        for i in range(self.max_depth):
            for walk in walks.copy():
                if is_reverse:
                    hops = kg.get_hops(walk[0], True)
                    for pred, obj in hops:
                        walks.add((obj, pred) + walk)
                        if (
                            obj in self.communities
                            and rng.random() < self.hop_prob
                        ):
                            comm = self.communities[obj]
                            comm_labels = self.labels_per_community[comm]
                            walks.add((rng.choice(comm_labels),) + walk)
                else:
                    hops = kg.get_hops(walk[-1])
                    for pred, obj in hops:
                        walks.add(walk + (pred, obj))
                        if (
                            obj in self.communities
                            and rng.random() < self.hop_prob
                        ):
                            comm = self.communities[obj]
                            comm_labels = self.labels_per_community[comm]
                            walks.add(walk + (rng.choice(comm_labels),))
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

        rng = np.random.RandomState(self.random_state)

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
                    if (
                        pred_obj[0] in self.communities
                        and rng.random() < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[pred_obj[0]]
                        ]
                        sub_walk = (
                            pred_obj[1],
                            rng.choice(community_nodes),
                        ) + sub_walk
                    else:
                        sub_walk = (pred_obj[1], pred_obj[0]) + sub_walk
                else:
                    if (
                        pred_obj[1] in self.communities
                        and rng.random() < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[pred_obj[1]]
                        ]
                        sub_walk += (
                            pred_obj[0],
                            rng.choice(community_nodes),
                        )
                    else:
                        sub_walk += (pred_obj[0], pred_obj[1])
                d = len(sub_walk) - 1
            walks.append(sub_walk)
        return list(walks)

    def extract(
        self, kg: KG, entities: Entities, verbose: int = 0
    ) -> List[List[SWalk]]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            entities: The entities to be extracted from the Knowledge Graph.
            verbose: The verbosity level.
                0: does not display anything;
                1: display of the progress of extraction and training of walks;
                2: debugging.
                Defaults to 0.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """

        self._community_detection(kg)
        return super().extract(kg, entities, verbose)

    def extract_walks(self, kg: KG, entity: Vertex) -> List[Walk]:
        """Extracts random walks of depth - 1 hops rooted in root.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            The list of unique walks for the provided entity.

        """
        if self.max_walks is None:
            fct_search = self._bfs
        else:
            fct_search = self._dfs
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
