import asyncio
import itertools
import math
from collections import defaultdict
from hashlib import md5
from typing import Dict, Iterable, List, Set, Tuple

import attr
import community
import networkx as nx
import numpy as np

from pyrdf2vec.graphs import KG, Vertex
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
    """Defines the community walking strategy.

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
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.
        hop_prob: The probability to hop.
            Defaults to 0.1.
        resolution: The resolution.
            Defaults to 1.

    """

    hop_prob: float = attr.ib(
        kw_only=True, default=0.1, validator=attr.validators.instance_of(float)
    )
    resolution: int = attr.ib(
        kw_only=True, default=1, validator=attr.validators.instance_of(int)
    )
    _is_support_remote: bool = attr.ib(init=False, repr=False, default=False)

    def _community_detection(self, kg: KG) -> None:
        """Converts the knowledge graph to a networkX graph.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.

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
        self, kg: KG, root: Vertex, is_reverse: bool = False
    ) -> List[Tuple[Vertex, ...]]:
        """Extracts random walks of depth - 1 hops rooted in root with
        Breadth-first search.

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
        walks: Set[Tuple[Vertex, ...]] = {(root,)}
        for i in range(self.depth):
            for walk in walks.copy():
                if is_reverse:
                    hops = kg.get_hops(walk[0], True)
                    for pred, obj in hops:
                        walks.add((obj, pred) + walk)
                        if (
                            obj in self.communities
                            and np.random.RandomState(
                                self.random_state
                            ).random()
                            < self.hop_prob
                        ):
                            walks.add(
                                (
                                    np.random.RandomState(
                                        self.random_state
                                    ).choice(
                                        self.labels_per_community[
                                            self.communities[obj]
                                        ]
                                    ),
                                )
                                + walk
                            )
                else:
                    hops = kg.get_hops(walk[-1])
                    for pred, obj in hops:
                        walks.add(walk + (pred, obj))
                        if (
                            obj in self.communities
                            and np.random.RandomState(
                                self.random_state
                            ).random()
                            < self.hop_prob
                        ):
                            walks.add(
                                walk
                                + (
                                    np.random.RandomState(
                                        self.random_state
                                    ).choice(
                                        self.labels_per_community[
                                            self.communities[obj]
                                        ]
                                    ),
                                )
                            )
                if len(hops) > 0:
                    walks.remove(walk)
        return list(walks)

    def _dfs(
        self, kg: KG, root: Vertex, is_reverse: bool = False
    ) -> List[Tuple[Vertex, ...]]:
        """Extracts a random limited number of walks of depth - 1 hops rooted
        in root with Depth-first search.

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
                    if (
                        pred_obj[0] in self.communities
                        and np.random.RandomState(self.random_state).random()
                        < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[pred_obj[0]]
                        ]
                        sub_walk = (
                            pred_obj[1],
                            np.random.RandomState(self.random_state).choice(
                                community_nodes
                            ),
                            community_nodes,
                        ) + sub_walk
                    else:
                        sub_walk = (pred_obj[1], pred_obj[0]) + sub_walk
                else:
                    if (
                        pred_obj[1] in self.communities
                        and np.random.RandomState(self.random_state).random()
                        < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[pred_obj[1]]
                        ]
                        sub_walk += (
                            pred_obj[0],
                            np.random.RandomState(self.random_state).choice(
                                community_nodes
                            ),
                            community_nodes,
                        )
                    else:
                        sub_walk += (pred_obj[0], pred_obj[1])
                d = len(sub_walk) - 1
            walks.append(sub_walk)
        return list(set(walks))

    def extract(
        self,
        kg: KG,
        instances: List[str],
        verbose: int = 0,
    ) -> Iterable[str]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to be extracted from the Knowledge Graph.
            verbose: If equal to 1 or 2, display a progress bar for the
                extraction of the walks.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        self._community_detection(kg)
        return super().extract(kg, instances, verbose)

    def extract_walks(self, kg: KG, root: Vertex) -> List[Tuple[Vertex, ...]]:
        """Extracts random walks of depth - 1 hops rooted in root.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root node to extract walks.

        Returns:
            The list of walks.

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

    def _extract(
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
        canonical_walks: Set[Tuple[str, ...]] = set()
        for walk in self.extract_walks(kg, instance):
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
        return {instance.name: canonical_walks}
