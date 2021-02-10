import itertools
import math
from collections import defaultdict
from hashlib import md5
from typing import Any, Dict, List, Tuple

import attr
import community
import networkx as nx
import numpy as np
import rdflib

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

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        hop_prob: The probability to hop.
            Defaults to 0.1.
        resolution: The resolution.
            Defaults to 1.
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        seed: The seed to use to ensure ensure random determinism to generate
            the same walks for entities.
            Defaults to None.

    """

    hop_prob: float = attr.ib(kw_only=True, default=0.1)
    resolution: int = attr.ib(kw_only=True, default=1)

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

    def extract_walks_bfs(self, kg: KG, root: str):
        """Extracts random walks of depth - 1 hops rooted in root with
        Breadth-first search.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            root: The root node.

        Returns:
            The list of walks for the root node.

        """
        walks = {(root,)}

        for i in range(self.depth):
            # In each iteration, iterate over the walks, grab the
            # last hop, get all its neighbors and extend the walks
            walks_copy = walks.copy()
            for walk in walks_copy:
                hops = kg.get_hops(walk[-1])
                if len(hops) > 0:
                    walks.remove(walk)
                for (pred, obj) in hops:
                    walks.add(walk + (pred, obj))  # type: ignore
                    if (
                        obj in self.communities
                        and np.random.RandomState(self.seed) < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[obj]
                        ]
                        rand_jump = np.random.RandomState(self.seed).choice(
                            community_nodes
                        )
                        walks.add(walk + (rand_jump,))  # type: ignore

        # Return a numpy array of these walks
        return list(walks)

    def extract_walks_dfs(self, kg: KG, root: str):
        """Extracts a random limited number of walks of depth - 1 hops rooted
        in root with Depth-first search.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            root: The root node.

        Returns:
            The list of limited  walks for the root node.

        """
        # Initialize one walk of length 1 (the root)
        self.sampler.initialize()

        walks: List[Tuple[str]] = []
        while len(walks) < self.max_walks:  # type: ignore
            new = (root,)
            d = 1
            while d // 2 < self.depth:
                last = d // 2 == self.depth - 1
                hop = self.sampler.sample_neighbor(kg, new, last)
                if hop is None:
                    break
                if (
                    hop[1] in self.communities
                    and np.random.RandomState(self.seed).random()
                    < self.hop_prob
                ):
                    community_nodes = self.labels_per_community[
                        self.communities[hop[1]]
                    ]
                    rand_jump = np.random.RandomState(self.seed).choice(
                        community_nodes
                    )
                    new = new + (hop[0], rand_jump)  # type: ignore
                else:
                    new = new + (hop[0], hop[1])  # type: ignore
                d = len(new) - 1
            walks.append(new)
        return list(set(walks))

    def extract_random_community_walks(
        self, kg: KG, root: str
    ) -> List[Vertex]:
        """Extracts random walks of depth - 1 hops rooted in root.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            kg: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root.

        Returns:
            The list of walks.

        """
        if self.max_walks is None:
            return self.extract_walks_bfs(kg, root)
        return self.extract_walks_dfs(kg, root)

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
        for walk in self.extract_random_community_walks(kg, str(instance)):
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
