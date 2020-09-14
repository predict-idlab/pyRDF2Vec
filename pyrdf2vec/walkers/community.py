import itertools
import math
from collections import defaultdict
from hashlib import md5
from typing import Any, List, Set, Tuple

import community
import networkx as nx
import numpy as np
import rdflib

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
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


class CommunityWalker(Walker):
    """Defines the community walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().
        hop_prob: The probability to hop.
            Defaults to 0.1.
        resolution: The resolution.
            Defaults to 1.

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = UniformSampler(),
        hop_prob: float = 0.1,
        resolution: int = 1,
    ):
        super().__init__(depth, walks_per_graph, sampler)
        self.hop_prob = hop_prob
        self.resolution = resolution

    def _community_detection(self, graph: KG) -> None:
        """Converts the knowledge graph to a networkX graph.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.

        """
        nx_graph = nx.Graph()

        for v in graph._vertices:
            if not v.predicate:
                name = str(v)
                nx_graph.add_node(name, vertex=v)

        for v in graph._vertices:
            if not v.predicate:
                v_name = str(v)
                # Neighbors are predicates
                for pred in graph.get_neighbors(v):
                    pred_name = str(pred)
                    for obj in graph.get_neighbors(pred):
                        obj_name = str(obj)
                        nx_graph.add_edge(v_name, obj_name, name=pred_name)

        # This will create a dictionary that maps the URI on a community
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

    def extract_random_community_walks_bfs(self, graph, root):
        """Extract random walks of depth - 1 hops rooted in root."""
        # Initialize one walk of length 1 (the root)

        walks = {(root,)}

        for i in range(self.depth):
            # In each iteration, iterate over the walks, grab the
            # last hop, get all its neighbors and extend the walks
            walks_copy = walks.copy()
            for walk in walks_copy:
                hops = graph.get_hops(walk[-1])
                if len(hops) > 0:
                    walks.remove(walk)
                for (pred, obj) in hops:
                    walks.add(walk + (pred, obj))
                    if (
                        hops in self.communities
                        and np.random.random() < self.hop_prob
                    ):
                        community_nodes = self.labels_per_community[
                            self.communities[hops]
                        ]
                        rand_jump = np.random.choice(community_nodes)
                        walks.add(walk + (rand_jump,))

        # Return a numpy array of these walks
        return list(walks)

    def extract_random_community_walks_dfs(self, graph, root):
        """Extract random walks of depth - 1 hops rooted in root."""
        # Initialize one walk of length 1 (the root)
        self.sampler.initialize()

        walks = []
        while len(walks) < self.walks_per_graph:
            new = (root,)
            d = (len(new) - 1) // 2
            while d // 2 < self.depth:
                last = d == self.depth - 1
                hop = self.sampler.sample_neighbor(graph, new, last)
                if hop is None:
                    break
                if (
                    hop in self.communities
                    and np.random.random() < self.hop_prob
                ):
                    community_nodes = self.labels_per_community[
                        self.communities[hop]
                    ]
                    rand_jump = np.random.choice(community_nodes)
                    new = new + (rand_jump,)
                else:
                    new = new + (hop[0], hop[1])
            walks.append(new)
        return list(set(walks))

    def extract_random_community_walks(
        self, graph: KG, root: str
    ) -> List[Vertex]:
        """Extracts random walks of depth - 1 hops rooted in root.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            graph: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            root: The root.

        Returns:
            The list of walks.

        """
        if self.walks_per_graph is None:
            return self.extract_random_community_walks_bfs(graph, root)
        return self.extract_random_community_walks_dfs(graph, root)

    def extract(
        self, graph: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        self._community_detection(graph)
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_community_walks(graph, str(instance))
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
