import itertools
import math
from collections import defaultdict
from hashlib import md5

import numpy as np

import community
import networkx as nx
from rdf2vec.graph import Vertex
from rdf2vec.walkers import Walker


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

np.random.permutation = lambda x: next(
    itertools.permutations(x)
)  # sample_from_iterable
class CommunityWalker(Walker):
    """Defines the community walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.
        hop_prob (float): The probability to hop.
            Defaults to 0.1.
        resolution (int): The resolution.
            Defaults to 1.

    """
    def __init__(self, depth, walks_per_graph, hop_prob=0.1, resolution=1):
        super(CommunityWalker, self).__init__(depth, walks_per_graph)
        self.hop_prob = hop_prob
        self.resolution = resolution

    def _community_detection(self, graph):
        """Converts the knowledge graph to a networkX graph.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.

        """
        nx_graph = nx.Graph()

        for v in graph._vertices:
            if not v.predicate:
                name = v.name
                nx_graph.add_node(name, vertex=v)

        for v in graph._vertices:
            if not v.predicate:
                v_name = v.name
                # Neighbors are predicates
                for pred in graph.get_neighbors(v):
                    pred_name = pred.name
                    for obj in graph.get_neighbors(pred):
                        obj_name = obj.name
                        nx_graph.add_edge(v_name, obj_name)

        # This will create a dictionary that maps the URI on a community
        partition = community.best_partition(
            nx_graph, resolution=self.resolution
        )
        self.labels_per_community = defaultdict(list)

        self.communities = {}
        vertices = nx.get_node_attributes(nx_graph, 'vertex')
        for node in partition:
            if node in vertices:
                self.communities[vertices[node]] = partition[node]

        for node in self.communities:
            self.labels_per_community[self.communities[node]].append(node)

    def extract_random_community_walks(self, graph, root):
        """Extracts random walks of depth - 1 hops rooted in root.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            root (Vertex): The root.

        Returns:
            numpy.array: The array of the walks.

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
                    walks.add(walk + (neighbor, ))
                    if neighbor in self.communities and np.random.random() < self.hop_prob:
                        community_nodes = self.labels_per_community[self.communities[neighbor]]
                        rand_jump = np.random.choice(community_nodes)
                        walks.add(walk + (rand_jump, ))


            # TODO: Should we prune in every iteration?
            if self.walks_per_graph is not None:
                n_walks = min(len(walks), self.walks_per_graph)
                walks_ix = np.random.choice(
                    range(len(walks)), replace=False, size=n_walks
                )
                if len(walks_ix) > 0:
                    walks_list = list(walks)
                    walks = {walks_list[ix] for ix in walks_ix}

        # Return a numpy array of these walks
        return list(walks)

    def extract(self, graph, instances):
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (array-like): The instances to extract the knowledge graph.

        Returns:
            list: The 2D matrix with its:
                number of rows equal to the number of provided instances;
                number of column equal to the embedding size.

        """
        self._community_detection(graph)
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_community_walks(
                graph, Vertex(str(instance))
            )
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
