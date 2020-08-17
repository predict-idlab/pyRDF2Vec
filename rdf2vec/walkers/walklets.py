from rdf2vec.walkers import RandomWalker
from rdf2vec.graph import Vertex
import numpy as np
from hashlib import md5

class WalkletWalker(RandomWalker):
    """Defines the walklet walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth, walks_per_graph):
        super(WalkletWalker, self).__init__(depth, walks_per_graph)

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
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                for n in range(1, len(walk)):
                    canonical_walks.add((walk[0].name, walk[n].name))
        return canonical_walks
