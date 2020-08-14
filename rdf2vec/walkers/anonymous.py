from rdf2vec.walkers import RandomWalker
from rdf2vec.graph import Vertex
import numpy as np
from hashlib import md5

class AnonymousWalker(RandomWalker):
    """Defines the anonymous walker of the walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth, walks_per_graph):
        super(AnonymousWalker, self).__init__(depth, walks_per_graph)

    def extract(self, graph, instances):
        """Extracts a knowledge graph and transform it into a 2D vector, based
        on provided instances.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (array-like): The instances to extract the knowledge graph.

        Returns:
            list: The 2D vector corresponding to the knowledge graph.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walk = []
                str_walk = [x.name for x in walk]
                for i, hop in enumerate(walk):
                    if i == 0:# or i % 2 == 1:
                        canonical_walk.append(hop.name)
                    else:
                        canonical_walk.append(str(str_walk.index(hop.name)))
                canonical_walks.add(tuple(canonical_walk))

        return canonical_walks
