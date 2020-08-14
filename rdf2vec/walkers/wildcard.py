from rdf2vec.walkers import RandomWalker
from rdf2vec.graph import Vertex
import numpy as np
import itertools
from hashlib import md5

class WildcardWalker(RandomWalker):
    """Defines the wild card of the walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.
    """

    def __init__(self, depth, walks_per_graph, wildcards=[1]):
        super(WildcardWalker, self).__init__(depth, walks_per_graph)
        self.wildcards = wildcards

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
                canonical_walks.add(tuple([x.name for x in walk]))

                for wildcard in self.wildcards:
                    combinations = itertools.combinations(range(1, len(walk)), 
                                                          wildcard)
                    for idx in combinations:
                        new_walk = []
                        for ix, hop in enumerate(walk):
                            if ix in idx:
                                new_walk.append(Vertex('*'))
                            else:
                                new_walk.append(hop.name)
                        canonical_walks.add(tuple(new_walk))
        return canonical_walks
