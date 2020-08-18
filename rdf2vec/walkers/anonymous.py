from hashlib import md5

import numpy as np

from rdf2vec.graph import Vertex
from rdf2vec.walkers import RandomWalker


class AnonymousWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph):
        super(AnonymousWalker, self).__init__(depth, walks_per_graph)

    def extract(self, graph, instances):
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walk = []
                str_walk = [x.name for x in walk]
                for i, hop in enumerate(walk):
                    if i == 0:
                        canonical_walk.append(hop.name)
                    else:
                        canonical_walk.append(str(str_walk.index(hop.name)))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
