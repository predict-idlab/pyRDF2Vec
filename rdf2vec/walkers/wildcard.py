from walkers import RandomWalker
from graph import Vertex
import numpy as np
import itertools
from hashlib import md5

class WildcardWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, wildcards=[1]):
        super(WildcardWalker, self).__init__(depth, walks_per_graph)
        self.wildcards = wildcards

    def extract(self, graph, instances):
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walks.add(walk)

                for wildcard in self.wildcards:
                    combinations = itertools.combinations(range(1, len(walk)), 
                                                          wildcard)
                    for idx in combinations:
                        new_walk = list(walk).copy()
                        values = [new_walk[ix] for ix in idx]
                        for val in values:
                            new_walk.remove(val)
                        canonical_walks.add(tuple(new_walk))
        return canonical_walks
