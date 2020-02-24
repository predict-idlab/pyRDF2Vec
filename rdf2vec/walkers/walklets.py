from walkers import RandomWalker
from graph import Vertex
import numpy as np
from hashlib import md5

class WalkletWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, granularity=3):
        super(WalkletWalker, self).__init__(depth, walks_per_graph)
        self.granularity = granularity

    def extract(self, graph, instances):
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for n in range(1, self.granularity + 1):
            	for walk in walks:
            		for i in range(len(walk) - n):
            			canonical_walks.add((walk[i].name, walk[i + n].name))
        return canonical_walks
