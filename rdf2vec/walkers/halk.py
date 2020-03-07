from collections import defaultdict
from walkers import RandomWalker
from graph import Vertex
import numpy as np
from hashlib import md5

class HalkWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, freq_threshold=0.5):
        super(HalkWalker, self).__init__(depth, walks_per_graph)
        self.freq_threshold = freq_threshold

    def extract(self, graph, instances):
        canonical_walks = set()
        all_walks=[]
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            all_walks.extend(walks)

        freq = defaultdict(set)
        for i in range(len(all_walks)):
            for hop in all_walks[i]:
                freq[hop.name].add(i)

        most_frequent_hops = set()
        for hop in freq:
            if len(freq[hop])/len(all_walks) > self.freq_threshold:
                most_frequent_hops.add(hop)

        for walk in all_walks:
            canonical_walk = []
            for i, hop in enumerate(walk):
                if i == 0:
                    canonical_walk.append(hop.name)
                else:
                    if hop.name not in most_frequent_hops:
                        digest = md5(hop.name.encode()).digest()[:8]
                        canonical_walk.append(str(digest))
            canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
