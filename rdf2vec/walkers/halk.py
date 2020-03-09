from collections import defaultdict
from walkers import RandomWalker
from graph import Vertex
import numpy as np
from hashlib import md5

class HalkWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, lb_freq_threshold=0.001,
                 ub_freq_threshold=0.1):
        super(HalkWalker, self).__init__(depth, walks_per_graph)
        self.ub_freq_threshold = ub_freq_threshold
        self.lb_freq_threshold = lb_freq_threshold

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

        uniformative_hops = set()
        for hop in freq:
            if len(freq[hop])/len(all_walks) > self.ub_freq_threshold:
                uniformative_hops.add(hop)
            if len(freq[hop])/len(all_walks) < self.lb_freq_threshold:
                uniformative_hops.add(hop)

        for walk in all_walks:
            canonical_walk = []
            for i, hop in enumerate(walk):
                if i == 0:
                    canonical_walk.append(hop.name)
                else:
                    if hop.name not in uniformative_hops:
                        digest = md5(hop.name.encode()).digest()[:8]
                        canonical_walk.append(str(digest))
            canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
