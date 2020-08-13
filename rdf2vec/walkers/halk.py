from collections import defaultdict
from hashlib import md5

import numpy as np

from rdf2vec.graph import Vertex
from rdf2vec.walkers import RandomWalker


class HalkWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, 
                 freq_thresholds=[0.001]):
        super(HalkWalker, self).__init__(depth, walks_per_graph)
        self.freq_thresholds = freq_thresholds
        # self.lb_freq_threshold = lb_freq_threshold

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

        for freq_threshold in self.freq_thresholds:
            uniformative_hops = set()
            for hop in freq:
                # if len(freq[hop])/len(all_walks) > self.ub_freq_threshold:
                #     uniformative_hops.add(hop)
                if len(freq[hop])/len(all_walks) < freq_threshold:
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
