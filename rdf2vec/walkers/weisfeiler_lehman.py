from collections import defaultdict
from hashlib import md5

from rdf2vec.graph import Vertex
from rdf2vec.walkers import RandomWalker


class WeisfeilerLehmanWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph, wl_iterations=4):
        super(WeisfeilerLehmanWalker, self).__init__(depth, walks_per_graph)
        self.wl_iterations = wl_iterations
    
    def _create_label(self, graph, vertex, n):
        """Take labels of neighbors, sort them lexicographically and join."""
        neighbor_names = [self._label_map[x][n - 1] 
                          for x in graph.get_inv_neighbors(vertex)]
        suffix = '-'.join(sorted(set(map(str, neighbor_names))))

        # TODO: Experiment with not adding the prefix
        return self._label_map[vertex][n - 1] + '-' + suffix
        # return suffix

    def _weisfeiler_lehman(self, graph):
        """Perform Weisfeiler-Lehman relabeling of the vertices"""
        self._label_map = defaultdict(dict)
        self._inv_label_map = defaultdict(dict)

        for v in graph._vertices:
            self._label_map[v][0] = v.name
            self._inv_label_map[v.name][0] = v
        
        for n in range(1, self.wl_iterations+1):
            for vertex in graph._vertices:
                # Create multi-set label
                s_n = self._create_label(graph, vertex, n)
                # Store it in our label_map
                self._label_map[vertex][n] = str(md5(s_n.encode()).digest())

        for vertex in graph._vertices:
            for key, val in self._label_map[vertex].items():
                self._inv_label_map[vertex][val] = key


    def extract(self, graph, instances):
        self._weisfeiler_lehman(graph)

        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for n in range(self.wl_iterations + 1):
                for walk in walks:
                    canonical_walk = []
                    for i, hop in enumerate(walk):
                        if i == 0 or i % 2 == 1:
                            canonical_walk.append(hop.name)
                        else:
                            canonical_walk.append(self._label_map[hop][n])

                    canonical_walks.add(tuple(canonical_walk))
                
        return canonical_walks
