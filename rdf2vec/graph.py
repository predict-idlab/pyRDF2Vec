import numpy as np
from collections import defaultdict


class Vertex(object):
    vertex_counter = 0
    
    def __init__(self, name, predicate=False, _from=None, _to=None):
        self.name = name
        self.predicate = predicate
        self._from = _from
        self._to = _to

        self.id = Vertex.vertex_counter
        Vertex.vertex_counter += 1
        
    def __eq__(self, other):
        if other is None: 
            return False
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        if self.predicate:
            return hash((self.id, self._from, self._to, self.name))
        else:
            return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


class KnowledgeGraph(object):
    def __init__(self):
        self._vertices = set()
        self._transition_matrix = defaultdict(set)
        self._label_map = {}
        self._inv_label_map = {}
        
    def add_vertex(self, vertex):
        """Add a vertex to the Knowledge Graph."""
        if vertex.predicate:
            self._vertices.add(vertex)
        else:
            self._vertices.add(vertex)

    def add_edge(self, v1, v2):
        """Add a uni-directional edge."""
        self._transition_matrix[v1].add(v2)
        
    def remove_edge(self, v1, v2):
        """Remove the edge v1 -> v2 if present."""
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)

    def get_neighbors(self, vertex):
        """Get all the neighbors of vertex (vertex -> neighbor)."""
        return self._transition_matrix[vertex]
    
    def visualise(self):
        """Visualise the graph using networkx & matplotlib."""
        import matplotlib.pyplot as plt
        import networkx as nx
        nx_graph = nx.DiGraph()
        
        for v in self._vertices:
            if not v.predicate:
                name = v.name.split('/')[-1]
                nx_graph.add_node(name, name=name, pred=v.predicate)
            
        for v in self._vertices:
            if not v.predicate:
                v_name = v.name.split('/')[-1]
                # Neighbors are predicates
                for pred in self.get_neighbors(v):
                    pred_name = pred.name.split('/')[-1]
                    for obj in self.get_neighbors(pred):
                        obj_name = obj.name.split('/')[-1]
                        nx_graph.add_edge(v_name, obj_name, name=pred_name)
        
        plt.figure(figsize=(10,10))
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        names = nx.get_edge_attributes(nx_graph, 'name')
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names)
        plt.show()
    
    def _create_label(self, vertex, n):
        """Take labels of neighbors, sort them lexicographically and join."""
        neighbor_names = [self._label_map[x][n - 1] 
                          for x in self.get_neighbors(vertex)]
        suffix = '-'.join(sorted(set(map(str, neighbor_names))))
        return self._label_map[vertex][n - 1] + '-' + suffix
        
    def weisfeiler_lehman(self, iterations=3):
        """Perform Weisfeiler-Lehman relabeling of the nodes."""
        # The idea of using a hashing function is taken from:
        # https://github.com/benedekrozemberczki/graph2vec
        from hashlib import md5
        # Store the WL labels in a dictionary with a two-level key:
        # First level is the vertex identifier
        # Second level is the WL iteration
        self._label_map = defaultdict(dict)
        self._inv_label_map = defaultdict(dict)

        for v in self._vertices:
            self._label_map[v][0] = v.name
            self._inv_label_map[v.name][0] = v
        
        for n in range(1, iterations+1):

            for vertex in self._vertices:
                # Create multi-set label
                s_n = self._create_label(vertex, n)

                # Store it in our label_map
                self._label_map[vertex][n] = str(md5(s_n.encode()).digest())

        for vertex in self._vertices:
            for key, val in self._label_map[vertex].items():
                self._inv_label_map[vertex][val] = key

    def extract_random_walks(self, depth, root, max_walks=None):
        """Extract random walks of depth - 1 hops rooted in root."""
        # Initialize one walk of length 1 (the root)
        walks = {(root,)}

        for i in range(depth):
            # In each iteration, iterate over the walks, grab the 
            # last hop, get all its neighbors and extend the walks
            walks_copy = walks.copy()
            for walk in walks_copy:
                node = walk[-1]
                neighbors = self.get_neighbors(node)

                if len(neighbors) > 0:
                    walks.remove(walk)

                for neighbor in neighbors:
                    walks.add(walk + (neighbor, ))

            # TODO: Should we prune in every iteration?
            if max_walks is not None:
                walks_ix = np.random.choice(range(len(walks)), replace=False, 
                                            size=min(len(walks), max_walks))
                if len(walks_ix) > 0:
                    walks_list = list(walks)
                    walks = {walks_list[ix] for ix in walks_ix}

        # Return a numpy array of these walks
        return list(walks)

def rdflib_to_kg(rdflib_g, label_predicates=[]):
    """Convert a rdflib.Graph to our KnowledgeGraph."""
    kg = KnowledgeGraph()
    for (s, p, o) in rdflib_g:
        if p not in label_predicates:
            s_v, o_v = Vertex(str(s)), Vertex(str(o))
            p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
    return kg