import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from collections import defaultdict, Counter
from functools import lru_cache

import os

# The idea of using a hashing function is taken from https://github.com/benedekrozemberczki/graph2vec
from hashlib import md5


class Vertex(object):
    vertex_counter = 0
    
    def __init__(self, name, predicate=False, _from=None, _to=None, wildcard=False):
        self.name = name
        self.predicate = predicate
        self._from = _from
        self._to = _to
        self.wildcard = wildcard

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

class KnowledgeGraph(object):
    def __init__(self):
        self.vertices = set()
        self.transition_matrix = defaultdict(set)
        self.label_map = {}
        self.inv_label_map = {}
        self.name_to_vertex = {}
        self.root = None
        
    def add_vertex(self, vertex):
        if vertex.predicate:
            self.vertices.add(vertex)
            
        if not vertex.predicate and vertex not in self.vertices:
            self.vertices.add(vertex)

        self.name_to_vertex[vertex.name] = vertex

    def add_edge(self, v1, v2):
        # Uni-directional edge
        self.transition_matrix[v1].add(v2)
        
    def remove_edge(self, v1, v2):
        if v2 in self.transition_matrix[v1]:
            self.transition_matrix[v1].remove(v2)

    def get_neighbors(self, vertex):
        return self.transition_matrix[vertex]
    
    def visualise(self):
        nx_graph = nx.DiGraph()
        
        for v in self.vertices:
            if not v.predicate:
                name = v.name.split('/')[-1]
                nx_graph.add_node(name, name=name, pred=v.predicate)
            
        for v in self.vertices:
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
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, 
                                     edge_labels=nx.get_edge_attributes(nx_graph, 'name'))
        plt.show()
    
    def _create_label(self, vertex, n):
        neighbor_names = [self.label_map[x][n - 1] for x in self.get_neighbors(vertex)]
        suffix = '-'.join(sorted(set(map(str, neighbor_names))))
        return self.label_map[vertex][n - 1] + '-' + suffix
        
    def weisfeiler_lehman(self, iterations=3):
        # Store the WL labels in a dictionary with a two-level key:
        # First level is the vertex identifier
        # Second level is the WL iteration
        self.label_map = defaultdict(dict)
        self.inv_label_map = defaultdict(dict)

        for v in self.vertices:
            self.label_map[v][0] = v.name
            self.inv_label_map[v.name][0] = v
        
        for n in range(1, iterations+1):

            for vertex in self.vertices:
                # Create multi-set label
                s_n = self._create_label(vertex, n)

                # Store it in our label_map (hash trick from: benedekrozemberczki/graph2vec)
                self.label_map[vertex][n] = str(md5(s_n.encode()).digest())

        for vertex in self.vertices:
            for key, val in self.label_map[vertex].items():
                self.inv_label_map[vertex][val] = key

    def extract_random_walks(self, depth, max_walks=None):
        # Initialize one walk of length 1 (the root)
        walks = [[self.root]]

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
                    walks.append(list(walk) + [neighbor])

            # TODO: Should we prune in every iteration?
            if max_walks is not None:
                walks_ix = np.random.choice(range(len(walks)), replace=False, 
                                            size=min(len(walks), max_walks))
                if len(walks_ix) > 0:
                    walks = np.array(walks)[walks_ix].tolist()

        # Return a numpy array of these walks
        return np.array(walks)

def rdflib_to_kg(rdflib_g, label_predicates=[]):
    # TODO: Make sure to filter out all tripels where p in label_predicates!
    # Iterate over triples, add s, p and o to graph and 2 edges (s-->p, p-->o)
    kg = KnowledgeGraph()
    for (s, p, o) in rdflib_g:
        if p not in label_predicates:
            s_v, o_v = Vertex(str(s)), Vertex(str(o))
            p_v = Vertex(str(p), predicate=True)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
    return kg

def extract_instance(kg, instance, depth=8):
    subgraph = KnowledgeGraph()
    subgraph.label_map = kg.label_map
    subgraph.inv_label_map = kg.inv_label_map
    root = kg.name_to_vertex[str(instance)]
    to_explore = { root }
    subgraph.add_vertex( root )
    subgraph.root = root
    for d in range(depth):
        for v in list(to_explore):
            for neighbor in kg.get_neighbors(v):
                subgraph.add_vertex(neighbor)
                subgraph.add_edge(v, neighbor)
                to_explore.add(neighbor)
            to_explore.remove(v)
    return subgraph