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
    
    @lru_cache(maxsize=2048)
    def find_walk(self, walk):
        #print(walk)
        # Process first element of walk: entity, root or wildcard
        if walk[0][1] is 'root':
            to_explore = { self.root }
        elif walk[0][1] == '*':
            to_explore = self.vertices
        else:
            if walk[0][1] not in self.name_to_vertex: 
                return False
            to_explore = { self.name_to_vertex[walk[0][1]] }
        
        # Process second element until end. Alternate between entities and predicates.
        for hop_nr, (depth, hop) in enumerate(walk[1:]):

            new_explore = set()
            if hop_nr % 2 > 0:  # Entity
                if hop == '*':
                    for node in to_explore:
                        for neighbor in self.get_neighbors(node):
                            new_explore.add(neighbor)
                else:
                    for node in to_explore:
                        if hop in self.name_to_vertex:
                            new_explore.add(self.name_to_vertex[hop])
            else:  # Predicate
                for node in to_explore:
                    for neighbor in self.get_neighbors(node):
                        if hop == '*' or neighbor.name == hop:
                            new_explore.add(neighbor)

            to_explore = new_explore
            
            if len(to_explore) == 0:
                return False

        return True


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

    def convert_to_rgcn(self, train_inst, test_inst, train_labels, test_labels):
        # Transform our KnowledgeGraph into the required data structured for RGCN

        # Map all non-predicate vertices to an id
        vertex_to_id = {}
        cntr = 0
        for i, v in enumerate(filter(lambda x: not x.predicate, self.vertices)):
            vertex_to_id[v.name] = i

        adj_shape = (len(vertex_to_id), len(vertex_to_id))
        adjacencies = defaultdict(list)

        for subject in self.vertices:
            if not subject.predicate:
                # Neighbors are predicates
                for pred in self.get_neighbors(subject):
                    pred_name = pred.name
                    for obj in self.get_neighbors(pred):
                        obj_name = obj.name
                        adjacencies[pred_name].append((vertex_to_id[subject.name],
                                                       vertex_to_id[obj.name]))

        adj_sparse = []
        for rel in adjacencies:
            print('Constructing adjacency matrix for {} ({} edges...)'.format(rel, len(adjacencies[rel])))
            row, col = np.transpose(adjacencies[rel])

            data = np.ones(len(row), dtype=np.int8)

            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape,
                                dtype=np.int8)

            adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape,
                                       dtype=np.int8)

            # TODO: Create csr matrix for each adjacency matrix
            adj_sparse.append(adj)
            adj_sparse.append(adj_transp)

        labels = sp.lil_matrix((adj_shape[0], len(set(train_labels))))
        label_dict = {lab: i for i, lab in enumerate(set(train_labels))}
        labeled_nodes_idx = []
        
        train_idx = []
        train_names = []

        for i, (inst, label) in enumerate(zip(train_inst, train_labels)):
            label_idx = label_dict[label]
            labeled_nodes_idx.append(vertex_to_id[inst.toPython()])
            labels[labeled_nodes_idx[-1], label_idx] = 1
            train_idx.append(vertex_to_id[inst.toPython()])
            train_names.append(inst)
        
        test_idx = []
        test_names = []

        for i, (inst, label) in enumerate(zip(test_inst, test_labels)):
            label_idx = label_dict[label]
            labeled_nodes_idx.append(vertex_to_id[inst.toPython()])
            labels[labeled_nodes_idx[-1], label_idx] = 1
            test_idx.append(vertex_to_id[inst.toPython()])
            test_names.append(inst)

        features = sp.identity(adj_shape[0], format='csr')
        relations_dict = {rel: i for i, rel in enumerate(adjacencies.keys())}

        return adj_sparse, features, labels, labeled_nodes_idx, train_idx, test_idx, relations_dict, train_names, test_names



def txt_to_kgs(data_dir, name, node_label='symbol', edge_label='valence', node_attr_names=['chem', 'charge', 'x', 'y']):
    # TODO: Create one large node that represents the entity, and connect everything to it.
    
    #  sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id).
    adjacency = open('{}/{}_A.txt'.format(data_dir, name), 'r').readlines()

    # labels for the edges in DS_A_sparse.txt
    if '{}_edge_labels.txt'.format(name) in os.listdir(data_dir):
        edge_labels = open('{}/{}_edge_labels.txt'.format(data_dir, name), 'r').readlines()
    else:
        edge_labels = ['0']*len(adjacency)

    # class labels for all graphs in the data set, the value in the i-th line is the class label of the graph with graph_id i
    graph_labels = open('{}/{}_graph_labels.txt'.format(data_dir, name), 'r').readlines()

    # column vector of node labels, the value in the i-th line corresponds to the node with node_id i
    node_labels = open('{}/{}_node_labels.txt'.format(data_dir, name), 'r').readlines()

    # column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i
    graph_ids = open('{}/{}_graph_indicator.txt'.format(data_dir, name), 'r').readlines()

    if '{}_node_attributes.txt'.format(name) in os.listdir(data_dir):
        node_attributes = open('{}/{}_node_attributes.txt'.format(data_dir, name), 'r').readlines()
    else:
        node_attributes = ['']*len(node_labels)

    # First, we iterate over the graph_indicator, node_labels and node_attribute files
    kgs = {}

    vertex_id_to_kg = {}
    id_to_vertex = {}

    for i, (graph_id, node_label, node_attr) in enumerate(zip(graph_ids, node_labels, node_attributes)):
        graph_id = int(graph_id.strip())

        if graph_id not in kgs:
            kgs[graph_id] = KnowledgeGraph()
            root = Vertex('graph_{}'.format(graph_id))
            kgs[graph_id].add_vertex(root)
            kgs[graph_id].root = root

        vertex_id_to_kg[i + 1] = kgs[graph_id]

        entity = Vertex(str(i + 1))
        label_pred = Vertex('label', predicate=True)
        label_vertex = Vertex(node_label.strip())
        has_pred = Vertex('has', predicate=True)
        id_to_vertex[i + 1] = entity

        kgs[graph_id].add_vertex(entity)
        kgs[graph_id].add_vertex(has_pred)
        kgs[graph_id].add_vertex(label_pred)
        
        kgs[graph_id].add_edge(kgs[graph_id].root, has_pred)
        kgs[graph_id].add_edge(has_pred, entity)
        kgs[graph_id].add_edge(entity, label_pred)
        kgs[graph_id].add_edge(label_pred, label_vertex)

        # Now let's add an edge for each attribute
        if len(node_attr) > 0:
            for attr, attr_name in zip(node_attr.split(', '), node_attr_names):
                attr_vertex = Vertex(attr.strip())
                attr_pred_vertex = Vertex(attr_name, predicate=True)

                kgs[graph_id].add_vertex(attr_vertex)
                kgs[graph_id].add_vertex(attr_pred_vertex)

                kgs[graph_id].add_edge(entity, attr_pred_vertex)
                kgs[graph_id].add_edge(attr_pred_vertex, attr_vertex)

    labels = {}
    for i, label in enumerate(graph_labels):
        labels[i + 1] = label

    for i, (adj_row, edge_label) in enumerate(zip(adjacency, edge_labels)):
        _from, _to = map(int, adj_row.split(', '))

        _from_vertex = id_to_vertex[_from]
        _to_vertex = id_to_vertex[_to]

        pred_vertex = Vertex(edge_label.strip(), predicate=True)

        kg = vertex_id_to_kg[_from]

        kg.add_vertex(pred_vertex)
        kg.add_edge(_from_vertex, pred_vertex)
        kg.add_edge(pred_vertex, _to_vertex)

    large_kg = KnowledgeGraph()
    for _id, kg in kgs.items():
        large_kg.vertices = large_kg.vertices.union(kg.vertices)
        for v in kg.transition_matrix:
            large_kg.transition_matrix[v] = large_kg.transition_matrix[v].union(kg.transition_matrix[v])

        for k in kg.name_to_vertex:
            large_kg.name_to_vertex[k] = kg.name_to_vertex[k]

    # Provide 1 large kg and a list of instances/URIs/vertices
    return large_kg, kgs, labels




    
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