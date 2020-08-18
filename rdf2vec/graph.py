from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


class Vertex(object):
    """Represents a vertex in a knowledge graph.

    Attributes:
        name (string): The vertex name.
        predicate (bool): The predicate.
            Defaults to False.
        _from: The previous vertex.
            Defaults to None.
        _to: The next vertex.
            Defaults to None.

    """

    vertex_counter = 0
    
    def __init__(self, name, predicate=False, _from=None, _to=None):
        self.name = name
        self.predicate = predicate
        self._from = _from
        self._to = _to

        self.id = Vertex.vertex_counter
        Vertex.vertex_counter += 1
        
    def __eq__(self, other):
        """Defines behavior for the equality operator, ==.

        Args:
            other (Vertex): The other vertex to test the equality.

        Returns:
            bool: True if the hash of the vertices are equal. False otherwise.

        """
        if other is None:
            return False
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        """Defines behavior for when hash() is called on a vertex.

        Returns:
            int: The identifier and name of the vertex, as well as its previous
                and next neighbor if the vertex has a predicate. The hash of
                the name of the vertex otherwise.

        """
        if self.predicate:
            return hash((self.id, self._from, self._to, self.name))
        else:
            return hash(self.name)

    def __lt__(self, other):
        """Defines behavior for the less-than operator, <.

        Args:
            other (Vertex): The other vertex to test the equality.

        Returns:
            bool: True if the name of the first vertex is less than equal to the
                name of the other vertex.

        """
        return self.name < other.name


class KnowledgeGraph(object):
    """Represents a knowledge graph. """

    def __init__(self):
        self._vertices = set()
        self._transition_matrix = defaultdict(set)
        self._inv_transition_matrix = defaultdict(set)
        
    def add_vertex(self, vertex):
        """Adds a vertex to the knowledge graph.

        Args:
            vertex (Vertex): The vertex

        """
        if vertex.predicate:
            self._vertices.add(vertex)
        else:
            self._vertices.add(vertex)

    def add_edge(self, v1, v2):
        """Adds a uni-directional edge.

        Args:
            v1 (string): The name of the first vertex.
            v2 (string): The name of the second vertex.

        """
        self._transition_matrix[v1].add(v2)
        self._inv_transition_matrix[v2].add(v1)
        
    def remove_edge(self, v1, v2):
        """Removes the edge (v1 -> v2) if present.

        Args:
            v1 (string): The name of the first vertex.
            v2 (string): The name of the second vertex.

        """
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)

    def get_neighbors(self, vertex):
        """Gets the neighbors of a vertex.

        Args:
            vertex (Vertex): The vertex.

        Returns:
            array-like: The neighbors of a vertex.

        """
        return self._transition_matrix[vertex]

    def get_inv_neighbors(self, vertex):
        """Gets the reverse neighbors of a vertex.

        Args:
            vertex (Vertex): The vertex.

        Returns:
            array-like: The reverse neighbors of a vertex.

        """
        return self._inv_transition_matrix[vertex]
    
    def visualise(self):
        """Visualises the knowledge graph."""
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
        
        plt.figure(figsize=(10, 10))
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        names = nx.get_edge_attributes(nx_graph, 'name')
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names)
