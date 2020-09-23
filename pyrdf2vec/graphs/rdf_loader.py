from collections import defaultdict
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import rdflib

from pyrdf2vec.graphs import KG, Vertex


class RDFLoader(KG):
    """Represents a Knowledge Graph from RDFLib."""

    def __init__(self, file_name, label_predicates, file_type=None):
        self.file_name = file_name
        self.file_type = file_type
        self.label_predicates = label_predicates

        self._inv_transition_matrix = defaultdict(set)
        self._transition_matrix = defaultdict(set)
        self._vertices = set()
        self._entities = set()

        self._read_file()

    def _read_file(self) -> None:
        """Parses a file with rdflib"""
        kg = rdflib.Graph()
        try:
            if self.file_type is None:
                kg.parse(self.file_name, format=self.file_name.split(".")[-1])
            else:
                kg.parse(self.file_name, self.file_type)
        except Exception:
            kg.parse(self.file_name)

        for (s, p, o) in kg:
            if p not in self.label_predicates:
                s_v = Vertex(str(s))
                o_v = Vertex(str(o))
                p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
                self.add_vertex(s_v)
                self.add_vertex(p_v)
                self.add_vertex(o_v)
                self.add_edge(s_v, p_v)
                self.add_edge(p_v, o_v)

    def add_vertex(self, vertex: Vertex) -> None:
        """Adds a vertex to the Knowledge Graph.

        Args:
            vertex: The vertex

        """
        self._vertices.add(vertex)
        if not vertex.predicate:
            self._entities.add(vertex)

    def add_edge(self, v1: Vertex, v2: Vertex) -> None:
        """Adds a uni-directional edge.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        """
        self._transition_matrix[v1].add(v2)
        self._inv_transition_matrix[v2].add(v1)

    def get_hops(self, vertex: str) -> List[Tuple[str, str]]:
        """Returns a hop (vertex -> predicate -> object)

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if isinstance(vertex, str):
            vertex = Vertex(vertex)  # type: ignore
        hops = []
        predicates = self._transition_matrix[vertex]
        for pred in predicates:
            assert len(self._transition_matrix[pred]) == 1
            for obj in self._transition_matrix[pred]:
                hops.append((pred, obj))
        return hops

    def get_inv_neighbors(self, vertex: Vertex) -> Set[Vertex]:
        """Gets the reverse neighbors of a vertex.

        Args:
            vertex: The vertex.

        Returns:
            The reverse neighbors of a vertex.

        """
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        return self._inv_transition_matrix[vertex]

    def get_neighbors(self, vertex: Vertex) -> Set[Vertex]:
        """Gets the neighbors of a vertex.

        Args:
            vertex: The vertex.

        Returns:
            The neighbors of a vertex.

        """
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        return self._transition_matrix[vertex]

    def remove_edge(self, v1: str, v2: str):
        """Removes the edge (v1 -> v2) if present.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        """
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)

    def visualise(self) -> None:
        """Visualises the Knowledge Graph."""
        nx_graph = nx.DiGraph()

        for v in self._vertices:
            if not v.predicate:
                name = v.name.split("/")[-1]
                nx_graph.add_node(name, name=name, pred=v.predicate)

        for v in self._vertices:
            if not v.predicate:
                v_name = v.name.split("/")[-1]
                # Neighbors are predicates
                for pred in self.get_neighbors(v):
                    pred_name = pred.name.split("/")[-1]
                    for obj in self.get_neighbors(pred):
                        obj_name = obj.name.split("/")[-1]
                        nx_graph.add_edge(v_name, obj_name, name=pred_name)

        plt.figure(figsize=(10, 10))
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        names = nx.get_edge_attributes(nx_graph, "name")
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names)
