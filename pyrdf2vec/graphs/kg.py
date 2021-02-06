import asyncio
import itertools
import json
import operator
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from urllib import parse

import matplotlib.pyplot as plt
import networkx as nx
import rdflib
import requests
from cachetools import Cache, TTLCache, cachedmethod
from requests.adapters import HTTPAdapter

try:
    import aiohttp

    is_aiohttp = False
except ModuleNotFoundError:
    is_aiohttp = True
    print("KOKOKO")


class Vertex(object):
    """Represents a vertex in a Knowledge Graph.

    Attributes:
        name: The name of the vertex.
        predicate: The predicate of the vertex.
            Defaults to False.
        vprev: The previous Vertex.
            Defaults to None
        vnext: The next Vertex.
            Defaults to None.

    """

    vertex_counter = itertools.count()

    def __init__(self, name, predicate=False, vprev=None, vnext=None):
        self.name = name
        self.predicate = predicate
        self.vprev = vprev
        self.vnext = vnext
        self.id = next(self.vertex_counter)

    def __eq__(self, other) -> bool:
        """Defines behavior for the equality operator, ==.

        Args:
            other: The other vertex to test the equality.

        Returns:
            True if the hash of the vertices are equal. False otherwise.

        """
        if other is None:
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        """Defines behavior for when hash() is called on a vertex.

        Returns:
            The identifier and name of the vertex, as well as its previous
            and next neighbor if the vertex has a predicate. The hash of
            the name of the vertex otherwise.

        """
        if self.predicate:
            return hash((self.id, self.vprev, self.vnext, self.name))
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name


class KG:
    """Represents a Knowledge Graph.

    Attributes:
        location: The location of the file to load.
            Defaults to None.
        file_type: The type of the file to load.
            Defaults to None.
        label_predicates: The label predicates.
            Defaults to None.
        is_remote: True if the file is in a SPARQL endpoint server.
            False otherwise.
            Defaults to False.
        cache: The cache policy to use for remote Knowledge Graphs.
            Defaults to TTLCache(maxsize=1024, ttl=1200)

    """

    def __init__(
        self,
        location: str,
        file_type: Optional[str] = None,
        label_predicates=None,
        is_remote: bool = False,
        cache: Cache = TTLCache(maxsize=1024, ttl=1200),
    ):
        self.cache = cache
        self.file_type = file_type
        if label_predicates is None:
            self.label_predicates = set()
        else:
            self.label_predicates = set(label_predicates)
        self.is_remote = is_remote
        self.location = location

        self._inv_transition_matrix: DefaultDict[Any, Any] = defaultdict(set)
        self._transition_matrix: DefaultDict[Any, Any] = defaultdict(set)
        self._entities: Set[Vertex] = set()
        self._vertices: Set[Vertex] = set()

        if is_remote:
            if is_valid_url(location):
                self.session = requests.Session()
                self.session.mount("http://", HTTPAdapter())
                self._headers = {"Accept": "application/sparql-results+json"}
                self.endpoint = location
            else:
                raise ValueError(f"Invalid URL: {location}")
        else:
            self.read_file()

    async def fetch(self, session, url: str):
        """Fetchs the hops for a URL

        Args:
            session: The session.
            url: The URL that contains the query.

        Returns:
            The hops of a URL.

        """
        async with session.get(
            url,
            headers=self._headers,
            raise_for_status=True,
        ) as response:
            return await response.text()

    @cachedmethod(operator.attrgetter("cache"))
    def fetch_hops(self, vertex: str):
        """Fetchs the hops of the vertex.

        Args:
            vertex: The vertex to get the hops.

        """
        if not vertex.startswith("http://"):
            return []
        query = parse.quote(
            "SELECT ?p ?o WHERE { <" + str(vertex) + "> ?p ?o . }"
        )

        url = self.endpoint + "/query?query=" + query

        res = self.session.get(url, headers=self._headers)
        if res.status_code != 200:
            res.raise_for_status()
        res = res.text

        hops = []
        for result in json.loads(res)["results"]["bindings"]:
            pred, obj = result["p"]["value"], result["o"]["value"]
            if obj not in self.label_predicates:
                hops.append((pred, obj))
                s_v = Vertex(str(vertex))
                o_v = Vertex(str(obj))
                p_v = Vertex(str(pred), predicate=True, vprev=s_v, vnext=o_v)
                self.add_vertex(s_v)
                self.add_vertex(o_v)
                self.add_vertex(p_v)
                self.add_edge(s_v, p_v)
                self.add_edge(p_v, o_v)
        return hops

    async def fetch_ehops(
        self, session: requests.Session, vertices: List[str]
    ) -> Dict[str, List[Tuple[Any, Any]]]:
        """Fetchs the hops of the vertices according to a session.

        Args:
            session: The session.
            vertices: The vertices to get the hops.

        Returns:
            The hops of the vertices.

        """
        urls = [
            self.endpoint
            + "/query?query="
            + parse.quote(
                "SELECT ?p ?o WHERE { <" + str(vertex) + "> ?p ?o . }"
            )
            for vertex in vertices
        ]

        entity_hops = {}
        for vertex, res in zip(
            vertices,
            await asyncio.gather(*(self.fetch(session, url) for url in urls)),
        ):
            hops = []
            for result in json.loads(res)["results"]["bindings"]:
                pred, obj = result["p"]["value"], result["o"]["value"]
                if obj not in self.label_predicates:
                    hops.append((pred, obj))
                    s_v = Vertex(str(vertex))
                    o_v = Vertex(str(obj))
                    p_v = Vertex(
                        str(pred), predicate=True, vprev=s_v, vnext=o_v
                    )
                    self.add_vertex(s_v)
                    self.add_vertex(o_v)
                    self.add_vertex(p_v)
                    self.add_edge(s_v, p_v)
                    self.add_edge(p_v, o_v)
            entity_hops.update({str(vertex): hops})
        return entity_hops

    async def _fill_entity_hops(self, vertices):
        """Fills the entity hops.

        Args:
            vertices: The vertices to get the hops.

        """
        if is_aiohttp:
            async with aiohttp.ClientSession() as session:
                self.entity_hops = await self.fetch_ehops(session, vertices)
        else:
            self.entity_hops = {}

    def _get_rhops(self, vertex: str) -> List[Tuple[str, str]]:
        """Gets the hops for a vertex.

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if isinstance(vertex, rdflib.term.URIRef):
            vertex = Vertex(str(vertex))  # type: ignore
        elif isinstance(vertex, str):
            vertex = Vertex(vertex)  # type: ignore
        hops = []

        predicates = self._transition_matrix[vertex]
        for pred in predicates:
            assert len(self._transition_matrix[pred]) == 1
            for obj in self._transition_matrix[pred]:
                hops.append((pred, obj))
        return hops

    def _get_shops(self, vertex: str) -> List[Tuple[str, str]]:
        if str(vertex) in self.entity_hops:
            return self.entity_hops[str(vertex)]
        return self.fetch_hops(vertex)

    def add_edge(self, v1: Vertex, v2: Vertex) -> None:
        """Adds a uni-directional edge.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        """
        self._transition_matrix[v1].add(v2)
        self._inv_transition_matrix[v2].add(v1)

    def add_vertex(self, vertex: Vertex) -> None:
        """Adds a vertex to the Knowledge Graph.

        Args:
            vertex: The vertex

        """
        self._vertices.add(vertex)
        if not vertex.predicate:
            self._entities.add(vertex)

    def get_hops(self, vertex: str) -> List[Tuple[str, str]]:
        """Returns the hops of a vertex.

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if self.is_remote:
            return self._get_shops(vertex)
        return self._get_rhops(vertex)

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

    def read_file(self) -> None:
        """Parses a file with rdflib."""
        if not os.path.exists(self.location) or not os.path.isfile(
            self.location
        ):
            raise FileNotFoundError(self.location)

        self.graph = rdflib.Graph()
        try:
            if self.file_type is None:
                self.graph.parse(
                    self.location, format=self.location.split(".")[-1]
                )
            else:
                self.graph.parse(self.location, format=self.file_type)
        except Exception:
            self.graph.parse(self.location)

        for (s, p, o) in self.graph:
            if p not in self.label_predicates:
                s_v = Vertex(str(s))
                o_v = Vertex(str(o))
                p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
                self.add_vertex(s_v)
                self.add_vertex(p_v)
                self.add_vertex(o_v)
                self.add_edge(s_v, p_v)
                self.add_edge(p_v, o_v)

    def remove_edge(self, v1: str, v2: str) -> None:
        """Removes the edge (v1 -> v2) if present.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        """
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)
            self._inv_transition_matrix[v2].remove(v1)

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


def is_valid_url(url: str) -> bool:
    """Checks if a URL is valid.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is valid. False otherwise.

    """
    try:
        requests.get(url)
    except Exception:
        return False
    return True
