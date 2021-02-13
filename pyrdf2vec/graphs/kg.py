import itertools
import json
import operator
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from urllib import parse

import attr
import rdflib
import requests
from cachetools import Cache, TTLCache, cachedmethod
from requests.adapters import HTTPAdapter

try:
    import asyncio

    import aiohttp

    is_aiohttp = True
except ModuleNotFoundError:
    is_aiohttp = False


@attr.s(eq=False, frozen=True, slots=True)
class Vertex:
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

    name: str = attr.ib(validator=attr.validators.instance_of(str))
    predicate: bool = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    vprev: Optional["Vertex"] = attr.ib(default=None)
    vnext: Optional["Vertex"] = attr.ib(default=None)

    _counter = itertools.count()
    id: int = attr.ib(init=False, factory=lambda: next(Vertex._counter))

    def __eq__(self, other) -> bool:
        """Defines behavior for the equality operator, ==.

        Args:
            other: The other vertex to test the equality.

        Returns:
            True if the hash of the vertices are equal. False otherwise.

        """
        if other is None:
            return False
        elif self.predicate:
            return (self.id, self.vprev, self.vnext, self.name) == (
                other.id,
                other.vprev,
                other.vnext,
                other.name,
            )
        return self.name == other.name

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

    def __lt__(self, other: "Vertex") -> bool:
        return self.name < other.name


@attr.s
class KG:
    """Represents a Knowledge Graph.

    Attributes:
        location: The location of the file to load.
            Defaults to None.
        file_type: The type of the file to load.
            Defaults to None.
        skip_predicates: The label predicates to skip from the KG.
            Defaults to None.
        is_remote: True if the file is in a SPARQL endpoint server.
            False otherwise.
            Defaults to False.
        cache: The cache policy to use for remote Knowledge Graphs.
            Defaults to TTLCache(maxsize=1024, ttl=1200)

    """

    location: Optional[str] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    file_type: Optional[str] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    skip_predicates: Set[str] = attr.ib(
        default=set(),
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )
    is_mul_req: bool = attr.ib(
        kw_only=True, default=True, validator=attr.validators.instance_of(bool)
    )
    is_remote: bool = attr.ib(
        kw_only=True,
        default=False,
        validator=attr.validators.instance_of(bool),
    )
    cache: Cache = attr.ib(
        kw_only=True,
        default=TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.instance_of(Cache),
    )

    _inv_transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, default=defaultdict(set)
    )
    _is_support_remote: bool = attr.ib(init=False, repr=False, default=False)
    _transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, default=defaultdict(set)
    )
    _entities: Set[Vertex] = attr.ib(init=False, repr=False, default=set())
    _vertices: Set[Vertex] = attr.ib(init=False, repr=False, default=set())

    @is_remote.validator
    def _check_is_remote(self, attribute, value):
        if value is True and not is_valid_url(self.location):
            raise ValueError(
                f"'location' must be a valid URL (got {self.location})"
            )
        elif value is False and self.location is not None:
            if not os.path.exists(self.location) or not os.path.isfile(
                self.location
            ):
                raise FileNotFoundError(
                    f"'location' must be a valid file (got {self.location})"
                )

    def __attrs_post_init__(self):
        if self.is_remote:
            self.session = requests.Session()
            self.session.mount("http://", HTTPAdapter())
            self._headers = {"Accept": "application/sparql-results+json"}
        elif self.location is not None:
            self.read_file()

    def add_edge(self, v1: Vertex, v2: Vertex) -> bool:
        """Adds a uni-directional edge.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        Returns:
            True if the edge has been added, False otherwise.

        """
        self._transition_matrix[v1].add(v2)
        self._inv_transition_matrix[v2].add(v1)
        return True

    def add_vertex(self, vertex: Vertex) -> bool:
        """Adds a vertex to the Knowledge Graph.

        Args:
            vertex: The vertex to add.

        Returns:
            True if the vertex has been added, False otherwise.

        """
        self._vertices.add(vertex)
        if not vertex.predicate:
            self._entities.add(vertex)
        return True

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
    def fetch_hops(self, vertex: Vertex):
        """Fetchs the hops of the vertex.

        Args:
            vertex: The vertex to get the hops.

        """
        if not vertex.name.startswith("http://"):
            return []
        query = parse.quote(
            "SELECT ?p ?o WHERE { <" + str(vertex) + "> ?p ?o . }"
        )

        url = self.location + "/query?query=" + query

        res = self.session.get(url, headers=self._headers)
        if res.status_code != 200:
            res.raise_for_status()

        hops = []
        for result in json.loads(res.text)["results"]["bindings"]:
            pred, obj = result["p"]["value"], result["o"]["value"]
            if obj not in self.skip_predicates:
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
        self, session, vertices: List[str]
    ) -> Dict[str, List[Tuple[Any, Any]]]:
        """Fetchs the hops of the vertices according to a session.

        Args:
            session: The session.
            vertices: The vertices to get the hops.

        Returns:
            The hops of the vertices.

        """
        urls = [
            self.location
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
                if obj not in self.skip_predicates:
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

    def _get_rhops(self, vertex: Vertex) -> List[Tuple[str, str]]:
        """Gets the hops for a vertex.

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        return [
            ((pred, obj))
            for pred in self._transition_matrix[vertex]
            for obj in self._transition_matrix[pred]
            if len(self._transition_matrix[pred]) != 0
        ]

    def _get_shops(self, vertex: Vertex) -> List[Tuple[str, str]]:
        if self.is_mul_req and vertex.name in self.entity_hops:
            return self.entity_hops[vertex.name]
        return self.fetch_hops(vertex)

    def get_hops(self, vertex: Vertex) -> List[Tuple[str, str]]:
        """Returns the hops of a vertex.

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if self.is_remote:
            return self._get_shops(vertex)
        return self._get_rhops(vertex)

    def get_neighbors(
        self, vertex: Vertex, reverse: bool = False
    ) -> Set[Vertex]:
        """Gets the reverse neighbors of a vertex.

        Args:
            vertex: The vertex.

        Returns:
            The reverse neighbors of a vertex.

        """
        if reverse:
            return self._inv_transition_matrix[vertex]
        return self._transition_matrix[vertex]

    def read_file(self) -> None:
        """Parses a file with rdflib."""
        assert self.location is not None
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
            if p not in self.skip_predicates:
                s_v = Vertex(str(s))
                o_v = Vertex(str(o))
                p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
                self.add_vertex(s_v)
                self.add_vertex(p_v)
                self.add_vertex(o_v)
                self.add_edge(s_v, p_v)
                self.add_edge(p_v, o_v)

    def remove_edge(self, v1: Vertex, v2: Vertex) -> bool:
        """Removes the edge (v1 -> v2) if present.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        Returns:
            True if the edge has been removed, False otherwise.

        """
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)
            self._inv_transition_matrix[v2].remove(v1)
            return True
        return False


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
