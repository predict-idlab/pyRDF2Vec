import json
import operator
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from urllib import parse

import attr
import requests
from cachetools import Cache, TTLCache, cachedmethod

from pyrdf2vec.graphs.vertex import Vertex

try:
    import asyncio

    import aiohttp

    is_aiohttp = True
except ModuleNotFoundError:
    is_aiohttp = False


@attr.s
class Connector(ABC):
    """Base class for the connectors."""

    @abstractmethod
    def query(self, query: str):
        """Gets the result of a query.

        Args:
            query: The query.

        Returns:
            The dictionary generated from the ['results']['bindings'] json.

        """
        raise NotImplementedError("This must be implemented!")


@attr.s
class SPARQLConnector(Connector):
    """Represents a SPARQL connector.

    Attributes:

        endpoint: The SPARQL endpoint server.
        is_mul_req: If True allows to bundle SPARQL requests.
            Defaults to True.
        cache: The cache policy to use for remote Knowledge Graphs.
            Defaults to TTLCache(maxsize=1024, ttl=1200)

    """

    endpoint: str = attr.ib(
        validator=attr.validators.instance_of(str),
    )
    is_mul_req: bool = attr.ib(
        kw_only=True,
        default=True,
        validator=attr.validators.instance_of(bool),
    )
    cache: Cache = attr.ib(
        kw_only=True,
        default=TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.instance_of(Cache),
    )

    _entity_hops: Dict[str, List[Tuple[Any, Any]]] = attr.ib(
        init=False, repr=False, default={}
    )
    _headers: Dict[str, str] = attr.ib(
        init=False,
        repr=False,
        default={"Accept": "application/sparql-results+json"},
    )

    @cachedmethod(operator.attrgetter("cache"))
    def fetch_hops(self, vertex: Vertex):
        """Fetchs the hops of the vertex.

        Args:
            vertex: The vertex to get the hops.

        Returns:
            The hops of the vertex.

        """
        if not vertex.name.startswith("http://"):
            return []
        elif vertex.name in self._entity_hops:
            return self._entity_hops[vertex.name]
        return self.query(
            "SELECT ?p ?o WHERE { <" + vertex.name + "> ?p ?o . }"
        )

    async def fetch(self, url: str, session):
        """Fetchs the hops for a URL

        Args:
            session: The session.
            url: The URL that contains the query.

        Returns:
            The response of the URL.

        """
        async with session.get(
            url, headers=self._headers, raise_for_status=True
        ) as response:
            return await response.text()

    async def _fill_hops(self, kg, vertices: List[Vertex]) -> None:
        """Fills the entity hops.

        Args:
            vertices: The vertices to get the hops.

        """
        if is_aiohttp:
            async with aiohttp.ClientSession() as session:
                urls = [
                    self.endpoint
                    + "/query?query="
                    + parse.quote(
                        "SELECT ?p ?o WHERE { <" + vertex.name + "> ?p ?o . }"
                    )
                    for vertex in vertices
                    if vertex.name.startswith("http://")
                ]
                for vertex, res in zip(
                    vertices,
                    await asyncio.gather(
                        *(self.fetch(url, session) for url in urls)
                    ),
                ):
                    hops = []
                    for result in json.loads(res)["results"]["bindings"]:
                        obj = Vertex(result["o"]["value"])
                        pred = Vertex(
                            result["p"]["value"],
                            predicate=True,
                            vprev=vertex,
                            vnext=obj,
                        )
                        if kg.add_walk(vertex, pred, obj):
                            hops.append((pred, obj))
                        self._entity_hops.update({str(vertex): hops})
        else:
            warnings.warn(
                "Sending multiple SPARQL requests simultaneously is only "
                + "available from Python >= 3.7",
                category=RuntimeWarning,
                stacklevel=2,
            )

    def query(self, query: str):
        """Gets the result of a query for a SPARQL endpoint server.

        Args:
            query: The SPARQL query.

        Returns:
            The dictionary generated from the ['results']['bindings'] json.

        """
        url = self.endpoint + "/query?query=" + parse.quote(query)
        with requests.Session() as session:
            res = session.get(url, headers=self._headers).text
        return json.loads(res)["results"]["bindings"]
