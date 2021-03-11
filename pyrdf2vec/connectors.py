import asyncio
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import parse

import aiohttp
import attr
import numpy as np
import requests
from cachetools import Cache, TTLCache, cachedmethod
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


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
    """Represents a SPARQL connector."""

    endpoint: str = attr.ib(
        validator=attr.validators.instance_of(str),
    )
    cache: Cache = attr.ib(
        kw_only=True,
        default=TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.optional(attr.validators.instance_of(Cache)),
    )
    _headers: Dict[str, str] = attr.ib(
        init=False,
        repr=False,
        default={"Accept": "application/sparql-results+json"},
    )

    def __attrs_post_init__(self):
        self._session = requests.Session()
        adapter = HTTPAdapter(
            Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
            )
        )
        self._session.mount("http", adapter)
        self._session.mount("https", adapter)

    async def afetch(
        self, queries: List[str]
    ) -> List[List[Optional[Dict[Any, Any]]]]:
        """Fetchs the result of queries asynchronously.

        Args:
            queries: The queries.

        Returns:
            The result of the queries.

        """
        urls = [
            f"{self.endpoint}/query?query={parse.quote(query)}"
            for query in queries
        ]
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(
                *(self.fetch(url, session) for url in urls)
            )

    @cachedmethod(operator.attrgetter("cache"))
    async def fetch(self, url: str, session) -> List[Dict[Any, Any]]:
        """Fetchs the result of the URL asynchronously.

        Args:
            url: The URL.
            session: The session.

        Returns:
            The response of the URL in a JSON format.

        """
        async with session.get(
            url, headers=self._headers, raise_for_status=True
        ) as res:
            res = await res.json()
            return res["results"]["bindings"]

    def get_query(self, entity: str, pchain: Optional[str] = None) -> str:
        """Gets the SPARQL query for an entity.

        Args:
            entity: The entity.
            pchain: The predicate chain.
                Defaults to None.

        Returns:
            The SPARQL query for the given entity.

        """
        query = f"SELECT ?p ?o WHERE {{ <{entity}> ?p "
        if pchain:
            query = f"SELECT ?o WHERE {{ <{entity}> <{pchain[0]}> "
            for i in range(1, len(pchain)):
                query += f"?o{i} . ?o{i} <{pchain[i]}> "
        query += "?o . }"
        return query

    @cachedmethod(operator.attrgetter("cache"))
    def query(self, query: str) -> List[Dict[Any, Any]]:
        """Gets the result of a query for a SPARQL endpoint server.

        Args:
            query: The SPARQL query.

        Returns:
            The dictionary generated from the ['results']['bindings'] json.

        """
        url = f"{self.endpoint}/query?query={parse.quote(query)}"
        res = self._session.get(url, headers=self._headers).json()
        return res["results"]["bindings"]

    def res2literals(self, res) -> Union[str, Tuple[str, ...]]:
        """Converts a JSON response from a SPARQL endpoint server to a literal.

        Args:
            res: The JSON response of the SPARQL endpoint server.

        Returns:
            The literal.

        """
        if len(res) == 0:
            return np.NaN
        literals = []
        for literal in res:
            try:
                literals.append(float(literal["o"]["value"]))
            except:
                literals.append(literal["o"]["value"])
        if len(literals) > 1:
            return tuple(literals)
        return literals[0]
