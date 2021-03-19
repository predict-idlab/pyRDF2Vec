import asyncio
import operator
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
from urllib import parse

import aiohttp
import attr
import numpy as np
import requests
from cachetools import Cache, TTLCache, cachedmethod
from cachetools.keys import hashkey
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

from pyrdf2vec.typings import Literal, Response


@attr.s
class Connector(ABC):
    """Base class of the connectors."""

    endpoint: str = attr.ib(
        validator=attr.validators.instance_of(str),
    )
    """The endpoint to execute the queries."""

    cache: Cache = attr.ib(
        kw_only=True,
        factory=lambda: TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.optional(attr.validators.instance_of(Cache)),
    )
    """The policy and size cache to use."""

    _headers: Dict[str, str] = attr.ib(
        init=False,
        repr=False,
        default={
            "Accept": "application/sparql-results+json",
        },
    )
    """The HTTP headers to use."""

    _asession = attr.ib(init=False, default=None)
    """The aiohttp session to use for asynchrone requests."""

    _session = attr.ib(
        init=False,
        factory=lambda: requests.Session(),
    )
    """The requests session to use for synchrone requests."""

    async def close(self) -> None:
        """Closes the aiohttp session."""
        await self._asession.close()

    @abstractmethod
    def fetch(self, query: str):
        """Fetchs the result of a query.

        Args:
            query: The query to fetch the result

        Returns:
            The generated dictionary from the ['results']['bindings'] json.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This must be implemented!")


@attr.s
class SPARQLConnector(Connector):
    """Represents a SPARQL connector."""

    def __attrs_post_init__(self):
        adapter = HTTPAdapter(
            Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
            )
        )
        self._session.mount("http", adapter)
        self._session.mount("https", adapter)

    async def afetch(self, queries: List[str]) -> List[List[Response]]:
        """Fetchs the result of SPARQL queries asynchronously.

        Args:
            queries: The queries.

        Returns:
            The response of the queries.

        """
        if self._asession is None:
            self._asession = aiohttp.ClientSession(raise_for_status=True)
        return await asyncio.gather(*(self._fetch(query) for query in queries))

    async def _fetch(self, query) -> Response:
        """Fetchs the result of a SPARQL query with the aiohttp session.

        This function is useful only to avoid unnecessarily filling the fetch
        function's cache with values that can never be retrieved because of a
        different session that uses a coroutine.

        Args:
            query: The query to fetch the result.

        Returns:
            The response of the query in a JSON format.

        """
        url = f"{self.endpoint}/query?query={parse.quote(query)}"
        async with self._asession.get(url, headers=self._headers) as res:
            res = await res.json()
            return res["results"]["bindings"]

    @cachedmethod(operator.attrgetter("cache"), key=partial(hashkey, "fetch"))
    def fetch(self, query: str) -> Response:
        """Fetchs the result of a SPARQL query.

        Args:
            query: The query to fetch the result.

        Returns:
            The response of the query in a JSON format.

        """
        url = f"{self.endpoint}/query?query={parse.quote(query)}"
        res = self._session.get(url, headers=self._headers).json()
        return res["results"]["bindings"]

    def get_query(self, entity: str, preds: Optional[List[str]] = None) -> str:
        """Gets the SPARQL query for an entity.

        Args:
            entity: The entity to get the SPARQL query.
            preds: The predicate chain to fetch a literal
                Defaults to None.

        Returns:
            The SPARQL query for the given entity.

        """
        query = f"SELECT ?p ?o WHERE {{ <{entity}> ?p "
        if preds:
            query = f"SELECT ?o WHERE {{ <{entity}> <{preds[0]}> "
            for i in range(1, len(preds)):
                query += f"?o{i} . ?o{i} <{preds[i]}> "
        query += "?o . }"
        return query

    def res2literals(self, res) -> Union[Literal, Tuple[Literal, ...]]:
        """Converts a JSON response server to literal(s).

        Args:
            res: The JSON response.

        Returns:
            The literal(s).

        """
        if len(res) == 0:
            return np.NaN
        literals = []
        for literal in res:
            try:
                literals.append(float(literal["o"]["value"]))
            except Exception:
                literals.append(literal["o"]["value"])
        if len(literals) > 1:
            return tuple(literals)
        return literals[0]
