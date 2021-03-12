import asyncio
import operator
from collections import defaultdict
from functools import partial
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import attr
import numpy as np
import rdflib
from cachetools import Cache, TTLCache, cachedmethod
from cachetools.keys import hashkey
from tqdm import tqdm

from pyrdf2vec.connectors import SPARQLConnector
from pyrdf2vec.graphs.vertex import Vertex
from pyrdf2vec.utils.validation import _check_location


@attr.s
class KG:
    """Represents a Knowledge Graph.

    Args:
        location: The location of the file to load.
            Defaults to None.
        skip_predicates: The label predicates to skip from the KG.
            Defaults to None.
        fmt: Used if format can not be determined from source.
            Defaults to None.
        mul_req: If True allows to bundle SPARQL requests.
            Defaults to True.

    """

    location: Optional[str] = attr.ib(  # type: ignore
        default=None,
        validator=[
            attr.validators.optional(attr.validators.instance_of(str)),
            _check_location,
        ],
    )
    skip_predicates: Set[str] = attr.ib(
        factory=set,
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )
    literals: Optional[List[List]] = attr.ib(
        factory=list,
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(List)
        ),
    )
    fmt: Optional[str] = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    mul_req: bool = attr.ib(
        kw_only=True,
        default=True,
        validator=attr.validators.instance_of(bool),
    )
    cache: Cache = attr.ib(
        kw_only=True,
        default=TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.optional(attr.validators.instance_of(Cache)),
    )

    _inv_transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(set)
    )
    _is_remote: bool = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    _transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(set)
    )
    _entities: Set[Vertex] = attr.ib(init=False, repr=False, factory=set)
    _vertices: Set[Vertex] = attr.ib(init=False, repr=False, factory=set)

    def __attrs_post_init__(self):
        if self.location is not None:
            self._is_remote = self.location.startswith(
                "http://"
            ) or self.location.startswith("https://")

            if self._is_remote is True:
                self.connector = SPARQLConnector(
                    self.location, cache=self.cache
                )
            elif self.location is not None:
                for subj, pred, obj in rdflib.Graph().parse(
                    self.location, format=self.fmt
                ):
                    subj = Vertex(str(subj))
                    obj = Vertex(str(obj))
                    self.add_walk(
                        subj,
                        Vertex(
                            str(pred), predicate=True, vprev=subj, vnext=obj
                        ),
                        obj,
                    )

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

    def add_walk(self, subj: Vertex, pred: Vertex, obj: Vertex) -> bool:
        """Adds a walk.

        Args:
            subj: The vertex of the subject.
            pred: The vertex of the predicate.
            obj: The vertex of the object.

        Returns:
            True if the walk has been added, False otherwise.

        """
        if pred.name not in self.skip_predicates:
            self.add_vertex(subj)
            self.add_vertex(pred)
            self.add_vertex(obj)
            self.add_edge(subj, pred)
            self.add_edge(pred, obj)
            return True
        return False

    @cachedmethod(
        operator.attrgetter("cache"), key=partial(hashkey, "fetch_hops")
    )
    def fetch_hops(self, vertex: Vertex) -> List[Tuple[Vertex, Vertex]]:
        """Fetchs the hops of the vertex from a SPARQL endpoint server and
        add the hops for this vertex in a cache dictionary.

        Args:
            vertex: The vertex to get the hops.

        Returns:
            The hops of the vertex.

        """
        hops = []
        if not self._is_remote:
            return hops
        elif vertex.name.startswith("http://") or vertex.name.startswith(
            "https://"
        ):
            res = self.connector.query(self.connector.get_query(vertex.name))
            hops = self._res2hops(vertex, res)
        return hops

    @cachedmethod(
        operator.attrgetter("cache"), key=partial(hashkey, "get_hops")
    )
    def get_hops(
        self, vertex: Vertex, is_reverse: bool = False
    ) -> List[Tuple[Vertex, Vertex]]:
        """Returns the hops of a vertex.

        Args:
            vertex: The name of the vertex to get the hops.
            is_reverse: If True, this function gets the parent nodes of a
                vertex. Otherwise, get the child nodes for this vertex.
                Defaults to False.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if self._is_remote:
            return self.fetch_hops(vertex)

        matrix = self._transition_matrix
        if is_reverse:
            matrix = self._inv_transition_matrix
        return [
            (pred, obj)
            for pred in matrix[vertex]
            for obj in matrix[pred]
            if len(matrix[pred]) != 0
        ]

    def get_literals(
        self,
        entities: Union[str, List[Union[str, Tuple[str, ...]]]],
        verbose: int = 0,
    ):
        """Gets the literals for one or more entities for all the predicates
        chain.

        Args:
            entities: The entity or entities to get the literals.

        Returns:
            The literals.

        """
        if len(self.literals) == 0:
            return []

        if self._is_remote:
            e = [entities] if isinstance(entities, str) else entities
            queries = [
                self.connector.get_query(entity, pchain)
                for entity in tqdm(e)
                for pchain in self.literals
                if len(pchain) > 0
            ]
            responses = asyncio.run(self.connector.afetch(queries))
            literals_responses = [
                self.connector.res2literals(res) for res in responses
            ]
            if isinstance(entities, str):
                return literals_responses
            return [
                literals_responses[
                    len(self.literals) * i : len(self.literals) * (i + 1) :
                ]
                for i in range(len(entities))
            ]

        t = []
        for entity in tqdm(entities):
            tmp2 = []
            for pred in self.literals:
                tmp2 += self.get_pliterals(entity, pred)
            t.append(tmp2)

        lit = []
        for literal in t:
            tmp = []
            if len(literal) == 0:
                tmp += ["nan"]
            else:
                tmp2 = []
                for value in literal:
                    try:
                        tmp2.append(float(value))
                    except:
                        tmp2.append(value)
                if len(tmp2) > 1:
                    tmp += tuple(tmp2)
                else:
                    tmp += tmp2
            lit.append(tmp)
        return lit

    def get_neighbors(
        self, vertex: Vertex, is_reverse: bool = False
    ) -> Set[Vertex]:
        """Gets the children or parents neighbors of a vertex.

        Args:
            vertex: The vertex.
            is_reverse: If True, this function gets the parent nodes of a
                vertex. Otherwise, get the child nodes for this vertex.
                Defaults to False.

        Returns:
            The children or parents neighbors of a vertex.

        """
        if is_reverse:
            return self._inv_transition_matrix[vertex]
        return self._transition_matrix[vertex]

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

    def get_pliterals(self, entity: str, pchain: str) -> List[Vertex]:
        frontier = {entity}
        for p in pchain:
            new_frontier = set()
            for node in frontier:
                for pred, obj in self.get_hops(Vertex(node)):
                    if pred.name == p:
                        new_frontier.add(obj.name)
            frontier = new_frontier
        return list(frontier)

    def _res2hops(self, vertex: Vertex, res) -> List[Tuple[Vertex, Vertex]]:
        hops = []
        for value in res:
            obj = Vertex(value["o"]["value"])
            pred = Vertex(
                value["p"]["value"],
                predicate=True,
                vprev=vertex,
                vnext=obj,
            )
            if self.add_walk(vertex, pred, obj):
                hops.append((pred, obj))
        return hops
