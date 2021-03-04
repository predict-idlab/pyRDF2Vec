from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Set, Tuple

import attr
import rdflib
from cachetools import Cache, TTLCache, cachedmethod

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
        is_mul_req: If True allows to bundle SPARQL requests.
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
        default=set(),
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
    is_mul_req: bool = attr.ib(
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
    _entities: Set[Vertex] = attr.ib(init=False, repr=False, default=set())
    _vertices: Set[Vertex] = attr.ib(init=False, repr=False, default=set())

    def __attrs_post_init__(self):
        if self.location is not None:
            self._is_remote = self.location.startswith(
                "http://"
            ) or self.location.startswith("https://")

            if self._is_remote is True:
                self.connector = SPARQLConnector(
                    self.location, is_mul_req=self.is_mul_req, cache=self.cache
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

    def is_valid_pchain(self, entity, pchain):
        if len(pchain) == 0:
            return False

        i = len(pchain) - 1

        while len(pchain) > 1:
            subjs = self.get_neighbors(Vertex(pchain[i]), is_reverse=True)
            if len(subjs) > 0:
                if pchain[i - 1] in [sub.vprev.name for sub in subjs]:
                    pchain.pop()
                else:
                    return False
            else:
                return False
            i -= 1
        return _transition_matrix(Vertex(pchain[i])).name == entity

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
            hops = []
            for result in self.connector.fetch_hops(vertex):
                obj = Vertex(result["o"]["value"])
                pred = Vertex(
                    result["p"]["value"],
                    predicate=True,
                    vprev=vertex,
                    vnext=obj,
                )
                if self.add_walk(vertex, pred, obj):
                    hops.append((pred, obj))
            return hops

        if is_reverse:
            return [
                (pred, obj)
                for pred in self._inv_transition_matrix[vertex]
                for obj in self._inv_transition_matrix[pred]
                if len(self._inv_transition_matrix[pred]) != 0
            ]
        return [
            (pred, obj)
            for pred in self._transition_matrix[vertex]
            for obj in self._transition_matrix[pred]
            if len(self._transition_matrix[pred]) != 0
        ]

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
