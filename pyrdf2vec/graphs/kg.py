import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from urllib import parse

import attr
import rdflib
import requests

from pyrdf2vec.connectors import SPARQLConnector
from pyrdf2vec.graphs.vertex import Vertex


@attr.s
class KG:
    """Represents a Knowledge Graph.

    Attributes:
        location: The location of the file to load.
            Defaults to None.
        skip_predicates: The label predicates to skip from the KG.
            Defaults to None.
        fmt: Used if format can not be determined from source.
            Defaults to None.

    """

    location: Optional[str] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )
    skip_predicates: Set[str] = attr.ib(
        default=set(),
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )
    fmt: Optional[str] = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    _inv_transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, default=defaultdict(set)
    )
    _is_remote: bool = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    _transition_matrix: DefaultDict[Any, Any] = attr.ib(
        init=False, repr=False, default=defaultdict(set)
    )
    _entities: Set[Vertex] = attr.ib(init=False, repr=False, default=set())
    _vertices: Set[Vertex] = attr.ib(init=False, repr=False, default=set())

    @location.validator
    def _check_location(self, attribute, value):
        if value is not None:
            self._is_remote = value.startswith("http://") or value.startswith(
                "https://"
            )
            if self._is_remote and not is_valid_url(value):
                raise ValueError(
                    f"'location' must be a valid URL (got {value})"
                )
            elif not self._is_remote and value is not None:
                if not os.path.exists(value) or not os.path.isfile(value):
                    raise FileNotFoundError(
                        f"'location' must be a valid file (got {value})"
                    )

    def __attrs_post_init__(self):
        if self._is_remote is True:
            self.connector = SPARQLConnector(self.location)
        elif self.location is not None:
            for (sub, pred, obj) in rdflib.Graph().parse(
                self.location, format=self.fmt
            ):
                sub = Vertex(str(sub))
                obj = Vertex(str(obj))
                self.add_walk(
                    sub,
                    Vertex(str(pred), predicate=True, vprev=sub, vnext=obj),
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

    def add_walk(self, sub: Vertex, pred: Vertex, obj: Vertex) -> bool:
        """Adds a walk.

        Args:
            sub: The vertex of the subject.
            pred: The vertex of the predicate.
            obj: The vertex of the object.

        Returns:
            True if the walk has been added, False otherwise.

        """
        if pred.name not in self.skip_predicates:
            self.add_vertex(sub)
            self.add_vertex(pred)
            self.add_vertex(obj)
            self.add_edge(sub, pred)
            self.add_edge(pred, obj)
            return True
        return False

    def get_hops(self, vertex: Vertex) -> List[Tuple[Vertex, Vertex]]:
        """Returns the hops of a vertex.

        Args:
            vertex: The name of the vertex to get the hops.

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
        return [
            (pred, obj)
            for pred in self._transition_matrix[vertex]
            for obj in self._transition_matrix[pred]
            if len(self._transition_matrix[pred]) != 0
        ]

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
