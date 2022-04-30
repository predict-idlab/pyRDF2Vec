import asyncio
import operator
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

import attr
import numpy as np
import rdflib
from cachetools import Cache, TTLCache, cachedmethod
from tqdm import tqdm

from pyrdf2vec.connectors import SPARQLConnector
from pyrdf2vec.graphs.vertex import Vertex
from pyrdf2vec.typings import Entities, Hop, Literal, Literals
from pyrdf2vec.utils.validation import _check_location


@attr.s
class KG:
    """Represents a Knowledge Graph.

    Attributes:
        _entity_hops: Caches the results of asynchronous requests.
            Defaults to {}.
        _entities: Stores the entities.
            Defaults to set.
        _is_remote: True if the Knowledge Graph is in remote, False otherwise.
            Defaults to False.
        _vertices: Stores the vertices.
            Defaults to set.
        _inv_transition_matrix: Contains the parents of vertices.
            Defaults to defaultdict.
        _transition_matrix: Contains the children of vertices.
            Defaults to defaultdict.
        cache: The policy and size cache to use.
            Defaults to TTLCache(maxsize=1024, ttl=1200).
        connector: The connector to use.
            Defaults to SPARQLConnector.
        fmt: The format of the file.
            It should be used only if the format can not be determined from
            source.
            Defaults to None.
        literals: The predicate chains to get the literals.
            Defaults to [].
        location: The location of the file to load.
            Defaults to None.
        mul_req: True to allow bundling of SPARQL queries, False otherwise.
            This attribute accelerates the extraction of walks for remote
            Knowledge Graphs. Beware that this may violate the policy of some
            SPARQL endpoint server.
            Defaults to False.
        skip_predicates: The label predicates to skip from the KG.
            Defaults to set.
        skip_verify: To skip or not the verification of existing entities in a
            Knowledge Graph. Its deactivation can improve HTTP latency for KG
            remotes.
            Defaults to False.

    """

    location = attr.ib(  # type: ignore
        default=None,
        type=Optional[str],
        validator=[
            attr.validators.optional(attr.validators.instance_of(str)),
            _check_location,
        ],
    )

    skip_predicates = attr.ib(
        factory=set,
        type=Set[str],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )

    literals = attr.ib(  # type: ignore
        factory=list,
        type=List[List[str]],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(List)
        ),
    )

    fmt = attr.ib(
        kw_only=True,
        type=Optional[str],
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    mul_req = attr.ib(
        kw_only=True,
        type=bool,
        default=False,
        validator=attr.validators.instance_of(bool),
    )

    skip_verify = attr.ib(
        kw_only=True,
        type=bool,
        default=False,
        validator=attr.validators.instance_of(bool),
    )

    cache = attr.ib(
        kw_only=True,
        type=Cache,
        factory=lambda: TTLCache(maxsize=1024, ttl=1200),
        validator=attr.validators.optional(attr.validators.instance_of(Cache)),
    )

    connector = attr.ib(
        init=False, default=None, type=SPARQLConnector, repr=False
    )

    _is_remote = attr.ib(
        default=False, type=bool, validator=attr.validators.instance_of(bool)
    )

    _inv_transition_matrix = attr.ib(
        init=False,
        repr=False,
        type=DefaultDict[Vertex, Set[Vertex]],
        factory=lambda: defaultdict(set),
    )
    _transition_matrix = attr.ib(
        init=False,
        repr=False,
        type=DefaultDict[Vertex, Set[Vertex]],
        factory=lambda: defaultdict(set),
    )

    _entity_hops = attr.ib(
        init=False, repr=False, type=Dict[str, List[Hop]], factory=dict
    )

    _entities = attr.ib(init=False, type=Set[Vertex], repr=False, factory=set)
    _vertices = attr.ib(init=False, type=Set[Vertex], repr=False, factory=set)

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
        """Adds a walk to the Knowledge Graph.

        Args:
            subj: The vertex of the subject.
            pred: The vertex of the predicate.
            obj: The vertex of the object.

        Returns:
            True if the walk has been added to the Knowledge Graph, False
            otherwise.

        """
        if pred.name not in self.skip_predicates:
            self.add_vertex(subj)
            self.add_vertex(pred)
            self.add_vertex(obj)
            self.add_edge(subj, pred)
            self.add_edge(pred, obj)
            return True
        return False

    def fetch_hops(self, vertex: Vertex) -> List[Hop]:
        """Fetchs the hops of the vertex from a SPARQL endpoint server and
        add the hops for this vertex in a cache dictionary.

        Args:
            vertex: The vertex to get the hops.

        Returns:
            The hops of the vertex.

        """
        hops: List[Hop] = []
        if not self._is_remote:
            return hops
        elif vertex.name in self._entity_hops:
            return self._entity_hops[vertex.name]
        elif vertex.name.startswith("http://") or vertex.name.startswith(
            "https://"
        ):
            res = self.connector.fetch(self.connector.get_query(vertex.name))
            hops = self._res2hops(vertex, res["results"]["bindings"])
        return hops

    def get_hops(self, vertex: Vertex, is_reverse: bool = False) -> List[Hop]:
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
        return self._get_hops(vertex, is_reverse)

    def get_literals(self, entities: Entities, verbose: int = 0) -> Literals:
        """Gets the literals for one or more entities for all the predicates
        chain.

        Args:
            entities: The entity or entities to get the literals.
            verbose: The verbosity level.
                0: does not display anything;
                1: display of the progress of extraction and training of walks;
                2: debugging.
                Defaults to 0.
        Returns:
            The list that contains literals for each entity.

        """
        if len(self.literals) == 0:
            return []

        if self._is_remote:
            queries = [
                self.connector.get_query(entity, pchain)
                for entity in tqdm(
                    entities, disable=True if verbose == 0 else False
                )
                for pchain in self.literals
                if len(pchain) > 0
            ]

            if self.mul_req:
                responses = asyncio.run(self.connector.afetch(queries))
            else:
                responses = [self.connector.fetch(query) for query in queries]

            literals_responses = [
                self.connector.res2literals(
                    res["results"]["bindings"]  # type: ignore
                )
                for res in responses
            ]
            return [
                literals_responses[
                    len(self.literals) * i : len(self.literals) * (i + 1) :
                ]
                for i in range(len(entities))
            ]
        entity_literals = []
        for entity in tqdm(entities, disable=True if verbose == 0 else False):
            entity_literal = [
                self.get_pliterals(entity, pred) for pred in self.literals
            ]
            entity_literals.append(self._cast_literals(entity_literal))
        return entity_literals

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

    def get_pliterals(self, entity: str, preds: List[str]) -> List[str]:
        """Gets the literals for an entity and a local KG based on a chain of
        predicates.

        Args:
            entity: The entity.
            preds: The chain of predicates.

        Returns:
            The literals for the given entity.

        """
        frontier = {entity}
        for p in preds:
            new_frontier = set()
            for node in frontier:
                for pred, obj in self.get_hops(Vertex(node)):
                    if pred.name == p:
                        new_frontier.add(obj.name)
            frontier = new_frontier
        return list(frontier)

    def is_exist(self, entities: Entities) -> bool:
        """Checks that all provided entities exists in the Knowledge Graph.

        Args:
            entities: The entities to check the existence

        Returns:
            True if all the entities exists, False otherwise.

        """
        if self._is_remote:
            queries = [
                f"ASK WHERE {{ <{entity}> ?p ?o . }}" for entity in entities
            ]
            if self.mul_req:
                responses = [
                    res["boolean"]  # type: ignore
                    for res in asyncio.run(self.connector.afetch(queries))
                ]
            else:
                responses = [self.connector.fetch(query) for query in queries]
                responses = [res["boolean"] for res in responses]
            return False not in responses
        return all([Vertex(entity) in self._vertices for entity in entities])

    def remove_edge(self, v1: Vertex, v2: Vertex) -> bool:
        """Removes the edge (v1 -> v2) if present.

        Args:
            v1: The first vertex.
            v2: The second vertex.

        Returns:
            True if the edge has been removed, False otherwise.

        """
        if self._is_remote:
            raise ValueError(
                "Can remove an edge only for a local Knowledge Graph."
            )

        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)
            self._inv_transition_matrix[v2].remove(v1)
            return True
        return False

    def _cast_literals(
        self, entity_literals: List[List[str]]
    ) -> List[Union[Literal, Tuple[Literal, ...]]]:
        """Converts the raw literals of entity according to their real types.

        Args:
            entity_literals: The raw literals.

        Returns:
            The literals with their type for the given entity.

        """
        literals: List[Union[Literal, Tuple[Literal, ...]]] = []
        for literal in entity_literals:
            if len(literal) == 0:
                literals.append(np.NaN)
            else:
                casted_value: Union[Literal, List[Literal]] = []
                for value in literal:
                    try:
                        casted_value.append(float(value))  # type:ignore
                    except Exception:
                        casted_value.append(value)  # type:ignore
                if len(casted_value) > 1:  # type:ignore
                    literals.append(tuple(casted_value))  # type:ignore
                else:
                    literals += casted_value  # type:ignore
        return literals

    def _fill_hops(self, entities: Entities) -> None:
        """Fills the entity hops in cache.

        Args:
            vertices: The vertices to get the hops.

        """
        queries = [self.connector.get_query(entity) for entity in entities]
        for entity, res in zip(
            entities,
            asyncio.run(self.connector.afetch(queries)),
        ):
            hops = self._res2hops(
                Vertex(entity), res["results"]["bindings"]  # type: ignore
            )
            self._entity_hops.update({entity: hops})

    @cachedmethod(operator.attrgetter("cache"))
    def _get_hops(self, vertex: Vertex, is_reverse: bool = False) -> List[Hop]:
        """Returns the hops of a vertex for a local Knowledge Graph.

        Args:
            vertex: The name of the vertex to get the hops.
            is_reverse: If True, this function gets the parent nodes of a
                vertex. Otherwise, get the child nodes for this vertex.
                Defaults to False.

         Returns:
             The hops of a vertex in a (predicate, object) form.

        """
        matrix = self._transition_matrix
        if is_reverse:
            matrix = self._inv_transition_matrix
        return [
            (pred, obj)
            for pred in matrix[vertex]
            for obj in matrix[pred]
            if len(matrix[pred]) != 0
        ]

    def _res2hops(self, vertex: Vertex, res) -> List[Hop]:
        """Converts a JSON response from a SPARQL endpoint server to hops.

        Args:
            vertex: The vertex to get the hops.
            res: The JSON response of the SPARQL endpoint server.

        Returns:
            The hops.

        """
        hops = []
        for value in res:
            obj = Vertex(value["o"]["value"])
            pred = Vertex(
                value["p"]["value"],
                predicate=True,
                vprev=vertex,
                vnext=obj,
            )
            if pred.name not in self.skip_predicates:
                hops.append((pred, obj))
        return hops
