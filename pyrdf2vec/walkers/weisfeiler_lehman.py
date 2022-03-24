from collections import defaultdict
from hashlib import md5
from typing import DefaultDict, Dict, List, Set, Union

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import Entities, EntityWalks, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WLWalker(RandomWalker):
    """Weisfeiler-Lehman walking strategy which relabels the nodes of the
    extracted random walks, providing additional information about the entity
    representations only when a maximum number of walks is not specified.

    Attributes:
        _inv_label_map: Stores the mapping of the inverse labels.
            Defaults to defaultdict.
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise.
            Defaults to False.
        _label_map: Stores the mapping of the inverse labels.
            Defaults to defaultdict.
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        md5_bytes: The number of bytes to keep after hashing objects in
            MD5. Hasher allows to reduce the memory occupied by a long text. If
            md5_bytes is None, no hash is applied.
            Defaults to 8.
        random_state: The random state to use to keep random determinism with
            the walking strategy.
            Defaults to None.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        wl_iterations: The Weisfeiler Lehman's iteration.
            Defaults to 4.

    """

    wl_iterations = attr.ib(
        kw_only=True,
        default=4,
        type=int,
        validator=attr.validators.instance_of(int),
    )

    _is_support_remote = attr.ib(
        init=False, repr=False, type=bool, default=False
    )

    _inv_label_map = attr.ib(
        init=False,
        repr=False,
        type=DefaultDict["Vertex", Dict[Union[str, int], Union[str, int]]],
        factory=lambda: defaultdict(dict),
    )
    _label_map = attr.ib(
        init=False,
        repr=False,
        type=DefaultDict["Vertex", Dict[int, str]],
        factory=lambda: defaultdict(dict),
    )

    def _create_label(self, kg: KG, vertex: Vertex, n: int) -> str:
        """Creates a label according to a vertex and its neighbors.

        kg: The Knowledge Graph.

            The graph from which the neighborhoods are extracted for the
            provided entities.
        vertex: The vertex to get its neighbors to create the suffix.
        n:  The index of the neighbor

        Returns:
            the label created for the vertex.

        """
        if len(self._label_map) == 0:
            self._weisfeiler_lehman(kg)

        suffix = "-".join(
            sorted(
                set(
                    [
                        self._label_map[neighbor][n - 1]
                        for neighbor in kg.get_neighbors(
                            vertex, is_reverse=True
                        )
                    ]
                )
            )
        )
        return f"{self._label_map[vertex][n - 1]}-{suffix}"

    def _weisfeiler_lehman(self, kg: KG) -> None:
        """Performs Weisfeiler-Lehman relabeling of the vertices.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.

        """
        for vertex in kg._vertices:
            self._label_map[vertex][0] = vertex.name
            self._inv_label_map[vertex][0] = vertex.name

        for n in range(1, self.wl_iterations + 1):
            for vertex in kg._vertices:
                if self.md5_bytes:
                    self._label_map[vertex][n] = str(
                        md5(
                            self._create_label(kg, vertex, n).encode()
                        ).digest()[: self.md5_bytes]
                    )
                else:
                    self._label_map[vertex][n] = str(
                        self._create_label(kg, vertex, n)
                    )

        for vertex in kg._vertices:
            for k, v in self._label_map[vertex].items():
                self._inv_label_map[vertex][v] = k

    def extract(
        self, kg: KG, entities: Entities, verbose: int = 0
    ) -> List[List[SWalk]]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.
            entities: The entities to be extracted from the Knowledge Graph.
            verbose: The verbosity level.
                0: does not display anything;
                1: display of the progress of extraction and training of walks;
                2: debugging.
                Defaults to 0.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        self._weisfeiler_lehman(kg)
        return super().extract(kg, entities, verbose)

    def _map_wl(self, entity: Vertex, pos: int, n: int) -> str:
        """Maps certain vertices to MD5 hashes to save memory. For entities of
        interest (provided by the user to the extract function) and predicates,
        the string representation is kept.

        Args:
            entity: The entity to be mapped.
            pos: The position of the entity in the walk.
            n: The iteration number of the WL algorithm.

        Returns:
            A hash (string) or original string representation.

        """
        if entity.name in self._entities or pos % 2 == 1:
            return entity.name
        else:
            return self._label_map[entity][n]

    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts random walks for an entity based on a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            A dictionary having the entity as key and a list of tuples as value
            corresponding to the extracted walks.

        """
        canonical_walks: Set[SWalk] = set()
        for n in range(self.wl_iterations + 1):
            for walk in self.extract_walks(kg, entity):
                canonical_walk: List[str] = [
                    self._map_wl(vertex, i, n) for i, vertex in enumerate(walk)
                ]
                canonical_walks.add(tuple(canonical_walk))
        return {entity.name: list(canonical_walks)}
