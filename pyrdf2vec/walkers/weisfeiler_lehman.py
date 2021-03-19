from collections import defaultdict
from hashlib import md5
from typing import DefaultDict, Dict, List, Set, Union

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import Entities, EntityWalks, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WLWalker(RandomWalker):
    """Defines the Weisfeler-Lehman walking strategy."""

    wl_iterations: int = attr.ib(
        kw_only=True, default=4, validator=attr.validators.instance_of(int)
    )
    """The Weisfeiler Lehman's iteration."""

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=False)
    """True if the walking strategy can be used with a remote Knowledge Graph,
    False Otherwise.
    """

    _inv_label_map: DefaultDict[
        Vertex, Dict[Union[str, int], Union[str, int]]
    ] = attr.ib(init=False, repr=False, factory=lambda: defaultdict(dict))
    """Stores the mapping of the inverse labels."""

    _label_map: DefaultDict[Vertex, Dict[int, str]] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )
    """Stores the mapping of the labels."""

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
                self._label_map[vertex][n] = str(
                    md5(self._create_label(kg, vertex, n).encode()).digest()
                )

        for vertex in kg._vertices:
            for k, v in self._label_map[vertex].items():
                self._inv_label_map[vertex][v] = k

    def extract(
        self, kg: KG, entities: Entities, verbose: int = 0
    ) -> List[str]:
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

    def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided entities which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided entities.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        canonical_walks: Set[SWalk] = set()
        for n in range(self.wl_iterations + 1):
            for walk in self.extract_walks(kg, instance):
                canonical_walk: List[str] = []
                for i, hop in enumerate(walk):
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(hop.name)
                    else:
                        canonical_walk.append(self._label_map[hop][n])
                canonical_walks.add(tuple(canonical_walk))
        return {instance.name: tuple(canonical_walks)}
