from collections import defaultdict
from hashlib import md5
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WLWalker(RandomWalker):
    """Defines the Weisfeler-Lehman walking strategy.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        wl_iterations: The Weisfeiler Lehman's iteration.
            Defaults to 4.
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    wl_iterations: int = attr.ib(
        default=4, validator=attr.validators.instance_of(int)
    )

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=False)

    def _create_label(self, kg: KG, vertex: Vertex, n: int):
        """Creates a label.

        kg: The Knowledge Graph.

            The graph from which the neighborhoods are extracted for the
            provided instances.
        vertex: The vertex.
        n:  The position.

        """
        neighbor_names = [
            self._label_map[neighbor][n - 1]
            for neighbor in kg.get_neighbors(vertex, reverse=True)
        ]
        suffix = "-".join(sorted(set(map(str, neighbor_names))))
        return self._label_map[vertex][n - 1] + "-" + suffix

    def _weisfeiler_lehman(self, kg: KG) -> None:
        """Performs Weisfeiler-Lehman relabeling of the vertices.

        Note:
            You can create a `graph.KnowledgeGraph` object from an
            `rdflib.Graph` object by using a converter method.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.

        """
        self._label_map: DefaultDict[Any, Any] = defaultdict(dict)
        self._inv_label_map: DefaultDict[Any, Any] = defaultdict(dict)

        for vertex in kg._vertices:
            self._label_map[vertex][0] = str(vertex)
            self._inv_label_map[vertex][0] = str(vertex)

        for n in range(1, self.wl_iterations + 1):
            for vertex in kg._vertices:
                self._label_map[vertex][n] = str(
                    md5(self._create_label(kg, vertex, n).encode()).digest()
                )

        for vertex in kg._vertices:
            for k, v in self._label_map[vertex].items():
                self._inv_label_map[vertex][v] = k

    # Tuple[Tuple[str, ...], ...]
    def _extract(
        self, kg: KG, instance: Vertex
    ) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        canonical_walks: Set[Tuple[str, ...]] = set()
        walks = self.extract_walks(kg, instance)
        for walk in walks:
            kg.get_hops(walk[-1])

        self._weisfeiler_lehman(kg)

        for n in range(self.wl_iterations + 1):
            for walk in walks:
                canonical_walk: List[str] = []
                for i, hop in enumerate(walk):
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(hop.name)
                    else:
                        canonical_walk.append(self._label_map[hop][n])
                canonical_walks.add(tuple(canonical_walk))
        return {instance.name: tuple(canonical_walks)}
