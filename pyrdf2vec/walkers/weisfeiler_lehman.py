from collections import defaultdict
from hashlib import md5
from typing import Any, DefaultDict, Dict, Optional, Tuple

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class WeisfeilerLehmanWalker(RandomWalker):
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

    """

    def __init__(
        self,
        depth: int,
        max_walks: Optional[int] = None,
        sampler: Sampler = UniformSampler(),
        wl_iterations: int = 4,
        n_jobs: int = 1,
    ):
        super().__init__(depth, max_walks, sampler, n_jobs, False)
        self.wl_iterations = wl_iterations
        self.is_support_remote = False

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
            for neighbor in kg.get_inv_neighbors(vertex)
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

        for v in kg._vertices:
            self._label_map[v][0] = str(v)
            self._inv_label_map[str(v)][0] = v

        for n in range(1, self.wl_iterations + 1):
            for vertex in kg._vertices:
                s_n = self._create_label(kg, vertex, n)
                self._label_map[vertex][n] = str(md5(s_n.encode()).digest())

        for vertex in kg._vertices:
            for key, val in self._label_map[vertex].items():
                self._inv_label_map[vertex][val] = key

    def _extract(
        self, kg: KG, instance: rdflib.URIRef
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
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
        canonical_walks = set()
        walks = self.extract_random_walks(kg, str(instance))
        for walk in walks:
            kg.get_hops(walk[-1])  # type: ignore

        self._weisfeiler_lehman(kg)
        for n in range(self.wl_iterations + 1):
            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):  # type: ignore
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(str(hop))
                    else:
                        canonical_walk.append(self._label_map[hop][n])
                canonical_walks.add(tuple(canonical_walk))
        return {instance: tuple(canonical_walks)}
