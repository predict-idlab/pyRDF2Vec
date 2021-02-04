from collections import defaultdict
from hashlib import md5
from typing import Any, Dict, List, Tuple

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.walkers import RandomWalker


class HalkWalker(RandomWalker):
    """Defines the Hierarchical Walking (HALK) strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        freq_thresholds: The thresholds frequencies.
            Defaults to [0.001].
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = UniformSampler(),
        freq_thresholds: List[float] = [0.001],
        n_jobs: int = 1,
    ):
        super().__init__(depth, walks_per_graph, sampler, n_jobs)
        self.freq_thresholds = freq_thresholds

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

        freq = defaultdict(set)
        for i in range(len(walks)):
            for hop in walks[i]:  # type: ignore
                freq[str(hop)].add(i)

        for freq_threshold in self.freq_thresholds:
            uniformative_hops = set()
            for hop in freq:
                if len(freq[hop]) / len(walks) < freq_threshold:
                    uniformative_hops.add(hop)

            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):  # type: ignore
                    if i == 0:
                        canonical_walk.append(str(hop))
                    else:
                        if str(hop) not in uniformative_hops:
                            canonical_walk.append(
                                str(md5(str(hop).encode()).digest()[:8])
                            )
                canonical_walks.add(tuple(canonical_walk))
        return {instance: tuple(canonical_walks)}
