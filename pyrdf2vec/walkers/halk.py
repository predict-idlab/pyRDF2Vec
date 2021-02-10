from collections import defaultdict
from hashlib import md5
from typing import Any, Dict, List, Tuple

import attr
import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker


@attr.s
class HalkWalker(RandomWalker):
    """Defines the Hierarchical Walking (HALK) strategy.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        freq_thresholds: The thresholds frequencies.
            Defaults to [0.001].
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        seed: The seed to use to ensure ensure random determinism to generate
            the same walks for entities.
            Defaults to None.

    """

    freq_thresholds: List[float] = attr.ib(default=[0.001])
    hop_prob: float = attr.ib(default=0.1)
    resolution: int = attr.ib(default=1)

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
        walks = self.extract_walks(kg, str(instance))

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
