from collections import defaultdict
from hashlib import md5
from typing import List, Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class HALKWalker(RandomWalker):
    """Walker that removes the rare entities from the random walks in order to
    increase the quality of the generated embeddings while decreasing the
    memory usage.

    """

    freq_thresholds: List[float] = attr.ib(
        kw_only=True,
        factory=lambda: [0.001],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(float),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )
    """The minimum frequency thresholds of a hop to be kept."""

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
        walks = self.extract_walks(kg, instance)

        canonical_walks: Set[SWalk] = set()
        hop_to_freq = defaultdict(set)
        for i in range(len(walks)):
            for hop in walks[i]:
                hop_to_freq[hop].add(i)

        for freq_threshold in self.freq_thresholds:
            uniformative_hops = set()
            for hop in hop_to_freq:
                if len(hop_to_freq[hop]) / len(walks) < freq_threshold:
                    uniformative_hops.add(hop)

            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):
                    if i == 0:
                        canonical_walk.append(hop.name)
                    elif hop.name not in uniformative_hops:
                        # Use a hash to reduce memory usage of long texts
                        # by using 8 bytes per hop, except for the first
                        # hop and odd hops (predicates).
                        canonical_walk.append(
                            str(md5(hop.name.encode()).digest()[:8])
                        )
                canonical_walks.add(tuple(canonical_walk))
        return {instance.name: tuple(canonical_walks)}
