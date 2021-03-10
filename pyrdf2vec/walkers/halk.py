import asyncio
from collections import defaultdict
from hashlib import md5
from typing import Dict, List, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker


@attr.s
class HALKWalker(RandomWalker):
    """Walker that removes the rare entities from the random walks in order to
    increase the quality of the generated embeddings while decreasing the
    memory usage.

    Args:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        freq_thresholds: The minimum frequency thresholds of a hop to be kept.
            Defaults to [0.001].
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        with_reverse: extracts children's and parents' walks from the root,
            creating (max_walks * max_walks) more walks of 2 * depth.
            Defaults to False.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    freq_thresholds: List[float] = attr.ib(
        kw_only=True,
        default=[0.001],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(float),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    async def _extract(
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
        literals = []
        if not kg.is_mul_req:
            walks = await asyncio.create_task(self.extract_walks(kg, instance))
            literals = await asyncio.create_task(
                kg.get_literals(instance.name)
            )
        else:
            walks = await self.extract_walks(kg, instance)
            literals = [
                [instance] + kg.get_pliterals(instance, pred)
                for pred in kg.literals
            ]

        canonical_walks: Set[Tuple[str, ...]] = set()
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
                canonical_walk: List[str] = []
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
        return {instance.name: [tuple(canonical_walks), literals]}
