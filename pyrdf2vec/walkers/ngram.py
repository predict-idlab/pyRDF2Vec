import asyncio
import itertools
from typing import Dict, List, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker


@attr.s
class NGramWalker(RandomWalker):
    """Walker that relabels the N-grams in random walks to define a mapping
    from one-to-many.

    The intuition behind this is that the predecessors of a node that two
    different walks have in common can be different.

    Args:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of process to use for multiprocessing.
            Defaults to 1.
        with_reverse: extracts children's and parents' walks from the root,
            creating (max_walks * max_walks) more walks of 2 * depth.
            Defaults to False.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.
        grams: The N-grams to relabel.
            Defaults to 3.
        wildcards: The wildcards to be used to match sub-sequences with small
            differences to be mapped onto the same label.
            Defaults to None.

    """

    grams: int = attr.ib(
        kw_only=True, default=3, validator=attr.validators.instance_of(int)
    )
    wildcards: list = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )
    _n_gram_map: Dict[Tuple, str] = attr.ib(
        init=False, repr=False, factory=dict
    )

    def _take_n_grams(self, walks: Tuple[Vertex, ...]) -> List[str]:
        """Takes the N-Grams.

        Args:
            walks: The walks.

        Returns:
            The N-Grams.

        """
        n_gram_walk: List[str] = []
        for i, hop in enumerate(walks):
            if i == 0 or i % 2 == 1 or i < self.grams:
                n_gram_walk.append(hop.name)
            else:
                n_gram = tuple(
                    walks[j].name
                    for j in range(max(0, i - (self.grams - 1)), i + 1)
                )
                if n_gram not in self._n_gram_map:
                    self._n_gram_map[n_gram] = str(len(self._n_gram_map))
                n_gram_walk.append(self._n_gram_map[n_gram])
        return n_gram_walk

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
        walks = await asyncio.create_task(self.extract_walks(kg, instance))
        if not kg.mul_req:
            literals = await asyncio.create_task(
                kg.get_literals(instance.name)
            )

        canonical_walks: Set[Tuple[str, ...]] = set()
        for walk in walks:
            canonical_walks.add(tuple(self._take_n_grams(walk)))

            # Introduce wild-cards and re-calculate n-grams
            if self.wildcards is None:
                continue

            for wildcard in self.wildcards:
                for idx in itertools.combinations(
                    range(1, len(walk)), wildcard
                ):
                    new_walk = list(walk).copy()
                    for ix in idx:
                        new_walk[ix] = Vertex("*")
                    canonical_walks.add(
                        tuple(self._take_n_grams(new_walk))  # type: ignore
                    )
        return {instance.name: [tuple(canonical_walks), literals]}
