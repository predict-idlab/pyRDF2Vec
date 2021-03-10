import asyncio
from typing import Dict, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker


@attr.s
class WalkletWalker(RandomWalker):
    """Walker that transforms walks into walklets, which are walks of length
    two consisting of the root of the original walk and one of the hop.

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

    """

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
                [instance] + kg.get_pliterals(instance.name, pred)
                for pred in kg.literals
            ]

        canonical_walks: Set[Tuple[str, ...]] = set()
        for walk in walks:
            if len(walk) == 1:
                canonical_walks.add((walk[0].name,))
            for i in range(1, len(walk)):
                if self.with_reverse:
                    canonical_walks.add((walk[i].name, walk[0].name))
                else:
                    canonical_walks.add((walk[0].name, walk[i].name))
        return {instance.name: [tuple(canonical_walks), literals]}
