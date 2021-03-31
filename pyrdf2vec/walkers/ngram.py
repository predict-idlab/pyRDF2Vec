import itertools
from typing import Dict, List, Optional, Set, Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class NGramWalker(RandomWalker):
    """Walker that relabels the N-grams in random walks to define a mapping
    from one-to-many.
    The intuition behind this is that the predecessors of a node that two
    different walks have in common can be different.
    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        _n_gram_map: Stores the mapping of N-gram.
            Defaults to {}.
        grams: The N-gram to relabel.
            Defaults to 3.
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        random_state: The random state to use to keep random determinism with
            the walking strategy.
            Defaults to None.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        wildcards: The wildcards to be used to match sub-sequences with small
            differences to be mapped onto the same label.
            Defaults to None.
        with_reverse: True to extracts children's and parents' walks from the
            root, creating (max_walks * max_walks) more walks of 2 * depth,
            False otherwise.
            Defaults to False.
    """

    grams = attr.ib(
        kw_only=True,
        default=3,
        type=int,
        validator=attr.validators.instance_of(int),
    )

    wildcards = attr.ib(
        kw_only=True,
        default=None,
        type=Optional[list],
        validator=attr.validators.optional(attr.validators.instance_of(list)),
    )

    _n_gram_map = attr.ib(
        init=False, repr=False, type=Dict[Tuple, str], factory=dict
    )

    def _take_n_grams(self, walk: Walk) -> List[str]:
        """Takes the N-Grams.

        Args:
            walk: The walk.

        Returns:
            The N-Grams.

        """
        n_gram_walk = []
        for i, hop in enumerate(walk):
            if i == 0 or i % 2 == 1 or i < self.grams:
                n_gram_walk.append(hop.name)
            else:
                n_gram = tuple(
                    walk[j].name
                    for j in range(max(0, i - (self.grams - 1)), i + 1)
                )
                if n_gram not in self._n_gram_map:
                    self._n_gram_map[n_gram] = str(len(self._n_gram_map))
                n_gram_walk.append(self._n_gram_map[n_gram])
        return n_gram_walk

    def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided entities which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, instance):
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
        return {instance.name: list(canonical_walks)}
