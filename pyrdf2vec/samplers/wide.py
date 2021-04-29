from collections import defaultdict
from typing import DefaultDict, Tuple

import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class WideSampler(Sampler):

    _pred_degs: DefaultDict[Tuple[str, str], int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    _obj_degs: DefaultDict[Tuple[str, str], int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    _neighbor_counts: DefaultDict[Tuple[str, str], int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    def fit(self, kg: KG) -> None:
        """Since the weights are uniform, this function does nothing.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)

        for vertex in kg._vertices:
            if vertex.predicate:
                self._neighbor_counts[vertex.name] = len(
                    kg.get_neighbors(vertex)
                )
                counter = self._pred_degs
            else:
                self._neighbor_counts[vertex.name] = len(
                    kg.get_neighbors(vertex, is_reverse=True)
                )
                counter = self._obj_degs

            if vertex.name in counter:
                counter[vertex.name] += 1
            else:
                counter[vertex.name] = 1

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        """
        if not (self._pred_degs and self._obj_degs and self._neighbor_counts):
            raise ValueError(
                "You must call the `fit(kg)` method before get the weight of"
                + " a hop."
            )
        return (
            self._neighbor_counts[hop[0].name]
            + self._neighbor_counts[hop[1].name]
        ) * ((self._pred_degs[hop[0].name] + self._obj_degs[hop[1].name]) / 2)
