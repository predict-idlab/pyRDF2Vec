from collections import defaultdict
from typing import DefaultDict

import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class WideSampler(Sampler):
    """Wide sampling node-centric sampling strategy which gives priority to
    walks containing edges with the highest degree of predicates and
    objects. The degree of a predicate and an object being defined by the
    number of predicates and objects present in its neighborhood, but also by
    their number of occurrence in a Knowledge Graph.

    Attributes:
        _is_support_remote: True if the sampling strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to False.
        _random_state: The random state to use to keep random determinism with
            the sampling strategy.
            Defaults to None.
        _vertices_deg: The degree of the vertices.
            Defaults to {}.
        _visited: Tags vertices that appear at the max depth or of which all
            their children are tagged.
            Defaults to set.
        inverse: True if the inverse algorithm must be used, False otherwise.
            Defaults to False.
        split: True if the split algorithm must be used, False otherwise.
            Defaults to False.

    """

    _pred_degs: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    _obj_degs: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    _neighbor_counts: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by couting the number of available
        neighbors for each vertex, but also by counting the number of
        occurrence that a predicate and an object appears in the Knowledge
        Graph.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        for vertex in kg._vertices:
            is_reverse = True if vertex.predicate else False
            counter = self._pred_degs if vertex.predicate else self._obj_degs
            self._neighbor_counts[vertex.name] = len(
                kg.get_neighbors(vertex, is_reverse=is_reverse)
            )

            if vertex.name in counter:
                counter[vertex.name] += 1
            else:
                counter[vertex.name] = 1

    def get_weight(self, hop: Hop) -> float:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if not (self._pred_degs or self._obj_degs or self._neighbor_counts):
            raise ValueError(
                "You must call the `fit(kg)` method before get the weight of"
                + " a hop."
            )
        return (
            self._neighbor_counts[hop[0].name]
            + self._neighbor_counts[hop[1].name]
        ) * ((self._pred_degs[hop[0].name] + self._obj_degs[hop[1].name]) / 2)
