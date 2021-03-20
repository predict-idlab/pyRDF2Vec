from collections import defaultdict
from typing import DefaultDict, Tuple

import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class ObjFreqSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric object frequency approach. With
    this strategy, entities which have a high in degree get visisted more
    often.

    """

    _counts: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )
    """Counter for vertices."""

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of available
        neighbors for each vertex.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        for vertex in kg._vertices:
            if not vertex.predicate:
                self._counts[vertex.name] = len(
                    kg.get_neighbors(vertex, is_reverse=True)
                )

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if len(self._counts) == 0:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._counts[hop[1].name]


@attr.s
class PredFreqSampler(Sampler):
    """Defines the Predicate Frequency Weight sampling strategy.

    This sampling strategy is an edge-centric approach. With this strategy,
    edges with predicates which are commonly used in the dataset are more often
    followed.

    """

    _counts: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )
    """Counter for vertices."""

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of occurance that
        a predicate appears in the Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        for vertex in kg._vertices:
            if vertex.predicate:
                if vertex.name in self._counts:
                    self._counts[vertex.name] += 1
                else:
                    self._counts[vertex.name] = 1

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if len(self._counts) == 0:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._counts[hop[0].name]


@attr.s
class ObjPredFreqSampler(Sampler):
    """Defines the Predicate-Object Frequency Weight sampling strategy.

    This sampling strategy is a edge-centric approach. This strategy is similar
    to the Predicate Frequency Weigh sampling strategy, but differentiates
    between the objects as well.

    """

    _counts: DefaultDict[Tuple[str, str], int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )
    """Counter for vertices."""

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of occurance of
        having two neighboring vertices.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        for vertex in kg._vertices:
            if vertex.predicate:
                neighbors = list(kg.get_neighbors(vertex))
                if len(neighbors) > 0:
                    obj = neighbors[0]
                    if (vertex.name, obj.name) in self._counts:
                        self._counts[(vertex.name, obj.name)] += 1
                    else:
                        self._counts[(vertex.name, obj.name)] = 1

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if len(self._counts) == 0:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._counts[(hop[0].name, hop[1].name)]
