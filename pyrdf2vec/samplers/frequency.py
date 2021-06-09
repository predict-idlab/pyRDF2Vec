from collections import defaultdict
from typing import DefaultDict, Tuple

import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class ObjFreqSampler(Sampler):
    """Object Frequency Weight node-centric sampling strategy which prioritizes
    walks containing edges with the highest degree objects. The degree of an
    object being defined by the number of predicates present in its
    neighborhood.

     Attributes:
         _counts: The counter for vertices.
             Defaults to defaultdict.
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

    _counts = attr.ib(
        init=False,
        type=DefaultDict[str, int],
        repr=False,
        factory=lambda: defaultdict(dict),
    )

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of parent
        predicates present in the neighborhood of each vertex.

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
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if not self._counts:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._counts[hop[1].name]


@attr.s
class PredFreqSampler(Sampler):
    """Predicate Frequency Weight edge-centric sampling strategy which
    prioritizes walks containing edges with the highest degree predicates. The
    degree of a predicate being defined by the number of occurences that a
    predicate appears in a Knowledge Graph.

    Attributes:
        _counts: The counter for vertices.
            Defaults to defaultdict.
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

    _counts: DefaultDict[str, int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of occurences that
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
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if not self._counts:
            raise ValueError(
                "You must call the `fit(kg)` method before get the weight of"
                + " a hop."
            )
        return self._counts[hop[0].name]


@attr.s
class ObjPredFreqSampler(Sampler):
    """Predicate-Object Frequency Weight edge-centric sampling strategy which
    prioritizes walks containing edges with the highest degree of (predicate,
    object) relations. The degree of a such relation being defined by the
    number of occurences that a (predicate, object) relation appears in a
    Knowledge Graph.

    Attributes:
        _counts: The counter for vertices.
            Defaults to defaultdict.
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

    _counts: DefaultDict[Tuple[str, str], int] = attr.ib(
        init=False, repr=False, factory=lambda: defaultdict(dict)
    )

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by counting the number of occurrences of
        an object belonging to a subject.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        for vertex in kg._vertices:
            if vertex.predicate:
                objs = list(kg.get_neighbors(vertex))
                if objs:
                    obj = objs[0]
                    if (vertex.name, obj.name) in self._counts:
                        self._counts[(vertex.name, obj.name)] += 1
                    else:
                        self._counts[(vertex.name, obj.name)] = 1

    def get_weight(self, hop: Hop) -> int:
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
        if not self._counts:
            raise ValueError(
                "You must call the `fit(kg)` method before get the weight of"
                + " a hop."
            )
        return self._counts[(hop[0].name, hop[1].name)]
