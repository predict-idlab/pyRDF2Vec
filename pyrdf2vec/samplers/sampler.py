import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import attr
import numpy as np

from pyrdf2vec.graphs import KG
from pyrdf2vec.typings import Hop, Walk


class SamplerNotSupported(Exception):
    """Base exception class for the lack of support of a sampling strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


@attr.s
class Sampler(ABC):
    """Base class of the sampling strategies.

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

    inverse = attr.ib(
        default=False, type=bool, validator=attr.validators.instance_of(bool)
    )

    split = attr.ib(default=False, validator=attr.validators.instance_of(bool))

    _is_support_remote = attr.ib(
        init=False, type=bool, repr=False, default=False
    )

    _random_state = attr.ib(
        init=False,
        type=Optional[int],
        repr=False,
        default=None,
    )

    _vertices_deg = attr.ib(
        init=False, type=Dict[str, int], repr=False, factory=dict
    )

    _visited = attr.ib(
        init=False, type=Set[Tuple[Hop, int]], repr=False, factory=set
    )

    @abstractmethod
    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy.

        Args:
            kg: The Knowledge Graph.

        Raises:
            SamplerNotSupported: If there is an attempt to use an invalid
                sampling strategy to a remote Knowledge Graph.

        """
        if kg._is_remote and not self._is_support_remote:
            raise SamplerNotSupported(
                "Invalid sampling strategy. Please, choose a sampling strategy"
                + " that can fetch walks via a SPARQL endpoint server."
            )
        if self.split:
            for vertex in kg._vertices:
                if not vertex.predicate:
                    self._vertices_deg[vertex.name] = len(
                        kg.get_neighbors(vertex, is_reverse=True)
                    )

    @abstractmethod
    def get_weight(self, hop: Hop):
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This has to be implemented")

    def get_weights(self, hops: List[Hop]) -> Optional[List[float]]:
        """Gets the weights of the provided hops.

        Args:
            hops: The hops to get the weights.

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        weights: List[float] = [self.get_weight(hop) for hop in hops]
        if {} in weights:
            return []
        if self.inverse:
            weights = [
                max(weights) - (weight - min(weights)) for weight in weights
            ]
        if self.split:
            weights = [
                weight / self._vertices_deg[hop[1].name]
                for weight, hop in zip(weights, hops)
                if self._vertices_deg[hop[1].name] != 0
            ]
        return [
            weight / sum(weights) for weight in weights if sum(weights) != 0
        ]

    def sample_hop(
        self, kg: KG, walk: Walk, is_last_hop: bool, is_reverse: bool = False
    ) -> Optional[Hop]:
        """Samples an unvisited random hop in the (predicate, object)
        form, according to the weight of hops for a given walk.

        Args:
            kg: The Knowledge Graph.
            walk: The walk with one or several vertices.
            is_last_hop: True if the next hop to be visited is the last
                one for the desired depth, False otherwise.
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False.

        Returns:
            An unvisited hop in the (predicate, object) form.

        """
        subj = walk[0] if is_reverse else walk[-1]

        untagged_neighbors = [
            pred_obj
            for pred_obj in kg.get_hops(subj, is_reverse)
            if (pred_obj, len(walk)) not in self.visited
        ]

        if len(untagged_neighbors) == 0:
            if len(walk) > 2:
                pred_obj = (
                    (walk[1], walk[0]) if is_reverse else (walk[-2], walk[-1])
                )
                self.visited.add((pred_obj, len(walk) - 2))
            return None

        rnd_id = np.random.RandomState(self._random_state).choice(
            range(len(untagged_neighbors)),
            p=self.get_weights(untagged_neighbors),
        )

        if is_last_hop:
            self.visited.add((untagged_neighbors[rnd_id], len(walk)))
        return untagged_neighbors[rnd_id]

    @property
    def visited(self) -> Set[Tuple[Hop, int]]:
        """Gets the tagged vertices that appear at the max depth or of which
        all their children are tagged.

        Returns:
            The tagged vertices.

        """
        return self._visited

    @visited.setter
    def visited(self, visited: Set[Tuple[Hop, int]]) -> None:
        """Sets the value of the tagged vertices.

        Args:
            visited: The tagged vertices.

        """
        self._visited = set() if visited is None else visited

    @property
    def random_state(self) -> Optional[int]:
        """Gets the random state.

        Returns:
            The random state.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]):
        """Sets the random state.

        Args:
            random_state: The random state.

        """
        self._random_state = random_state
        random.seed(random_state)
