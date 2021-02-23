import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import attr
import numpy as np

from pyrdf2vec.graphs import KG, Vertex


class RemoteNotSupported(Exception):
    """Base exception class for the lack of support of a sampling strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


@attr.s
class Sampler(ABC):
    """Base class for the sampling strategies.

    Attributes:
        inverse: True if the inverse sampling strategy must be used,
            False otherwise.
            Defaults to False.
        split: True if the split sampling strategy must be used,
            False otherwise.
            Defaults to False.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.  Defaults to
            None.

    """

    inverse: bool = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    split: bool = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    _is_support_remote: bool = attr.ib(init=False, repr=False, default=False)
    _random_state: Optional[int] = attr.ib(
        init=False,
        repr=False,
        default=None,
    )
    _vertices_deg: Dict[str, int] = attr.ib(init=False, repr=False, default={})
    # Tags vertices that appear at the max depth or of which all their children
    # are tagged.
    _visited: Set[Tuple[Tuple[Vertex, Vertex], int]] = attr.ib(
        init=False, repr=False, default=set()
    )

    @abstractmethod
    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        if kg._is_remote and not self._is_support_remote:
            raise RemoteNotSupported(
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
    def get_weight(self, hop: Tuple[Vertex, Vertex]):
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for this hop.

        """
        raise NotImplementedError("This has to be implemented")

    def get_weights(self, hops: List[Tuple[Vertex, Vertex]]) -> List[float]:
        """Gets the weights of the hops

        Args:
            hops: The hops.

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        weights: List[float] = [self.get_weight(hop) for hop in hops]
        if self.inverse:
            weights = [
                max(weights) - (weight - min(weights)) for weight in weights
            ]
        if self.split:
            weights = [
                weight / self._vertices_deg[hop[1].name]
                for weight, hop in zip(weights, hops)
            ]
        return [weight / sum(weights) for weight in weights]

    def sample_neighbor(
        self,
        kg: KG,
        walk: Tuple[Vertex, ...],
        is_last_depth: bool,
        is_reverse: bool = False,
    ) -> Optional[Tuple[Vertex, Vertex]]:
        """Samples an unvisited random neighbor in the (predicate, object)
        form, according to the weight of hops for a given walk.

        kg: The Knowledge Graph.
        walk: The walk with one or several vertices.
        is_last_hop: True if the next neighbor to be visited is the last one
            for the desired depth. Otherwise False.
        is_reverse: True to get the parent neighbors instead of the child
            neighbors. Otherwise False.
            Defaults to False

        Returns:
            An unvisited neighbor in the form (predicate, object).

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

        if is_last_depth:
            self.visited.add((untagged_neighbors[rnd_id], len(walk)))
        return untagged_neighbors[rnd_id]

    @property
    def visited(self) -> Set[Tuple[Tuple[Vertex, Vertex], int]]:
        """Gets the tagged vertices that appear at the max depth or of which
        all their children are tagged.

        Returns:
            The tagged vertices.

        """
        return self._visited

    @visited.setter
    def visited(self, visited: Set[Tuple[Tuple[Vertex, Vertex], int]]) -> None:
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
