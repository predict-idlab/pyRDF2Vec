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
        if kg.is_remote and not self._is_support_remote:
            raise RemoteNotSupported(
                "Invalid sampling strategy. Please, choose a sampling strategy"
                + " that can fetch walks via a SPARQL endpoint server."
            )
        if self.split:
            for vertex in kg._vertices:
                if not vertex.predicate:
                    self._vertices_deg[vertex.name] = len(
                        kg.get_neighbors(vertex, reverse=True)
                    )

    @abstractmethod
    def get_weight(self, hop: Tuple[Vertex, Vertex]) -> int:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

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
        self, kg: KG, walk: Tuple[Vertex], is_last_neighbor: bool
    ) -> Optional[Tuple[Vertex, Vertex]]:
        """Samples a random neighbor and check if all its children are
        tagged. If there are no untagged neighbors, this function will tag the
        vertex and return None.

        kg: The Knowledge Graph.
        walk: The walk.
        is_last_neighbor: True if the neighbor is the class, False otherwise.

        Returns:
            The sample neighbor

        """
        untagged_neighbors = [
            hop
            for hop in kg.get_hops(walk[-1])
            if (hop, len(walk)) not in self.visited
        ]

        if len(untagged_neighbors) == 0:
            if len(walk) > 2:
                self.visited.add(((walk[-2], walk[-1]), len(walk) - 2))
            return None

        rnd_id = np.random.RandomState(self._random_state).choice(
            range(len(untagged_neighbors)),
            p=self.get_weights(untagged_neighbors),  # type: ignore
        )
        if is_last_neighbor:
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
