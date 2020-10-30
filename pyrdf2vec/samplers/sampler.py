import abc
from typing import Any, Set

import numpy as np

from pyrdf2vec.graphs import KG


class Sampler(metaclass=abc.ABCMeta):
    """Base class for the sampling strategies.

    Attributes:
        inverse: True if the inverse sampling strategy must be used,
            False otherwise. Default to False.
        split: True if the split sampling strategy must be used,
            False otherwise. Default to False.

    """

    def __init__(self, inverse=False, split=False):
        self.inverse = inverse
        self.split = split
        self.remote_supported = False

    @abc.abstractmethod
    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        if kg.is_remote and not self.remote_supported:
            raise ValueError("This sampler is not supported for remote KGs.")
        if self.split:
            self.degrees = {}
            for vertex in kg._vertices:
                if not vertex.predicate:
                    self.degrees[vertex.name] = len(
                        kg.get_inv_neighbors(vertex)
                    )

    def initialize(self) -> None:
        """Tags vertices that appear at the max depth or of which all their
        children are tagged.

        """
        self.visited: Set[Any] = set()  #

    def sample_neighbor(self, kg: KG, walk, last):
        not_tag_neighbors = [
            x
            for x in kg.get_hops(walk[-1])
            if (x, len(walk)) not in self.visited
        ]

        # If there are no untagged neighbors, then tag
        # this vertex and return None
        if len(not_tag_neighbors) == 0:
            if len(walk) > 2:
                self.visited.add(((walk[-2], walk[-1]), len(walk) - 2))
            return None

        weights = [self.get_weight(hop) for hop in not_tag_neighbors]
        if self.inverse:
            weights = [max(weights) - (x - min(weights)) for x in weights]
        if self.split:
            weights = [
                w / self.degrees[v[1]]
                for w, v in zip(weights, not_tag_neighbors)
            ]
        weights = [x / sum(weights) for x in weights]

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = np.random.choice(range(len(not_tag_neighbors)), p=weights)
        if last:
            self.visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]

    @abc.abstractmethod
    def get_weight(self, hop):
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
