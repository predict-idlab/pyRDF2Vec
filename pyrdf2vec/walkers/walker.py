import abc
from typing import Any, List, Set, Tuple

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler


class Walker(metaclass=abc.ABCMeta):
    """Base class for the walking strategies.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = None,
    ):
        self.depth = depth
        self.walks_per_graph = walks_per_graph
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = UniformSampler()

    def extract(
        self, kg: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        self.sampler.fit(kg)
        return self._extract(kg, instances)

    @abc.abstractmethod
    def _extract(
        self, kg: KG, instances: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        raise NotImplementedError("This must be implemented!")

    def print_walks(
        self,
        kg: KG,
        instances: List[rdflib.URIRef],
        file_name: str,
    ) -> None:
        """Prints the walks of a knowledge graph.

        Args:
            kg: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.
            file_name: The filename that contains the rdflib.Graph

        """
        walks = self.extract(kg, instances)
        walk_strs = []
        for _, walk in enumerate(walks):
            s = ""
            for i in range(len(walk)):
                s += f"{walk[i]} "
                if i < len(walk) - 1:
                    s += "--> "
            walk_strs.append(s)

        with open(file_name, "w+") as f:
            for s in walk_strs:
                f.write(s)
                f.write("\n\n")
