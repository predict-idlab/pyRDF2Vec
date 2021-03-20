from typing import Dict

import attr
import networkx as nx

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class PageRankSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric approach. With this strategy, some
    nodes are more important than others and hence there will be resources
    which are more frequent in the walks as others.

    Args:

        alpha: The damping for PageRank.
            Defaults to 0.85.

    """

    alpha: float = attr.ib(
        kw_only=True,
        default=0.85,
        validator=attr.validators.instance_of(float),
    )
    """The damping for Page Rank."""

    _pageranks: Dict[str, float] = attr.ib(
        init=False, repr=False, factory=dict
    )
    """The Page Rank dictionary."""

    def fit(self, kg: KG) -> None:
        """Fits the sampling strategy by running PageRank on a provided KG
        according to the specified damping.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        nx_graph = nx.DiGraph()

        for vertex in kg._vertices:
            if not vertex.predicate:
                nx_graph.add_node(vertex.name, vertex=vertex)
                for predicate in kg.get_neighbors(vertex):
                    for obj in kg.get_neighbors(predicate):
                        nx_graph.add_edge(
                            vertex.name, obj.name, name=predicate.name
                        )
        self._pageranks = nx.pagerank(nx_graph, alpha=self.alpha)

    def get_weight(self, hop: Hop) -> float:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        Raises:
            ValueError: If there is an attempt to access the weight of a hop
                without the sampling strategy having been trained.

        """
        if len(self._pageranks) == 0:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._pageranks[hop[1].name]
