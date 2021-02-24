from typing import Dict, Tuple

import attr
import networkx as nx

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler


@attr.s
class PageRankSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric approach. With this strategy, some
    nodes are more important than others and hence there will be resources
    which are more frequent in the walks as others.

    Args:
        inverse: True if Inverse PageRank Weight must be used, False otherwise.
            Default to False.
        split: True if PageRank Split Weight must be used, False otherwise.
            Default to False.
        alpha: The damping for PageRank.
            Default to 0.85.
        random_state: The random_state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    alpha: float = attr.ib(
        kw_only=True,
        default=0.85,
        validator=attr.validators.instance_of(float),
    )
    _pageranks: Dict[str, float] = attr.ib(
        init=False, repr=False, factory=dict
    )

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

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

    def get_weight(self, hop: Tuple[Vertex, Vertex]):
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for this hop.

        """
        if len(self._pageranks) == 0:
            raise ValueError(
                "You must call the `fit(kg)` function before get the weight of"
                + " a hop."
            )
        return self._pageranks[hop[1].name]
