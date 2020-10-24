from typing import Dict

import networkx as nx

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler


class PageRankSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric approach. With this strategy, some
    nodes are more important than others and hence there will be resources
    which are more frequent in the walks as others.

    Attributes:
        inverse: True if Inverse PageRank Weight must be used, False otherwise.
            Default to False.
        split: True if PageRank Split Weight must be used, False otherwise.
            Default to False.
        alpha: The threshold.
            Default to 0.85.

    """

    def __init__(
        self, inverse: bool = False, split: bool = False, alpha: float = 0.85
    ):
        super().__init__(inverse, split)
        self.alpha = alpha

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
        self.pageranks = nx.pagerank(nx_graph, alpha=self.alpha)

    def get_weight(self, hop) -> Dict:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return self.pageranks[hop[1].name]
