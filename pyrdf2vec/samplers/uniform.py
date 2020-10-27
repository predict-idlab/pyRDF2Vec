from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler


class UniformSampler(Sampler):
    """Defines the Uniform Weight Weight sampling strategy.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    Attributes:
        inverse: True if Inverse Uniform Weight sampling satrategy must be
            used, False otherwise. Default to False.

    """

    def __init__(self, inverse=False):
        super().__init__(inverse)
        self.remote_supported = True

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        pass

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
        return 1
