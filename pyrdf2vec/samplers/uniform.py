from typing import Tuple

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler


@attr.s
class UniformSampler(Sampler):
    """Sampler that assigns a uniform weight to each hop in a Knowledge Graph.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    Attributes:
        inverse: True if Inverse Uniform Weight sampling satrategy must be
            used, False otherwise.
            Default to False.
        random_state: The random_state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    def fit(self, kg: KG) -> None:
        """Since the weights are uniform, this function does nothing.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop: Tuple[Vertex, Vertex]) -> int:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for this hop.

        """
        return 1
