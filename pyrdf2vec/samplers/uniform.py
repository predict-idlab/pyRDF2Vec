import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class UniformSampler(Sampler):
    """Sampler that assigns a uniform weight to each hop in a Knowledge Graph.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    """

    inverse: bool = attr.ib(
        init=False, default=False, validator=attr.validators.instance_of(bool)
    )
    """True if the inverse algorithm must be used, False otherwise."""

    split: bool = attr.ib(
        init=False, default=False, validator=attr.validators.instance_of(bool)
    )
    """True if the split algorithm must be used, False otherwise."""

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)
    """True if the sampling strategy can be used with a remote Knowledge Graph,
    False Otherwise.
    """

    def fit(self, kg: KG) -> None:
        """Since the weights are uniform, this function does nothing.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop (pred, obj) to get the weight.

        Returns:
            The weight for a given hop.

        """
        return 1
