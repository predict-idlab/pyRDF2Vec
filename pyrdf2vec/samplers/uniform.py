import attr

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop


@attr.s
class UniformSampler(Sampler):
    """Uniform sampling strategy that assigns a uniform weight to each edge in
    a Knowledge Graph, in order to prioritizes walks with strongly connected
    entities.

    Attributes:
        _is_support_remote: True if the sampling strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        _random_state: The random state to use to keep random determinism with
            the sampling strategy.
            Defaults to None.
        _vertices_deg: The degree of the vertices.
            Defaults to {}.
        _visited: Tags vertices that appear at the max depth or of which all
            their children are tagged.
            Defaults to set.
        inverse: True if the inverse algorithm must be used, False otherwise.
            Defaults to False.
        split: True if the split algorithm must be used, False otherwise.
            Defaults to False.

    """

    inverse = attr.ib(
        init=False,
        default=False,
        type=bool,
        validator=attr.validators.instance_of(bool),
    )

    split = attr.ib(
        init=False,
        default=False,
        type=bool,
        validator=attr.validators.instance_of(bool),
    )

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    def fit(self, kg: KG) -> None:
        """Since the weights are uniform, this function does nothing.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop: Hop) -> int:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        """
        return 1
