import abc


class Embedder(metaclass=abc.ABCMeta):
    """Base class for the embedding techniques."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, corpus):
        raise NotImplementedError("This has to be implemented")

    @abc.abstractmethod
    def transform(self, entities):
        raise NotImplementedError("This has to be implemented")
