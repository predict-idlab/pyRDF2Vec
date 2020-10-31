import abc
from typing import List

import rdflib


class Embedder(metaclass=abc.ABCMeta):
    """Base class for the embedding techniques."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, corpus: List[List[str]]):
        """Fits the Word2Vec model based on provided corpus.

        Args:
            corpus: The corpus.

        Returns:
            The fitted model according to an embedding technique.

        """
        raise NotImplementedError("This has to be implemented")

    @abc.abstractmethod
    def transform(self, entities: List[rdflib.URIRef]) -> List[str]:
        """Constructs a features vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        raise NotImplementedError("This has to be implemented")
