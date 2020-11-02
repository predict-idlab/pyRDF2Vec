from typing import List

import rdflib
from gensim.models.word2vec import Word2Vec as W2V
from sklearn.utils.validation import check_is_fitted

from pyrdf2vec.embedders import Embedder


class Word2Vec(Embedder):
    """Defines Word2Vec embedding technique.

    For more details: https://radimrehurek.com/gensim/models/word2vec.html

    """

    def __init__(self, **kwargs):
        kwargs.setdefault("min_count", 0)
        self.kwargs = kwargs

    def fit(self, corpus: List[List[str]]) -> "Word2Vec":
        """Fits the Word2Vec model based on provided corpus.

        Args:
            corpus: The corpus.

        Returns:
            The fitted Word2Vec model.

        """
        self.model_ = W2V(corpus, **self.kwargs)
        return self

    def transform(self, entities: List[rdflib.URIRef]) -> List[str]:
        """Constructs a features vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        check_is_fitted(self, ["model_"])
        if not all([str(entity) in self.model_.wv for entity in entities]):
            raise ValueError(
                "The entities must have been provided to fit() first "
                "before they can be transformed into a numerical vector."
            )
        return [self.model_.wv.get_vector(str(entity)) for entity in entities]
