from __future__ import annotations

from typing import List

import attr
from gensim.models.word2vec import Word2Vec as W2V

from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities


@attr.s(init=False)
class Word2Vec(Embedder):
    """Defines the Word2Vec embedding technique.

    SEE: https://radimrehurek.com/gensim_3.8.3/models/word2vec.html

    """

    kwargs = attr.ib(init=False, default=None)
    """The keyword arguments dictionary.
    Defaults to {size=500, min_count=0, negative=20}.
    """

    _model: W2V = attr.ib(init=False, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "size": 500,
            "min_count": 0,
            "negative": 20,
            **kwargs,
        }
        self._model = W2V(**self.kwargs)

    def fit(self, corpus: List[Entities], is_update: bool = False) -> Embedder:
        """Fits the Word2Vec model based on provided corpus.

        Args:
            corpus: The corpus to fit the model.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The fitted Word2Vec model.

        """
        self._model.build_vocab(corpus, update=is_update)
        self._model.train(
            corpus,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs,
        )
        return self

    def transform(self, entities: Entities) -> Embeddings:
        """The features vector of the provided entities.

            Args:
                entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The features vector of the provided entities.

        """
        if not all([entity in self._model.wv for entity in entities]):
            raise ValueError(
                "The entities must have been provided to fit() first "
                "before they can be transformed into a numerical vector."
            )
        return [self._model.wv.get_vector(entity) for entity in entities]
