from __future__ import annotations

from typing import List

import attr
from gensim.models.doc2vec import Doc2Vec as D2V, TaggedDocument
from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities, SWalk
import numpy as np


@attr.s(init=False)
class Doc2Vec(Embedder):
    """Defines the Doc2Vec embedding technique.

    SEE: https://radimrehurek.com/gensim/models/doc2vec.html

    Attributes:
        _model: The gensim.models.word2vec model.
            Defaults to None.
        kwargs: The keyword arguments dictionary.
            Defaults to { min_count=0, negative=20, vector_size=500 }.

    """

    kwargs = attr.ib(init=False, default=None)
    _model = attr.ib(init=False, type=D2V, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "vector_size": 500,
            "min_count": 0,
            "negative": 20,
            **kwargs,
        }
        self._model = D2V(**self.kwargs)

    def fit(
        self, walks: List[List[SWalk]], is_update: bool = False
    ) -> Embedder:
        """Fits the Doc2Vec model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The fitted Doc2Vec model.

        """
        if is_update:
            corpus = [
                TaggedDocument([walk for walk in e_walk], [8 + i])
                for i, e_walk in enumerate(walks)
            ]
            print(corpus)
        else:
            corpus = [
                TaggedDocument([walk for walk in e_walk], [i])
                for i, e_walk in enumerate(walks)
            ]
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
        return [
            np.add.reduce(
                [
                    self._model.wv.get_vector(walk)
                    for walk in self._model.wv.key_to_index
                    if walk[0] == entity
                ]
            )
            for entity in entities
        ]
