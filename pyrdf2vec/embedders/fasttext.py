from __future__ import annotations

import re
from typing import Any, List

import attr
import numpy as np
from gensim.models.fasttext import FastText as FT
from gensim.models.fasttext import FastTextKeyedVectors
from numpy import float32 as REAL
from numpy import ones

from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities, SWalk


@attr.s(init=False)
class FastText(Embedder):
    """Defines the FastText embedding technique.

    SEE: https://radimrehurek.com/gensim/models/fasttext.html

    The RDF2Vec implementation of FastText does not consider the min_n and
    max_n parameters for n_gram splitting.

    This implementation for RDF2Vec computes ngrams for walks only by splitting
    (by their symbol "#") the URIs of subjects and predicates. Indeed, objects
    being encoded in MD5, splitting in ngrams does not make sense.

    It is likely that you want to provide another split strategy for the
    calculation of the n-grams of the entities. If this is the case, provide
    your own compute_ngrams_bytes function to FastText.

    Attributes:
        _model: The gensim.models.word2vec model.
            Defaults to None.
        kwargs: The keyword arguments dictionary.
            Defaults to { bucket=2000000, min_count=0, max_n=0, min_n=0,
                negative=20, vector_size=500 }
        func_computing_ngrams: The function to call for the computation of
            ngrams. In case of reimplementation, it is important to respect the
            signature imposed by gensim:
            func(entity: str, minn: int = 0, maxn: int = 0) -> List[bytes]
            Defaults to compute_ngrams_bytes

    """

    kwargs = attr.ib(init=False, default=None)
    func_computing_ngrams = attr.ib(kw_only=True, repr=False)
    _model = attr.ib(init=False, type=FT, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "bucket": 2000000,
            "min_count": 0,
            "negative": 20,
            "vector_size": 500,
            **kwargs,
        }
        if "func_computing_ngrams" in self.kwargs:
            self.func_computing_ngrams = self.kwargs["func_computing_ngrams"]
            self.kwargs.pop("func_computing_ngrams")
        else:
            self.func_computing_ngrams = None

        self._model = FT(**self.kwargs)
        self._model.wv = RDFFastTextKeyedVectors(
            vector_size=self.kwargs["vector_size"],
            bucket=self.kwargs["bucket"],
            func_computing_ngrams=self.func_computing_ngrams,
        )
        self._model.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        self._model.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)

    def fit(
        self, walks: List[List[SWalk]], is_update: bool = False
    ) -> Embedder:
        """Fits the FastText model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new corpus should be added to old model's
                walks, False otherwise.
                Defaults to False.

        Returns:
            The fitted FastText model.

        """
        corpus = [walk for entity_walks in walks for walk in entity_walks]
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


@attr.s
class RDFFastTextKeyedVectors(FastTextKeyedVectors):

    bucket: int = attr.ib(default=2000000)
    vector_size: int = attr.ib(default=500)
    func_computing_ngrams = attr.ib(kw_only=True, default=None, repr=False)

    def __attrs_post_init__(self):
        super().__init__(self.vector_size, 0, 0, self.bucket)
        if self.func_computing_ngrams is None:
            self.func_computing_ngrams = self.compute_ngrams_bytes

    def get_vector(self, word, norm=False):
        if word in self.key_to_index:
            return super().get_vector(word, norm=norm)
        elif self.bucket == 0:
            raise KeyError(
                "cannot calculate vector for OOV word without ngrams"
            )
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngram_weights = self.vectors_ngrams
            ngram_hashes = self.ft_ngram_hashes(word, 0, 0, self.bucket)
            if len(ngram_hashes) == 0:
                #
                # If it is impossible to extract _any_ ngrams from the input
                # word, then the best we can do is return a vector that points
                # to the origin.  The reference FB implementation does this,
                # too.
                #
                # https://github.com/RaRe-Technologies/gensim/issues/2402
                #
                return word_vec
            for nh in ngram_hashes:
                word_vec += ngram_weights[nh]
            if norm:
                return word_vec / np.linalg.norm(word_vec)
            else:
                return word_vec / len(ngram_hashes)

    def recalc_char_ngram_buckets(self) -> None:
        """Reimplementation of the recalc_char_ngram_buckets method of
        gensim. This overwrite is needed to call our ft_ngram_hashes method.

        """
        if self.bucket == 0:
            self.buckets_word = [np.array([], dtype=np.uint32)] * len(
                self.index_to_key
            )
            return

        self.buckets_word = [None] * len(self.index_to_key)  # type:ignore

        for i, word in enumerate(self.index_to_key):
            self.buckets_word[i] = np.array(
                self.ft_ngram_hashes(word, 0, 0, self.bucket),
                dtype=np.uint32,
            )

    def compute_ngrams_bytes(
        self, entity: str, minn: int = 0, maxn: int = 0
    ) -> List[bytes]:
        """Reimplementation of the compute_ngrams_bytes method of gensim. This
           overwrite is needed to call our compute_ngrams_bytes method.

        Args:
            entity: The entity to hash the ngrams.
            minn: Minimum length of char n-grams to be used for training entity
                representations.
                Defaults to 0.
            maxn: Maximum length of char n-grams to be used for training
                entity representations.
                Defaults to 0.

        Returns:
            The ngrams bytes.

        """
        if "http" in entity:
            ngrams = " ".join(re.split("[#]", entity)).split()
            return [str.encode(ngram) for ngram in ngrams]
        return [str.encode(entity)]

    def ft_hash_bytes(self, bytez: bytes) -> int:
        """Computes hash based on `bytez`.

        Args:
            bytez: The byte to hash

        Returns:
            The hash of the string.

        """
        h = 2166136261
        for b in bytez:
            h = h ^ b
            h = h * 16777619
        return h

    def ft_ngram_hashes(
        self,
        entity: str,
        minn: int = 0,
        maxn: int = 0,
        num_buckets: int = 2000000,
    ) -> List[Any]:
        """Reimplementation of the ft_ngram_hahes method of gensim. This
        overwrite is needed to call our compute_ngrams_bytes method.

        Args:
             entity: The entity to hash the ngrams.
             minn: Minimum length of char n-grams to be used for training
                entity representations.
                Defaults to 0.
             maxn: Maximum length of char n-grams to be used for training
                entity representations.
                Defaults to 0.
             num_buckets: Character ngrams are hashed into a fixed number of
                buckets, in order to limit the memory usage of the model.
                Defaults to 2000000.

         Returns:
             The ngrams hashes.

        """
        encoded_ngrams = self.func_computing_ngrams(entity, minn, maxn)
        hashes = [self.ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
        return hashes
