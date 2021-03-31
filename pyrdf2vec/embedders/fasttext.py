from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import attr
from gensim.models.fasttext import (
    FastText as FT,
    FastTextKeyedVectors,
    Word2Vec as W2V,
)
import re

from gensim import utils

from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities, SWalk
from gensim.models.fasttext_inner import ft_hash_bytes, MAX_WORDS_IN_BATCH
from numpy import ones, vstack, float32 as REAL
import logging

logger = logging.getLogger(__name__)


@attr.s(init=False)
class FastText(Embedder):
    """Defines the FastText embedding technique.

    SEE: https://radimrehurek.com/gensim/models/fasttext.html

    WARNING: The RDF2Vec implementation of FastText does not consider the min_n
    and max_n parameters for n_gram splitting.

    This implementation for RDF2Vec computes ngrams for walks only by splitting
    (by their symbol "#") the URIs of subjects and predicates. Indeed, objects
    being encoded in MD5, splitting in ngrams does not make sense.

    It is likely that you want to provide another split strategy for the
    calculation of the n-grams of the entities. If this is the case, provide your
    own compute_ngrams_bytes function to FastText.

    Attributes:
        _model: The gensim.models.word2vec model.
            Defaults to None.
        kwargs: The keyword arguments dictionary.
            Defaults to { bucket=2000000, min_count=0, max_n=0, min_n=0, negative=20,
                vector_size=500 }

    """

    kwargs = attr.ib(init=False, default=None)
    _model = attr.ib(init=False, type=FT, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "bucket": 2000000,
            "min_count": 0,
            "negative": 20,
            "vector_size": 500,
            **kwargs,
        }
        self._model = FT(**self.kwargs)
        self._model.wv = RDFFastTextKeyedVectors(
            vector_size=self.kwargs["vector_size"],
            bucket=self.kwargs["bucket"],
        )
        self._model.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        self._model.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)

    def estimate_memory(
        self, vocab_size: int = None, report: Dict[Any, Any] = None
    ) -> Dict[Any, Any]:
        """Reimplementation of the estimate_memory method of gensim. This
        overwrite is needed to call our ft_ngram_hashes method.

        Args:
            vocab_size: The size of the vocabulary
            report: The existing memory report.

        Returns:
            The memory report.

        """
        vocab_size = vocab_size or len(self._model.wv)
        vec_size = self._model.vector_size * np.dtype(np.float32).itemsize
        l1_size = self._model.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report["vocab"] = len(self._model.wv) * (
            700 if self._model.hs else 500
        )
        report["syn0_vocab"] = len(self._model.wv) * vec_size
        num_buckets = self._model.wv.bucket
        if self._model.hs:
            report["syn1"] = len(self._model.wv) * l1_size
        if self._model.negative:
            report["syn1neg"] = len(self._model.wv) * l1_size
        if self._model.wv.bucket:
            report["syn0_ngrams"] = self._model.wv.bucket * vec_size
            num_ngrams = 0
            for word in self._model.wv.key_to_index:
                hashes = ft_ngram_hashes(word, 0, 0, self._model.wv.bucket)
                num_ngrams += len(hashes)
            # A list (64 bytes) with one np.array (100 bytes) per key, with a total of
            # num_ngrams uint32s (4 bytes) amongst them.
            # Only used during training, not stored with the model.
            report["buckets_word"] = (
                64 + (100 * len(self._model.wv)) + (4 * num_ngrams)
            )
        report["total"] = sum(report.values())
        logger.info(
            "estimated required memory for %i words, %i buckets and %i dimensions: %i bytes",
            len(self._model.wv),
            num_buckets,
            self._model.vector_size,
            report["total"],
        )
        return report

    def fit(
        self, walks: List[List[SWalk]], is_update: bool = False
    ) -> Embedder:
        """Fits the FastText model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
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

    def __attrs_post_init__(self):
        super().__init__(self.vector_size, 0, 0, self.bucket)

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
            ngram_hashes = ft_ngram_hashes(word, 0, 0, self.bucket)
            if len(ngram_hashes) == 0:
                #
                # If it is impossible to extract _any_ ngrams from the input
                # word, then the best we can do is return a vector that points
                # to the origin.  The reference FB implementation does this,
                # too.
                #
                # https://github.com/RaRe-Technologies/gensim/issues/2402
                #
                logger.warning(
                    "could not extract any ngrams from %r, returning origin vector",
                    word,
                )
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
                ft_ngram_hashes(word, 0, 0, self.bucket),
                dtype=np.uint32,
            )


def compute_ngrams_bytes(
    entity: str, minn: int = 0, maxn: int = 0
) -> List[bytes]:
    """Reimplementation of the ft_ngram_hahes method of gensim. This overwrite
    is needed to call our compute_ngrams_bytes method.

    Args:
        entity: The entity to hash the ngrams.
        minn: Minimum length of char n-grams to be used for training entity
            representations.
            Defaults to 0.
        maxn: Maximum length of char n-grams to be used for training entity
            representations.
            Defaults to 0.

    Returns:
        The ngrams bytes.

    """
    if "http" in entity:
        ngrams = " ".join(re.split("[#]", entity)).split()
        return [str.encode(ngram) for ngram in ngrams]
    return [str.encode(entity)]


def ft_ngram_hashes(
    entity: str, minn: int = 0, maxn: int = 0, num_buckets: int = 2000000
) -> List[Any]:
    """Reimplementation of the ft_ngram_hahes method of gensim. This overwrite
    is needed to call our compute_ngrams_bytes method.

    Args:
        entity: The entity to hash the ngrams.
        minn: Minimum length of char n-grams to be used for training entity
            representations.
            Defaults to 0.
        maxn: Maximum length of char n-grams to be used for training entity
            representations.
            Defaults to 0.
        num_buckets: Character ngrams are hashed into a fixed number of
            buckets, in order to limit the memory usage of the model.
            Defaults to 2000000.
    Returns:
        The ngrams hashes.

    """
    encoded_ngrams = compute_ngrams_bytes(entity, minn, maxn)
    hashes = [ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
    return hashes
