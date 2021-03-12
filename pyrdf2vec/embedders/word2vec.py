from typing import List

from gensim.models.word2vec import Word2Vec as W2V

from pyrdf2vec.embedders import Embedder


class Word2Vec(Embedder):
    """Defines Word2Vec embedding technique.

    For more details: https://radimrehurek.com/gensim/models/word2vec.html

    """

    def __init__(self, **kwargs):
        self.kwargs = {
            "size": 500,
            "negative": 25,
            "iter": 10,
            "min_count": 1,
            "sg": 1,
            **kwargs,
        }
        self.model_ = W2V(**self.kwargs)

    def fit(
        self, corpus: List[List[str]], is_update: bool = False
    ) -> "Embedder":
        """Fits the Word2Vec model based on provided corpus.

        Args:
            corpus: The corpus.
            is_update: If true, the new corpus will be added to old model's
                corpus.

        Returns:
            The fitted Word2Vec model.

        """
        self.model_.build_vocab(corpus, update=is_update)
        self.model_.train(
            corpus,
            total_examples=self.model_.corpus_count,
            epochs=self.model_.epochs,
        )
        return self

    def transform(self, entities: List[str]) -> List[str]:
        """Constructs a features vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        if not all([entity in self.model_.wv for entity in entities]):
            raise ValueError(
                "The entities must have been provided to fit() first "
                "before they can be transformed into a numerical vector."
            )
        return [self.model_.wv.get_vector(entity) for entity in entities]
