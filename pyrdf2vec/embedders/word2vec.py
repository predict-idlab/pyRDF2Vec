from typing import List

import rdflib
from gensim.models.word2vec import Word2Vec as W2V
from sklearn.utils.validation import check_is_fitted

from pyrdf2vec.embedders import Embedder


class Word2Vec(Embedder):
    """Defines Word2Vec embedding technique.

    Attributes:
        vector_size: The dimension of the embeddings.
            Defaults to 500.
        walkers: The walking strategy.
            Defaults to pyrdf2vec.walkers.RandomWalker(2, None,
            UniformSampler(inverse=False)).
        n_jobs: The number of threads to train the model.
            Defaults to 1.
        sg: The training algorithm. 1 for skip-gram; otherwise CBOW.
            Defaults to 1.
        epochs: The number of iterations over the corpus.
            Defaults to 10.
        negative: The negative sampling.
            If > 0, the negative sampling will be used. Otherwise no negative
            sampling is used.
            Defaults to 25.
        min_count: The total frequency to ignores all words.
            Defaults to 1.

    """

    def __init__(
        self,
        vector_size: int = 500,
        n_jobs: int = 1,
        window: int = 5,
        sg: int = 1,
        epochs: int = 10,
        negative: int = 25,
        min_count: int = 1,
    ):
        self.vector_size = vector_size
        self.n_jobs = n_jobs
        self.window = window
        self.sg = sg
        self.epochs = epochs
        self.negative = negative
        self.min_count = min_count

    def fit(self, corpus):
        self.model_ = W2V(
            corpus,
            size=self.vector_size,
            window=self.window,
            workers=self.n_jobs,
            sg=self.sg,
            iter=self.epochs,
            negative=self.negative,
            min_count=self.min_count,
            seed=42,
        )
        return self.model_

    def transform(self, entities: List[rdflib.URIRef]) -> List[str]:
        """Constructs a feature vector for the provided entities.

        Args:
            entities: The entities to create the embedding.
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
