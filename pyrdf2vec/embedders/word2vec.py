from typing import List, Tuple

from gensim.models.word2vec import Word2Vec as W2V

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

    def fit(self, sentences):
        return W2V(
            sentences,
            size=self.vector_size,
            window=self.window,
            workers=self.n_jobs,
            sg=self.sg,
            iter=self.epochs,
            negative=self.negative,
            min_count=self.min_count,
            seed=42,
        )
