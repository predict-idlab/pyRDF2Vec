from gensim.models.word2vec import Word2Vec
from sklearn.utils.validation import check_is_fitted

from rdf2vec.walkers import RandomWalker


class RDF2VecTransformer:
    """Transforms nodes in a knowledge graph into an embedding.

    Attributes:
        vector_size (int): The dimension of the embeddings.
            Defaults to 500.
        walkers (Walker): The walking strategy.
            Defaults to RandomWalker(2, float("inf)).
        n_jobs (int): The number of threads to train the model.
            Defaults to 1.
        sg (int): The training algorithm. 1 for skip-gram; otherwise CBOW.
            Defaults to 1.
        max_iter (int): The number of iterations (epochs) over the corpus.
            Defaults to 10.
        negative (int): The negative sampling.
            If > 0, the negative sampling will be used. Otherwise no negative
            sampling is used.
            Defaults to 25.
        min_count (int): The total frequency to ignores all words.
            Defaults to 1.

    """

    def __init__(
        self,
        vector_size=500,
        walkers=[RandomWalker(2, float("inf"))],
        n_jobs=1,
        window=5,
        sg=1,
        max_iter=10,
        negative=25,
        min_count=1,
    ):
        self.max_iter = max_iter
        self.min_count = min_count
        self.n_jobs = n_jobs
        self.negative = negative
        self.sg = sg
        self.vector_size = vector_size
        self.walkers = walkers
        self.window = window

    def fit(self, graph, instances):
        """Fits the embedding network based on provided instances.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (array-like): The instances to create the embedding.
                The test instances should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        """
        self.walks_ = []
        for walker in self.walkers:
            self.walks_ += list(walker.extract(graph, instances))
        print(
            f"Extracted {len(self.walks_)} walks for {len(instances)} instances!"
        )
        sentences = [list(map(str, x)) for x in self.walks_]

        self.model_ = Word2Vec(
            sentences,
            size=self.vector_size,
            window=self.window,
            workers=self.n_jobs,
            sg=self.sg,
            iter=self.max_iter,
            negative=self.negative,
            min_count=self.min_count,
            seed=42,
        )

    def transform(self, graph, instances):
        """Constructs a feature vector for the provided instances.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph
                The graph from which we will extract neighborhoods for the
                provided instances.
            instances (array-like): The instances to create the embedding.
                The test instances should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            array-like: The embeddings of the provided instances.

        """
        check_is_fitted(self, ["model_"])
        feature_vectors = []
        for instance in instances:
            feature_vectors.append(self.model_.wv.get_vector(str(instance)))
        return feature_vectors

    def fit_transform(self, graph, instances):
        """Creates a Word2Vec model and generate embeddings for the provided
        instances.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph
                The graph from which we will extract neighborhoods for the
                provided instances.
            instances (array-like): The instances to create the embedding.
                The test instances should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            array-like: The embeddings of the provided instances.

        """
        self.fit(graph, instances)
        return self.transform(graph, instances)
