import rdflib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from gensim.models.word2vec import Word2Vec
import tqdm
import copy
from graph import Vertex, rdflib_to_kg
from hashlib import md5


class UnknownEntityError(Exception):
    pass


class RDF2VecTransformer():
    """Project random walks or subtrees in graphs into embeddings, suited
    for classification.

    Parameters
    ----------
    vector_size: int (default: 500)
        The dimension of the embeddings.

    max_path_depth: int (default: 1)
        The maximum number of hops to take in the knowledge graph. Due to the 
        fact that we transform s -(p)-> o to s -> p -> o, this will be 
        translated to `2 * max_path_depth` hops internally.

    wl: bool (default: True)
        Whether to use Weisfeiler-Lehman embeddings

    wl_iterations: int (default: 4)
        The number of Weisfeiler-Lehman iterations. Ignored if `wl` is False.

    walks_per_graph: int (default: infinity)
        The maximum number of walks to extract from the neighborhood of
        each instance.

    n_jobs: int (default: 1)
        gensim.models.Word2Vec parameter.

    window: int (default: 5)
        gensim.models.Word2Vec parameter.

    sg: int (default: 1)
        gensim.models.Word2Vec parameter.

    max_iter: int (default: 10)
        gensim.models.Word2Vec parameter.

    negative: int (default: 25)
        gensim.models.Word2Vec parameter.

    min_count: int (default: 1)
        gensim.models.Word2Vec parameter.

    Attributes
    ----------
    model: gensim.models.Word2Vec
        The fitted Word2Vec model. Embeddings can be accessed through
        `self.model.wv.get_vector(str(instance))`.

    """
    def __init__(self, vector_size=500, max_path_depth=1, wl=True, 
                 wl_iterations=4, walks_per_graph=float('inf'), n_jobs=1, 
                 window=5, sg=1, max_iter=10, negative=25, min_count=1):
        
        self.vector_size = vector_size
        self.max_path_depth = max_path_depth
        self.wl = wl
        self.wl_iterations = wl_iterations
        self.walks_per_graph = walks_per_graph

        self.n_jobs = n_jobs
        self.window = window
        self.sg = sg
        self.max_iter = max_iter
        self.negative = negative
        self.min_count = min_count

    def print_walks(self, walks):
        walk_strs = []
        for walk_nr, walk in enumerate(walks):
            s = ''
            for i in range(len(walk)):
                if i % 2:
                    s += '{} '.format(walk[i])
                else:
                    s += '{} '.format(walk[i])
                
                if i < len(walk) - 1:
                    s += '--> '

            walk_strs.append(s)

        with open("test.txt", "w") as myfile:
            for s in walk_strs:
                myfile.write(s)
                myfile.write('\n\n')

    def _extract_random_walks(self, graph, instance):
        walks = graph.extract_random_walks(self.max_path_depth*2, instance, 
                                           max_walks=self.walks_per_graph)

        canonical_walks = set()
        for walk in walks:
            canonical_walk = []
            for i, hop in enumerate(walk):
                if i == 0:
                    canonical_walk.append(hop.name)
                else:
                    # Take the first 8 bytes of the hash, allowing for 
                    # 255**8 unique entities
                    digest = md5(hop.name.encode()).digest()[:8]
                    canonical_walk.append(str(digest))

            canonical_walks.add(tuple(canonical_walk))

        return list(canonical_walks)


    def _extract_wl_walks(self, graph, instance, verbose=False):
        walks = graph.extract_random_walks(self.max_path_depth*2, instance, 
                                           max_walks=self.walks_per_graph)

        canonical_walks = set()
        for n in range(self.wl_iterations + 1):
            for walk in walks:
                canonical_walk = []
                for i, hop in enumerate(walk):
                    # For the root and predicates, we just append the name
                    if i == 0 or i % 2 == 1:
                        canonical_walk.append(hop.name)
                    # For entities, we take the Weisfeiler-Lehman label
                    else:
                        canonical_walk.append(graph.label_map[hop][n])
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks

    def fit(self, graph, instances):
        """Fit the embedding network based on provided instances.
        
        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using `rdflib_to_kg`.

        instances: array-like
            The instances for which an embedding will be created. It important
            to note that the test instances should be passed to the fit method
            as well. Due to RDF2Vec being unsupervised, there is no 
            label leakage.
        -------
        """
        if self.wl:
            graph.weisfeiler_lehman(iterations=self.wl_iterations)

        all_walks = []
        for i, instance in tqdm.tqdm(enumerate(instances)):
            if self.wl:
                walks =  self._extract_wl_walks(graph, Vertex(str(instance)))
            else:
                walks =  self._extract_random_walks(graph, 
                                                    Vertex(str(instance)))
            all_walks += list(walks)

        print('Extracted {} walks for {} instances!'.format(len(all_walks),
                                                            len(instances)))
        sentences = [list(map(str, x)) for x in all_walks]

        self.model = Word2Vec(sentences, size=self.vector_size, 
                              window=self.window, workers=self.n_jobs, 
                              sg=self.sg, iter=self.max_iter, 
                              negative=self.negative, 
                              min_count=self.min_count, seed=42)


    def transform(self, graph, instances):
        """Construct a feature vector for the provided instances.

        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using `rdflib_to_kg`.

        instances: array-like
            The instances for which an embedding will be created. These 
            instances must have been passed to the fit method as well,
            or their embedding will not exist in the model vocabulary.

        Returns
        -------
        embeddings: array-like
            The embeddings of the provided instances.
        """
        check_is_fitted(self, ['model'])

        feature_vectors = []
        for instance in instances:
            feature_vectors.append(self.model.wv.get_vector(str(instance)))
        return feature_vectors


    def fit_transform(self, graph, instances):
        """First apply fit to create a Word2Vec model and then generate
        embeddings for the provided instances.

        Parameters
        ----------
        graphs: graph.KnowledgeGraph
            The graph from which we will extract neighborhoods for the
            provided instances. You can create a `graph.KnowledgeGraph` object
            from an `rdflib.Graph` object by using `rdflib_to_kg`.

        instances: array-like
            The instances for which an embedding will be created. 

        Returns
        -------
        embeddings: array-like
            The embeddings of the provided instances.
        """
        self.fit(graph, instances)
        return self.transform(graph, instances)
