import rdflib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from gensim.models.word2vec import Word2Vec
import tqdm
import copy
from graph import extract_instance, rdflib_to_kg


class UnknownEntityError(Exception):
    pass


class RDF2VecTransformer():
    """Project random walks or subtrees in graphs into embeddings, suited
    for classification.

    Parameters
    ----------
    vector_size: int
        The dimension of the embeddings

    max_path_depth: int
        The maximum length of the sequence will be 2 * `max_path_depth` + 1

    max_tree_depth: int
        The size of the trees in the sequence. Only used when _type == 'tree'

    _type: str of ['walk' or 'tree']
        How to construct the sequences fed to the embedder.

    Attributes
    ----------

    Example
    ------- 

    """
    def __init__(self, vector_size=500, max_path_depth=4, max_tree_depth=2,
                 _type='walk', walks_per_graph=500, window=5, n_jobs=4,
                 sg=1, max_iter=10, negative=25, min_count=1, wfl_iterations=4):
        if _type not in ['walk', 'wl']:
            raise Exception('_type should be "walk" or "wl"')
        self.vector_size = vector_size
        self.max_path_depth = max_path_depth
        self.max_tree_depth = max_tree_depth
        self._type = _type
        self.walks_per_graph = walks_per_graph
        self.window = window
        self.n_jobs = n_jobs
        self.sg = sg
        self.max_iter = max_iter
        self.negative = negative
        self.min_count = min_count
        self.wfl_iterations = wfl_iterations

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


    def _extract_random_walks(self, graph, label_predicates=[]):
        walks = graph.extract_random_walks(self.max_path_depth*2, max_walks=self.walks_per_graph)

        canonical_walks = set()
        for walk in walks:
            canonical_walk = []
            for hop in walk:
                canonical_walk.append(hop.name.split('/')[-1])

            canonical_walks.add(tuple(canonical_walk))

        return list(canonical_walks)


    def _extract_wl_walks(self, graph, verbose=False, label_predicates=[]):
        """Weisfeiler-Lehman relabeling algorithm, used to calculate the
        corresponding kernel.
        
        Parameters
        ----------
            g (`Graph`): the knowledge graph, mostly first extracted
                         from a larger graph using `extract_instance`
            s_n_to_counter (dict): global mapping function that maps a 
                                   multi-set label to a unique integer
            n_iterations (int): maximum subtree depth
            
        Returns
        -------
            label_mappings (dict): for every subtree depth (or iteration),
                                   the mapping of labels is stored
                                   (key = old, value = new)
        """
        
        # Our resulting label function (a map for each iterations)
        walks = list(graph.extract_random_walks(self.max_path_depth*2, max_walks=self.walks_per_graph))

        # Take a deep copy of our graph, since we are going to relabel its nodes
        # g = copy.deepcopy(g)

        graph.weisfeiler_lehman(iterations=self.wfl_iterations)

        canonical_walks = set()

        for walk in walks:
            canonical_walk = []
            for hop in walk:
                canonical_walk.append(hop.name.split('/')[-1])
            canonical_walks.add(tuple(canonical_walk))

        for n in range(1, self.wfl_iterations + 1):
            for walk in walks:
                canonical_walk = [walk[0].name.split('/')[-1]]
                for hop in walk[1:]:
                    canonical_walk.append(graph.label_map[hop][n])
                canonical_walks.add(tuple(canonical_walk))

        canonical_walks = list(canonical_walks)

        if len(canonical_walks)==0:
            return []
        else:
            walks_ix = np.random.choice(range(len(canonical_walks)), replace=False, 
                                        size=min(len(canonical_walks), self.walks_per_graph))
            return np.array(canonical_walks)[walks_ix]

        

    def fit(self, graphs, label_predicates=[]):
        """ Fit the embedding network based on provided graphs and labels
        
        Parameters
        ----------
        graphs: array-like of `rdflib.Graph`
            The training graphs, which are used to extract random walks or
            subtrees in order to train the embedding model

        labels: array-like
            Not used, since RDF2Vec is unsupervised
        """
        walks = []
        for i, graph in tqdm.tqdm(enumerate(graphs)):
            if self._type == 'wl':
                walks += list(self._extract_wl_walks(graph, label_predicates=label_predicates))
            else:
                walks += list(self._extract_random_walks(graph, label_predicates=label_predicates))

        sentences = [list(map(str, x)) for x in walks]

        self.model = Word2Vec(sentences, size=self.vector_size, window=self.window, 
                              workers=self.n_jobs, sg=self.sg, 
                              iter=self.max_iter, negative=self.negative, 
                              min_count=self.min_count)


    def transform(self, graphs):
        """ Construct a feature vector for each graph

        Parameters
        ----------
        graphs: array-like of `rdflib.Graph`
            The graphs for which we need to calculate a feature vector
        """
        check_is_fitted(self, ['model'])

        feature_vectors = []
        for graph in graphs:
            #if str(graph.root) not in self.model.wv.vocab.keys():
            #    raise UnknownEntityError
            feature_vectors.append(self.model.wv.get_vector(graph.root.name.split('/')[-1]))
        return feature_vectors


    def fit_transform(self, graphs, label_predicates=[]):
        """ First fit the embedding model and then construct feature vectors."""
        self.fit(graphs, label_predicates=label_predicates)
        return self.transform(graphs)
