0.2.3 (2021-06-09)
-------------------

🚀 Features
^^^^^^^^^^^^^

- Add the ``skip_verify`` attribute to the ``KG`` class to skip or not the
  verification of the entity existence with remote Knowledge Graphs (default to
  ``skip_verify=False``).
- Add ``WideSampler`` as a new sampling strategy.
- Add ``SplitWalker`` as a new walking strategy.

Fixed
^^^^^

- Fix the installation dependencies with ``poetry``.
- Fix the cache memory for local Knowledge Graphs.
- Fix validation URL for remote Knowledge Graphs.
- Fix the ``HALKWalker`` walking strategy.
- Fix the DFS algorithm of ``RandomWalker`` and ``CommunityWalker`` to return
  duplicate walks and prevent a different number of walks for the entities.
- Fix the walk extraction with the ``with_reverse`` parameter for the different
  walking strategies.

Added
^^^^^

- Add the ``_post_extract`` private method in the ``Walker`` class for a post
  processing of walks by a walking strategy.

Changed
^^^^^^^

- Replace the default minimum frequency thresholds of a hop to keep with
  ``HALKWalker`` (0.001 -> 0.01).
- Drop support for Python 3.7.0
- Remove ``negative=20`` and ``vector_size=500`` for Word2Vec.

0.2.2 (2021-04-02)
-------------------

🚀 Features
^^^^^^^^^^^^^

- Add a first support of FastText as embedding technique.

Fixed
^^^^^

- Fix the ``size`` hyperparameter by ``vector_size`` of the default dictionary
  in the ``Word2Vec`` class.
- Fix random determinism with walking strategies.
- Fix the calculation of walks for duplicate entities in a file.
- Fix the total recovery of entities, walks, literals and embeddings of a model
  after multiple online learning.

Added
^^^^^

- Add the ``_update`` private method in the ``RDF2VecTransformer`` class.
- Add the ``md5_bytes`` attribute in the ``CommuniWalker``, ``HALKWalker``,
  ``RandomWalker``, and ``WLWalker`` classes to hash or not an object in MD5
  and with how many bytes to keep.

Changed
^^^^^^^

- Replace the ``extract`` method in the ``Walker`` to returns a list of
  entities with their walks instead of a list of walks.

0.2.1 (2021-03-22)
-------------------

Fixed
^^^^^
- Fix the issue with ``nest-asyncio`` as dependency.

0.2.0 (2021-03-20)
-------------------

🚀 Features
^^^^^^^^^^^^^

- Add support for Python 3.9
- Add the ``cache`` (default to ``cachetools.TTLCache(maxsize=1024,
  ttl=1200)``) attribute to the ``KG`` class to significantly speed up the walks
  extraction through caching.
- Add the ``is_update`` (default to ``False``) hyper-parameter in the ``fit``
  method of the ``Embedder`` and ``Word2Vec`` classes to update an existing
  vocabulary.
- Add the ``literals`` (default to ``[]``) attribute in the ``KG`` class to
  support a basic literal extraction.
- Add the ``mul_req`` (default to ``False``) attribute to the ``KG`` class to
  speed up the extraction of walks and literals for remote Knowledge Graph by
  sending asynchronous requests.
- Add the ``n_jobs`` (default to ``None``) attribute to the ``Walker`` class
  to speed up the extraction of walks with multiprocessing.
- Add the ``random_state`` (default to ``None``) parameter for the ``Walker``
  class to handle better random determinism with walking and sampling
  strategies.
- Add the ``verbose`` (default to ``0``) attribute to the
  ``RDF2VecTransformer`` class to display useful debugging information and to
  measure the time of extraction, fit and generation of embeddings and
  literals.
- Add the ``with_reverse`` (default to ``False``) parameter for the ``Walker``
  class to generate more walks and improve the accuracy with ``Word2Vec``, by
  including the parents of the entities in the walks.
- Add the possibility to do online learning of a model with the ``load`` and
  the ``save`` methods in the ``RDF2VecTransformer`` class.
- Add the validators for class parameter attributes.

Added
^^^^^

- Add the ``Connector`` generic class to simplify the implementation of new
  connectors.
- Add the ``SPARQLConnector`` class to delegate the connection part to the
  SPARQL endpoint server.
- Add the ``Vertex`` class in a slot to reduce RAM usage.
- Add the ``WalkerNotSupported`` and ``SamplerNotSupported`` exceptions in the
  ``Walker`` and ``Sampler`` classes when a walking strategy and a sampling
  strategy is not supported.
- Add the ``_cast_literals`` private method to the ``KG`` class to convert the
  raw literals of an entity according to their real types.
- Add the ``_embeddings``, ``_entities``, ``_literals``, and ``_walks``,
  attributes in the ``RDF2VecTransformer`` class to be able to get all the
  embeddings, entities, literals, and walks after the online training of a
  model.
- Add the ``_fill_hops`` private method in the ``KG`` class to fill the entity
  hops in cache when ``mul_req=True`` is provided for a remote Knowledge Graph.
- Add the ``_get_hops`` private method in the ``KG`` class to get the hops of a
  vertex for a local Knowledge Graph.
- Add the ``_is_support_remote`` (default to ``False``) private attribute in
  the ``Walker`` and ``Sampler`` classes to restrict the use of walking and
  sampling strategies for some remote/local Knowledge Graph.
- Add the ``_res2hops`` private method in the ``KG`` class to convert a JSON
  response from a SPARQL endpoint server to hops.
- Add the ``add_walk`` method to the ``KG`` class to simplify the addition of
  walk in a Knowledge Graph.
- Add the `attr <https://github.com/python-attrs/attrs>`__ decorator for all
  classes.
- Add the ``examples/online-training`` and ``examples/literals`` files to
  illustrate the use of online training and literals with ``pyRDF2Vec``.
- Add the ``fetch_hops`` method to the ``KG`` class to fetch to get the hops of
  a vertex on a remote Knowledge Graph.
- Add the ``get_pliterals`` method to the ``KG`` class to gets the literals for
  an entity and a local KG based on a chain of predicates.
- Add the ``get_walks`` method in the ``RDF2VecTransformer`` class to get the
  walks of a given entities in a Knowledge Graph.
- Add the ``get_weights`` method in the ``Sampler`` class to get the hops weights.
- Add the ``pyrdf2vec.typings`` file to contains the aliases of the most
  commonly used typing with `mypy <https://github.com/python/mypy>`__.

Fixed
^^^^^

- Fix the ``get_weight`` method in the ``PageRankSampler`` to raise an error if
  the method is called before the ``fit`` method.
- Fix the ``remove_edge`` method of the ``KG`` class to also remove the edge of
  a children for a parent node.
- Fix the addition of predicate in memory for remote Knowledge Graphs.
- Fix the initialization of the ``_counts`` dictionary with the
  ``PredFreqSampler`` and ``ObjPredFreqSampler`` classes.

Changed
^^^^^^^

- Remove support for Python 3.6
- Remove the ``_get_shops`` and ``_get_rhops`` functions in the ``KG`` class.
- Remove the ``id`` attribute of the ``Vertex`` class.
- Remove the ``print_walks`` method of the ``Walker`` class.
- Remove the ``read_file`` method in the ``KG`` class.
- Remove the ``visualise`` method in the ``KG`` class.
- Replace the ``HalkWalker`` class by ``HALKWalker``.
- Replace the ``SPARQLWrapper`` library in favor of using ``requests`` for
  synchronous requests and ``aiohttp`` for asynchronous requests.
- Replace the ``WeisfeilerLehmanWalker`` class by ``WLWalker``.
- Replaces the ``add_edge``, ``add_vertex``, and ``remove_edge`` methods in the
  ``KG`` class to return a boolean value indicating that the addition/removal
  of an edge/vertex has been performed.
- Replace the ``depth`` parameter with ``max_depth`` for the ``Walker`` class.
- Replace the ``extract_random_community_walks``,
  ``extract_random_community_walks_bfs``, and
  ``extract_random_community_walks_dfs`` methods in the ``CommunityWalker``
  class by ``extract_walks``, ``_bfs``, and ``_dfs`` methods.
- Replace the ``extract_random_walks``, ``extract_random_walks_bfs``, and
  ``extract_random_walks_dfs`` methods in the ``RandomWalker`` class by
  ``extract_walks``, ``_bfs``, and ``_dfs`` methods.
- Replace the ``file_type`` attribute in the ``KG`` class by ``fmt``.
- Replace the ``get_inv_neighbors`` method in the ``KG`` class by a
  ``is_reverse`` (default to ``False``) parameter in the ``get_neighbors``
  method.
- Replace the ``initialize`` method in the ``Sampler`` class by the use of ``@property``.
- Replace the ``is_remote`` parameter in the ``KG`` class for automatic link
  detection based on the http and https prefix.
- Replace the ``last`` parameter with ``is_last_depth`` in the
  ``sample_neighbor`` method of the ``Sampler`` class.
- Replace the ``label_predicates`` attribute in the ``KG`` class by
  ``skip_predicates`` and now use a set instead of a list.
- Replace the ``pyrdf2vec.graphs.kg.Vertex`` class with
  ``pyrdf2vec.graphs.Vertex``.
- Replace the ``fit_transform`` and ``transform`` functions in the
  ``RDF2VecTransformer`` class to return a tuple containing the list of
  embeddings and literals.
- Replace the default embedding technique in the ``RDF2VecTransformer`` class
  for ``Word2Vec``.
- Replace the default hyper-parameters of the ``Word2Vec`` class to
  ``size=500``, ``min_count=0``, and ``negative=20``.
- Replace the default list of walkers in the ``RDF2VecTransformer`` class to
  ``[RandomWalker(2)]``.

0.1.0 (2020-11-02)
-------------------

Features
^^^^^^^^

- Add a ``verbose`` (default to ``False``) hyper-parameter for the ``fit`` method.
- Add basic support for remote Knowledge Graphs through SPARQL endpoint.
- Add configuration for Embedding Techniques through the ``Embedder`` abstract class
  (currently only Word2Vec is included).
- Add online documentation.
- Add sampling strategies (default to ``UniformSampler``) from Cochez et al. to
  better deal with larger Knowledge Graphs.
- Add static typing for methods.
- Add support for Python 3.6 and 3.7.
- Add the `Google Style Python Docstrings
  <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__.
- Add the ``extract_random_walks_dfs`` and ``extract_random_walks_bfs`` methods
  for the ``RamdomWalker`` class.
- Add the ``get_hops`` method along with the private ``_get_rhops`` and
  ``_get_shops`` methods in the ``KG`` class.
- Add three examples (``examples/countries.py``, ``examples/mutag.py`` and
  ``examples/samplers.py``) for ``pyRDF2vec``.

Changed
^^^^^^^

- Replace ``graph`` for ``kg`` in the ``fit`` and ``fit_transform`` methods of
  the ``RDF2VecTransformer`` class.
- Replace ``instance`` for ``entities`` in the ``transform``
  and ``fit_transform`` methods of the ``RDF2VecTransformer`` class.
- Replace default values of hyper-parameters of Word2Vec to match with the
  `default ones
  <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`__
  of the ``gensim`` implementation.
- Replace the ``KnowledgeGraph`` class for ``KG``.
- Replace the ``Walker`` class to be abstract.
- Replace the ``_rdf2vec.py`` file for ``rdf2vec.py``.
- Replace the ``extract_random_community_walks`` method in the
  ``CommunityWalker`` to be private.
- Replace the ``extract`` methods in ``walkers`` to be private.
- Replace the ``graph.py`` file for ``graphs/kg.py``.
- Replace the ``rdf2vec`` module for ``pyrdf2vec``.
- Replace the ``sample_neighbor`` method of the ``sampler`` class by
  ``sample_hop``.
- Replace the imec licence for an MIT licence.
- Remove ``graph`` hyper-parameter in the ``transform`` method of the ``RDF2VecTransformer`` class.
- Remove hyper-parameters of ``RDF2VecTransformer`` for ``embedder`` and ``walkers`` ones.
- Remove the ``WildcardWalker`` walking strategy.
- Remove the ``converter.py`` file.
- Remove the ``create_kg``, ``endpoint_to_kg``, ``rdflib_to_kg`` functions
  for the ``location``, ``file_type``, ``is_remote`` hyper-parameters in
  ``KG`` with the ``read_file`` private method.
- Replace ``Vertex.vertex_count`` for ``itertools.count`` in the ``Vertex`` class.
