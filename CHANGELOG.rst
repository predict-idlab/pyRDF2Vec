0.1.0 (2020-10-30)
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
- Replace the imec licence for an MIT licence.
- Remove ``graph`` hyper-parameter in the ``transform`` method of the ``RDF2VecTransformer`` class.
- Remove hyper-parameters of ``RDF2VecTransformer`` for ``embedder`` and ``walkers`` ones.
- Remove the ``WildcardWalker`` walking strategy.
- Remove the ``converter.py`` file.
- Remove the ``create_kg``, ``endpoint_to_kg``, ``rdflib_to_kg`` functions
  for the ``location``, ``file_type``, ``is_remote`` hyper-parameters in
  ``KG`` with the ``read_file`` private method.
- Replace ``Vertex.vertex_count`` for ``itertools.count`` in the ``Vertex`` class.
