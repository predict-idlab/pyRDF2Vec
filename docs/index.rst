===========
pyRDFVec
===========

pyRDFVec is a Python implementation and extension of `RDF2Vec <RDF2Vec_>`_ **to
create a 2D feature matrix from a knowledge graph** for downstream Machine
Learning tasks.

.. _RDF2Vec: http://rdf2vec.org/

RDF2Vec
-------

RDF2Vec is an unsupervised technique that builds further on Word2Vec, where an
embedding is learned per word, in two ways:

#.  the word based on its context: Continuous Bag-of-Words (CBOW);
#.  the context based on a word: Skip-Gram (SG).

To create this embedding, RDF2Vec first creates "sentences" which can be fed to
Word2Vec by extracting walks of a certain depth from a knowledge graph.

This repository contains an implementation of the algorithm in "RDF2Vec: RDF
Graph Embeddings and Their Applications" by Petar Ristoski, Jessica Rosati,
Tommaso Di Noia, Renato De Leone, Heiko Paulheim ([paper] [original code]).

Installation
------------

It's as simple as that:

.. code-block:: bash

   pip install pyRDF2vec

Getting Started
---------------

To create embeddings for a list of entities, there are two steps to do
beforehand:

#.  **create a Knowledge Graph object**;
#.  **define a walking strategy**.

For a more elaborate example, check at the
`example.py <example_>`_ file.

.. _example: https://github.com/IBCNServices/pyRDF2Vec/blob/master/example.py/

.. code-block:: bash

  PYTHONHASHSEED=42 python3 rdf2vec/example.py

**NOTE:** the `PYTHONHASHSEED` (*e.g.,* 42) is to ensure determinism.

To create a Knowledge Graph object, you can initialize it in several ways:

.. code-block:: python

   from rdf2vec.converters import endpoint_to_kg, rdflib_to_kg

   # Define the label predicates, all triples with these predicates
   # will be excluded from the graph
   label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]

   # Create a Knowledge Graph from rdflib
   kg = rdflib_to_kg("samples/mutag.owl", label_predicates=label_predicates)

   # Create a Knowledge Graph from a SPARQL endpoint
   kg = endpoint_to_kg("http://localhost:5820/db/query?query=", label_predicates=label_predicates)

To define a walking strategy, start by choosing one available on the `Wiki page
<Wiki page_>`_.

.. _Wiki page: https://github.com/IBCNServices/pyRDF2Vec/wiki/Walking-Strategies

For example, the definition of the Random walking strategy with a depth of 4 is
implemented as follows:

.. code-block:: python

   from rdf2vec.walkers import RandomWalker

   random_walker = RandomWalker(4, float("inf"))


Finally, the creation of embeddings for a list of entities simply goes like
this:

.. code-block:: python

   from rdf2vec import RDF2VecTransformer

   transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
   # Entities should be a list of URIs that can be found in the Knowledge Graph
   embeddings = transformer.fit_transform(kg, entities)

More information
================

.. toctree::
   :maxdepth: 1

* :ref:`modindex`
