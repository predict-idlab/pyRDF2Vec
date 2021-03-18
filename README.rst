
.. raw:: html

   <p align="center">
       <img width="100%" src="assets/embeddings.svg">
   </p>
   <p align="center">
       <a href="https://www.ugent.be/ea/idlab/en">
           <img src="assets/imec-idlab.svg" alt="Logo" width=350>
       </a>
   </p>
   <p align="center">
       <a href="https://pypi.org/project/pyrdf2vec/">
           <img src="https://img.shields.io/pypi/pyversions/pyrdf2vec.svg" alt="Python Versions">
       </a>
       <a href="https://pypi.org/project/pyrdf2vec">
           <img src="https://img.shields.io/pypi/v/pyrdf2vec?logo=pypi&color=1082C2" alt="Downloads">
       </a>
       <a href="https://pypi.org/project/pyrdf2vec">
           <img src="https://img.shields.io/pypi/dm/pyrdf2vec.svg?logo=pypi&color=1082C2" alt="Version">
       </a>
       <a href="https://github.com/IBCNServices/pyRDF2Vec/blob/master/LICENSE">
           <img src="https://img.shields.io/github/license/IBCNServices/pyRDF2vec" alt="License">
       </a>
   </p>
   <p align="center">
       <a href="https://github.com/IBCNServices/pyRDF2Vec/actions">
           <img src="https://github.com/IBCNServices/pyRDF2Vec/workflows/CI/badge.svg" alt="Actions Status">
       </a>
        <a href="https://pyrdf2vec.readthedocs.io/en/latest/?badge=latest">
           <img src="https://readthedocs.org/projects/pyrdf2vec/badge/?version=latest" alt="Documentation Status">
       </a>
        <a href="https://codecov.io/gh/IBCNServices/pyRDF2Vec?branch=master">
           <img src="https://codecov.io/gh/IBCNServices/pyRDF2Vec/coverage.svg?branch=master&precision=2" alt="Coverage Status">
       </a>
       <a href="https://github.com/psf/black">
           <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
       </a>
   </p>
   <p align="center">Python implementation and extension of <a href="http://rdf2vec.org/">RDF2Vec</a> <b>to create a 2D feature matrix from a Knowledge Graph</b> for downstream ML tasks.<p>

--------------

.. rdf2vec-begin

What is RDF2Vec?
----------------

RDF2Vec is an unsupervised technique that builds further on
`Word2Vec <https://en.wikipedia.org/wiki/Word2vec>`__, where an
embedding is learned per word, in two ways:

1. **the word based on its context**: Continuous Bag-of-Words (CBOW);
2. **the context based on a word**: Skip-Gram (SG).

To create this embedding, RDF2Vec first creates "sentences" which can be
fed to Word2Vec by extracting walks of a certain depth from a Knowledge
Graph.

This repository contains an implementation of the algorithm in "RDF2Vec:
RDF Graph Embeddings and Their Applications" by Petar Ristoski, Jessica
Rosati, Tommaso Di Noia, Renato De Leone, Heiko Paulheim
(`[paper] <http://semantic-web-journal.net/content/rdf2vec-rdf-graph-embeddings-and-their-applications-0>`__
`[original
code] <http://data.dws.informatik.uni-mannheim.de/rdf2vec/>`__).

.. rdf2vec-end
.. getting-started-begin

Getting Started
---------------

For most uses-cases, here is how ``pyRDF2Vec`` should be used to generate
embeddings and get literals from a given Knowledge Graph (KG) and entities:

.. code:: python

   import pandas as pd

   from pyrdf2vec import RDF2VecTransformer
   from pyrdf2vec.embedders import Word2Vec
   from pyrdf2vec.graphs import KG
   from pyrdf2vec.walkers import RandomWalker

   data = pd.read_csv("samples/countries-cities/entities.tsv", sep="\t")

   embeddings, literals = RDF2VecTransformer(
       Word2Vec(iter=10),
       walkers=[RandomWalker(4, 10, n_jobs=2)],
       # verbose=1
   ).fit_transform(
       KG(
           "https://dbpedia.org/sparql",
           skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
           literals=[
               [
                   "http://dbpedia.org/ontology/wikiPageWikiLink",
                   "http://www.w3.orgb/2004/02/skos/core#prefLabel",
               ],
               ["http://dbpedia.org/ontology/humanDevelopmentIndex"],
           ],
       ),
       [entity for entity in data["location"]],
   )

In a more concrete way, we provide a blog post with a tutorial on how to use
``pyRDF2Vec`` `here
<https://towardsdatascience.com/how-to-create-representations-of-entities-in-a-knowledge-graph-using-pyrdf2vec-82e44dad1a0>`__.

**NOTE:** this blog uses some an older version of ``pyRDF2Vec``, some commands
need be to adapted

Installation
~~~~~~~~~~~~

``pyRDF2Vec`` can be installed in two ways:

1. from `PyPI <https://pypi.org/project/pyrdf2vec>`__ using ``pip``:

.. code:: bash

   pip install pyRDF2vec

2. from any compatible Python dependency manager (e.g., ``poetry``):

.. code:: bash

   poetry add pyRDF2vec

Introduction
~~~~~~~~~~~~

To create embeddings for a list of entities, there are two steps to do
beforehand:

1. **use a KG**;
2. **define a walking strategy**.

For more elaborate examples, check the `examples
<https://github.com/IBCNServices/pyRDF2Vec/blob/master/examples>`__ folder.

If no sampling strategy is defined, ``UniformSampler`` is used. Similarly for
the embedding techniques, ``Word2Vec`` is used by default.

Use a Knowledge Graph
~~~~~~~~~~~~~~~~~~~~~

To use a KG, you can initialize it in three ways:

1. **from a endpoint server using SPARQL**:

.. code:: python

   from pyrdf2vec.graphs import KG

   # Defined the DBpedia endpoint server, as well as a set of predicates to
   # exclude from this KG and a list of predicate chains to fetch the literals.
   KG(
       "https://dbpedia.org/sparql",
       skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
       literals=[
           [
               "http://dbpedia.org/ontology/wikiPageWikiLink",
               "http://www.w3.orgb/2004/02/skos/core#prefLabel",
           ],
           ["http://dbpedia.org/ontology/humanDevelopmentIndex"],
        ],
    ),

2. **from a file using RDFLib**:

.. code:: python

   from pyrdf2vec.graphs import KG

   # Defined the MUTAG KG, as well as a set of predicates to exclude from
   # this KG and a list of predicate chains to get the literals.
   KG(
       "samples/mutag/mutag.owl",
       skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
       literals=[
           [
               "http://dl-learner.org/carcinogenesis#hasBond",
               "http://dl-learner.org/carcinogenesis#inBond",
           ],
           [
               "http://dl-learner.org/carcinogenesis#hasAtom",
               "http://dl-learner.org/carcinogenesis#charge",
           ],
       ],
   ),

3. **from scratch**:

.. code:: python

   from pyrdf2vec.graphs import KG, Vertex

   GRAPH = [
      ["Alice", "knows", "Bob"],
      ["Alice", "knows", "Dean"],
      ["Dean", "loves", "Alice"],
   ]
   URL = "http://pyRDF2Vec"
   CUSTOM_KG = KG()

   for row in GRAPH:
      subj = Vertex(f"{URL}#{row[0]}")
      obj = Vertex((f"{URL}#{row[2]}"))
      pred = Vertex((f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj)
      CUSTOM_KG.add_walk(subj, pred, obj)

Define Walking Strategies With Their Sampling Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All supported walking strategies can be found on the
`Wiki
page <https://github.com/IBCNServices/pyRDF2Vec/wiki/Walking-Strategies>`__.

As the number of walks grows exponentially in function of the depth,
exhaustively extracting all walks quickly becomes infeasible for larger
Knowledge Graphs. In order to avoid this issue, `sampling strategies
<http://www.heikopaulheim.com/docs/wims2017.pdf>`__ can be applied. These will
extract a fixed maximum number of walks per entity and sampling the walks
according to a certain metric.

For example, if one wants to extract a maximum of 10 walks of a maximum depth
of 4 for each entity using the Random walking strategy and Page Rank sampling
strategy (**SEE:** the `Wiki page
<https://github.com/IBCNServices/pyRDF2Vec/wiki/Sampling-Strategies>`__ for
other sampling strategies), the following code snippet can be used:

.. code:: python

   from pyrdf2vec.samplers import PageRankSampler
   from pyrdf2vec.walkers import RandomWalker

   walkers = [RandomWalker(4, 10, PageRankSampler())]

.. getting-started-end

Documentation
-------------

For more information on how to use ``pyRDF2Vec``, `visit our online documentation
<https://pyrdf2vec.readthedocs.io/en/latest/>`__ which is automatically updated
with the latest version of the ``master`` branch.

From then on, you will be able to learn more about the use of the
modules as well as their functions available to you.

Contributions
-------------

Your help in the development of ``pyRDF2Vec`` is more than welcome. In order to
better understand how you can help either through pull requests and/or issues,
please take a look at the `CONTRIBUTING
<https://github.com/IBCNServices/pyRDF2Vec/blob/master/CONTRIBUTING.rst>`__
file.

FAQ
---
How to Ensure the Generation of Similar Embeddings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyRDF2Vec``'s walking strategies, sampling strategies and Word2Vec work with
randomness. To get reproducible embeddings, you firstly need to **use a seed** to
ensure determinism:

.. code:: bash

   PYTHONHASHSEED=42 python foo.py

Added to this, you must **also specify a random state** to the walking strategy
which will implicitly use it for the sampling strategy:

.. code:: python

   from pyrdf2vec.walkers import RandomWalker

   RandomWalker(2, None, random_state=42)

**NOTE:** the ``PYTHONHASHSEED`` (e.g., 42) is to ensure determinism.

Finally, to ensure random determinism for Word2Vec, you must **specify a single
worker** to Word2Vec:

.. code:: python

   from pyrdf2vec.embedders import Word2Vec

   Word2Vec(workers=1)

**NOTE:** using the ``n_jobs`` and ``mul_req`` parameters does not affect the
random determinism.

Why the extraction time of walks is faster if ``max_walks=None``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, the BFS function (using the Breadth-first search algorithm) is used
when ``max_walks=None`` which is significantly faster than the DFS function
(using the Depth-first search algorithm).

We hope that this algorithmic complexity issue will be solved for the next
release of ``pyRDf2Vec``

Referencing
-----------

If you use ``pyRDF2Vec`` in a scholarly article, we would appreciate a
citation:

.. code:: bibtex

   @inproceedings{pyrdf2vec,
     author       = {Gilles Vandewiele and Bram Steenwinckel and Terencio Agozzino
                     and Michael Weyns and Pieter Bonte and Femke Ongenae
                     and Filip De Turck},
     title        = {{pyRDF2Vec: Python Implementation and Extension of RDF2Vec}},
     organization = {IDLab},
     year         = {2020},
     url          = {https://github.com/IBCNServices/pyRDF2Vec}
   }
