
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

Installation
~~~~~~~~~~~~

``pyRDF2Vec`` can be installed in two ways:

1. from `PyPI <https://pypi.org/project/pyrdf2vec>`__ using ``pip``:

.. code:: bash

   pip install pyRDF2vec

2. from any compatible Python dependency manager (*e.g.,* ``poetry``):

.. code:: bash

   poetry add pyRDF2vec

Introduction
~~~~~~~~~~~~

To create embeddings for a list of entities, there are two steps to do
beforehand:

1. **create a Knowledge Graph object**;
2. **define a walking strategy**.

For a more elaborate example, check at the
`example.py <https://github.com/IBCNServices/pyRDF2Vec/blob/master/example.py>`__
file:

.. code:: bash

   PYTHONHASHSEED=42 python3 example.py

**NOTE:** the ``PYTHONHASHSEED`` (*e.g.,* 42) is to ensure determinism.

Create a Knowledge Graph Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a Knowledge Graph object, you can initialize it in two ways.

1. **from a file using RDFlib**:

.. code:: python

   from pyrdf2vec.graphs import KG

   # Define the label predicates, all triples with these predicates
   # will be excluded from the graph
   label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]
   kg = KG(location="samples/mutag/mutag.owl", label_predicates=label_predicates)

2. **from a server using SPARQL**:

.. code:: python

   from pyrdf2vec.graphs import KG

   kg = KG(location="https://dbpedia.org/sparql", is_remote=True)

Define Walking Strategies With Their Sampling Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All supported walking strategies can be found on the
`Wiki
page <https://github.com/IBCNServices/pyRDF2Vec/wiki/Walking-Strategies>`__.

As the number of walks grows exponentially in function of the depth,
exhaustively extracting all walks quickly becomes infeasible for larger
Knowledge Graphs. In order to circumvent this issue, `sampling strategies
<http://www.heikopaulheim.com/docs/wims2017.pdf>`__ can be applied. These will
extract a fixed maximum number of walks per entity. The walks are sampled
according to a certain metric.

For example, if one wants to extract a maximum of 5 walks of depth 4 for each
entity using the Random walking strategy and Uniform sampling strategy (**SEE:**
the `Wiki page
<https://github.com/IBCNServices/pyRDF2Vec/wiki/Sampling-Strategies>`__ for
other sampling strategies), the following code snippet can be used:

.. code:: python

   from pyrdf2vec.samplers import UniformSampler
   from pyrdf2vec.walkers import RandomWalker

   walkers = [RandomWalker(4, 5, UniformSampler())]

Create Embeddings
~~~~~~~~~~~~~~~~~

Finally, the creation of embeddings for a list of entities simply goes
like this:

.. code:: python

   from pyrdf2vec import RDF2VecTransformer

   transformer = RDF2VecTransformer(walkers=[walkers], sg=1)
   # Entities should be a list of URIs that can be found in the Knowledge Graph
   embeddings = transformer.fit_transform(kg, entities)

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

Referencing
-----------

If you use ``pyRDF2Vec`` in a scholarly article, we would appreciate a
citation:

.. code:: bibtex

   @inproceedings{pyrdf2vec,
     author       = {Gilles Vandewiele and Bram Steenwinckel and Terencio Agozzino
                     and Michael Weyns and Pieter Bonte and Femke Ongenae
                     and Filip De Turck},
     title        = {{pyRDF2Vec: A python library for RDF2Vec}},
     organization = {IDLab},
     year         = {2020},
     url          = {https://github.com/IBCNServices/pyRDF2Vec}
   }
