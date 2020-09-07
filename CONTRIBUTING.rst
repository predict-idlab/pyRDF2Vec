Contributing
============

Thank you for wanting to bring your part to the ``pyRDF2Vec``
development. It's always heartwarming to see other people putting their
stone in the building of something much bigger.

This document is created to facilitate your contributions by sharing the
necessary knowledge and expectations. Feel free to open a pull request
if you have implemented half a feature and/or an issue if you are stuck.
We will be happy to help you and thank you for your work!

The project being new, there are many opportunities for contributions:

1. `Add a new embedding technique <#add-a-new-embedding-technique>`__:
   the field of natural language processing is advancing rapidly. As
   such, many of the techniques used in the original implementation of
   RDF2Vec were outdated.
2. `Add a new walking strategy <#add-a-new-walking-strategy>`__: add
   your own custom walking algorithm.
3. `Add a new sampling strategy <#add-a-new-sampling-strategy>`__:
   heuristically sample when extracting walks for scalability.
4. `Improve the online
   documentation <#improve-the-online-documentation>`__: correcting
   spelling mistakes and using better sentence structures may help to
   better understand the use of ``pyRDF2Vec``.
5. `Submit your bugs and
   suggestions <#submit-your-bugs-and-suggestions>`__: reproduce the bug
   you encountered with ``pyRDF2Vec`` and/or suggest your ideas for the
   development of ``pyRDF2Vec``, would help us to improve this library.
6. `Improve the code <#improve-the-code>`__: whether by refactoring or
   optimizing the complexity of certain functions, if an implementation
   seems not adequate to you, do not hesitate to modify it by opening a
   pull request, or to inform us by opening an issue.

Each of the following sub-sections will give you more information on the
opportunities listed below.

--------------

Add a new embedding technique
-----------------------------

Currently, ``pyRDF2Vec`` does not have a generic implementation. Only
`Word2Vec <https://en.wikipedia.org/wiki/Word2vec>`__ is implemented,
and other more powerful embedding techniques are available (*e.g.,*
`BERT <https://en.wikipedia.org/wiki/BERT_(language_model)>`__,
`fasttext <https://fasttext.cc/>`__,
`GloVe <https://nlp.stanford.edu/projects/glove/>`__).

Before adding a new embedding technique, it is important to implement an
adequate architecture that must be discussed in an issue.

A good architecture should allow a user to easily choose a embedding
technique (*e.g.,* BERT) with a walking strategy (*e.g.,*
Weisfeler-Lehman) and a sampling strategy.

Add a new walking strategy
--------------------------

To add your own walking strategy, 6 steps are essential:

1. install the dependencies: before you can install the dependencies of ``pyRDF2Vec``, you must first make
   sure you have ``poetry`` to install:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec``:

.. code:: bash

   poetry install

2. create your walker (*e.g.,* ``foo.py``) in ``pyrdf2vec/walkers``;
3. import your walker at the end of the
   ``pyrdf2vec/walkers/__init__.py`` file:

.. code:: python

   from .walker import *
   ...
   from .wildcard import *
   from .foo import *

4. in your walker's class, extend the
   `Walker <https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/walkers/walker.py>`__
   class and implement at least the ``extract(self, graph, instances)``
   function:

.. code:: python3

   class FooWalker(Walker):
       """Defines the foo walking strategy.

       Attributes:
           depth (int): The depth per entity.
           walks_per_graph (float): The maximum number of walks per entity.

       """

       def __init__(self, depth, walks_per_graph):
           super().__init__(depth, walks_per_graph)

       def extract(self, graph, instances):
           """Extracts walks rooted at the provided instances which are then each
           transformed into a numerical representation.

           Args:
               graph (graph.KnowledgeGraph): The knowledge graph.
                   The graph from which the neighborhoods are extracted for the
                   provided instances.
               instances (array-like): The instances to extract the knowledge graph.

           Returns:
               set: The 2D matrix with its:
                   number of rows equal to the number of provided instances;
                   number of column equal to the embedding size.

           """
           # TODO: to be implemented

**NOTE:** don't forget to update the docstring of your walker.

5. create the unit tests (*e.g.,* in the ``test_foo.py`` file) related
   to your walker in ``tests/walkers``:

.. code:: python3

   import random

   import rdflib

   from pyrdf2vec.converters import rdflib_to_kg
   from pyrdf2vec.walkers import FooWalker

   LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
   KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


   def generate_entities():
       return [
           rdflib.URIRef(
               f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
           )
           for _ in range(random.randint(0, 200))
       ]


   class TestFooWalker:
       def test_extract(self):
           canonical_walks = FooWalker(4, float("inf")).extract(
               KG, generate_entities()
           )
           assert type(canonical_walks) == set

6. run unit tests and check that the style of code is still correct:

.. code:: bash

   tox -e lint
   tox -e tests

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Add a new sampling strategy
---------------------------

**COMING SOON**

Adding your own sampling strategy is similar to adding a walking
strategy:

1. Create a file in ``pyrdf2vec/samplers`` and add to
   ``pyrdf2vec/samplers/__init__.py``.
2. Extend the
   `Sampler <https://github.com/IBCNServices/pyRDF2Vec/blob/samplers/rdf2vec/samplers/sampler.py>`__
   class and implement the ``get_weights(self, hop)`` function. This
   should return a score for the provided ``hop``, where a higher score
   means it is more useful to include this hop in the walks. The
   returned score will be converted to a probability which is used to
   sample to next hop in a walk being extracted by a walker.

Improve the online documentation
--------------------------------

The `online documentation of
pyRDF2Vec <https://pyrdf2vec.readthedocs.io/en/latest/>`__ is hosted on
`Read the Docs <https://readthedocs.org/>`__. To generate this online
documentation, we use:

-  `Sphinx <https://www.sphinx-doc.org/en/master/>`__ as a Python
   documentation generator;
-  `Google style
   docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`__:
   as a docstring writing convention.

Before you can modify the ``pyRDF2Vec`` documentation with, you must
first make sure you have ``poetry`` to install:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to the documentation:

.. code:: bash

   poetry install -E docs

Once you have modified what needed to be modified in the documentation
(available in the ``docs`` folder), it is important to generate this
documentation locally with ``tox``, to ensure that your modification has
been taken into account:

.. code:: bash

   tox -e docs

As the documentation is updated, check that the changes made are correct
with your web browser:

.. code:: bash

   $BROWSER _build/html/index.html

Everything is well rendered? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Submit your bugs and suggestions
--------------------------------

Feel free to `open an
issue <https://github.com/IBCNServices/pyRDF2Vec/issues/new>`__ in case
something is not working as expected, or if you have any
questions/suggestions.

In order to help you out as good as possible:

-  **describe the question/problem as precise as possible**;
-  **inform your operating system**;
-  **provide an example of minimal work with sample data (if possible)
   to reproduce the bug**.

Improve the code
----------------

The refactoring and optimization of code complexity is an art that must
be necessary to facilitate future contributions of ``pyRDF2Vec``.

Before you can modify the ``pyRDF2Vec`` code, you must first make sure
you have ``poetry`` to install:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies:

.. code:: bash

   poetry install

Once you have made your modifications, check that the style of the code
is still respected:

.. code:: bash

   tox -e lint

Then, launch the unit tests which can take several minutes:

.. code:: bash

   tox -e tests

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!
