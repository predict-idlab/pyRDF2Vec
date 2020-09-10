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

Getting Started
---------------

Before reading the sub-section that interests you in detail, here are some
golden rules that you should know before you contribute to ``pyRDF2Vec``:

-  **No contribution is too small:** submit as many fixes for typos and grammar
   bloopers as you can.
-  Whenever possible, **limit each pull request to one change only**.
-  **Add tests and docs for your code:** who better than you to explain and
   test that the code you have implemented works?
-  **Make sure your changes pass our CI:** during each commit several checks to
   verify the style of the code, the unit tests and the documentation are done
   though `tox
   <https://tox.readthedocs.io/en/latest/>`__
   to avoid unpleasant surprises.
-  **Attach a short note to the pull request:** it would help us to better
   understand what you did.
-  **It's up to you how you handle updates to the master branch:** since we
   squash on merge, whether you prefer to rebase on ``master`` or merge
   ``master`` into your branch, do whatever is more comfortable for you.


Changelog
---------

If your change is noteworthy, there needs to be a changelog entry so our users
can learn about it!

To avoid merge conflicts, we use the `towncrier
<https://github.com/twisted/towncrier>`__` package to manage our changelog.
``towncrier`` uses independent files (called *news fragments*) for each pull
request. On release, those news fragments are compiled into our
``CHANGELOG.rst`` file.

You don't need to install ``towncrier`` yourself, you just have to abide by a
few simple rules:

- For each pull request, add a new file into ``changelog.d`` with a filename
  adhering to the ``pr#.(bugfix|doc|feature).rst`` (*e.g.,*
  ``changelog.d/42.feature.rst`` for a non-breaking change that is proposed in
  pull request #42.
- Wrap symbols like modules, functions, or classes into double backticks so
  they are rendered in a ``monospace font``.
- Wrap arguments into asterisks like in docstrings: *these* or *attributes*.
- If you mention functions or other callables, add parentheses at the end of
  their names: ``foo.func()`` or ``Foo.method()``.
  This makes the changelog a lot more readable.
- Prefer simple past tense or constructions with "now".
- If you want to reference multiple issues, copy the news fragment to another
  filename. ``towncrier`` will merge all news fragments with identical
  contents into one entry with multiple links to the respective pull requests.

Example entries:

  .. code-block:: rst

     Added ``wallkers.Foo``.
     This walker is a new walker.

or:

  .. code-block:: rst

     ``fit_transform()`` now can deal with bigger Knowledge Graph Embeddings (KGE).

Conventions
-----------

We try as much as possible to follow Python conventions through the PEP
specifications. Don't be afraid of the list of conventions below. Indeed,
``tox`` and/or the CI will alert you and/or format your code for you if needed.

Here are the conventions established for ``pyRDF2Vec``:

-  `black <https://github.com/psf/black>`__: as code style, with a max line length of 79
   characters (according to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__);
-  `isort <https://github.com/PyCQA/isort>`__: to sort the imports;
-  `mypy <http://www.mypy-lang.org/>`__: as a optional static typing for Python
   (according to `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`__);
-  `doc8 <https://github.com/PyCQA/doc8>`__: as style checker for the
   documentation, with a max line length of 100 characters.

These checks are done by ``tox`` using `pre-commit
<https://github.com/pre-commit/pre-commit>`__.

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

A good architecture should allow a user to easily choose a embedding technique
with a walking strategy (*e.g.,* Weisfeler-Lehman) and a sampling strategy.

Add a new walking strategy
--------------------------

To add your own walking strategy, 6 steps are essential:

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that `poetry
   <https://python-poetry.org/>`__ is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec``:

.. code:: bash

   poetry install

2. **Create your walker** (*e.g.,* ``foo.py``) in ``pyrdf2vec/walkers``.
3. **Import your walker** at the end of the ``pyrdf2vec/walkers/__init__.py``
   file and in the ``__all__`` list:

.. code:: python

   from .anonymous import AnonymousWalker
   ...
   from .weisfeiler_lehman import WeisfeilerLehmanWalker
   from .wildcard import WildcardWalker
   from .foo import FooWalker

   __all__ = [
    "AnonymousWalker",
    ...
    "WeisfeilerLehmanWalker",
    "WildcardWalker",
    "FooWalker",
  ]

4. in your walker's class, **extend the** `Walker
   <https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/walkers/walker.py>`__
   **class** and implement at least the ``def extract(self, graph:
   KnowledgeGraph, instances: List[rdflib.URIRef]):`` function:

.. code:: python3

   from typing import List

   import rdflib

   from pyrdf2vec.graph import KnowledgeGraph
   from pyrdf2vec.walkers import Walker

   class FooWalker(Walker):
       """Defines the foo walking strategy.

       Attributes:
           depth: The depth per entity.
           walks_per_graph: The maximum number of walks per entity.

       """

       def __init__(self, depth: int, walks_per_graph: float):
           super().__init__(depth, walks_per_graph)

       def extract(self, graph: KnowledgeGraph, instances: List[rdflib.URIRef]):
           """Extracts walks rooted at the provided instances which are then each
           transformed into a numerical representation.

           Args:
               graph: The knowledge graph.
                   The graph from which the neighborhoods are extracted for the
                   provided instances.
               instances: The instances to extract the knowledge graph.

           Returns:
               The 2D matrix with its number of rows equal to the number of
               provided instances; number of column equal to the embedding size.

           """
           # TODO: to be implemented

**NOTE:** don't forget to update the docstring of your walker.

5. **create the unit tests** (*e.g.,* in the ``test_foo.py`` file) related
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

6. **Run unit tests, check that the code style and documentation are still correct:**

.. code:: bash

   pytest tests/walkers/test_foo.py
   tox -e lint,docs

In case you had to modify classes from ``pyRDF2Vec``, it will be necessary to
make sure that all tests still work:

.. code:: bash

   tox -e tests

**NOTE:** this may take some time (between 5-10 minutes), do this step only necessary.

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

- `Sphinx <https://www.sphinx-doc.org/en/master/>`__ as a Python documentation generator ;
-  `Google style
   docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`__:
   as a docstring writing convention.
- ``mypy``: as a optional static typing for Python.

To update the documentation, 5 steps are essential:

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that `poetry
   <https://python-poetry.org/>`__ is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to the documentation of ``pyRDF2Vec``:

.. code:: bash

   poetry install -E docs

2. **Modify what needed to be modified in the documentation**: available in the
   ``docs`` folder.

3. **Generate this documentation locally**:

.. code:: bash

   tox -e docs

4. **Check that the changes made are correct with your web browser:**

.. code:: bash

   $BROWSER _build/html/index.html

5. **Check that the code style of the documentation is still correct:**

.. code:: bash

   tox -e lint

Everything ok? Make a `pull request
<https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Submit your bugs and suggestions
--------------------------------

Feel free to `open an issue
<https://github.com/IBCNServices/pyRDF2Vec/issues/new/choose>`__ in case something is
not working as expected, or if you have any questions/suggestions.

Improve the code
----------------

The refactoring and optimization of code complexity is an art that must
be necessary to facilitate future contributions of ``pyRDF2Vec``.

To improve the code, 3 steps are essential:

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that ``poetry`` is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related to
``pyRDF2Vec``:

.. code:: bash

   poetry install

2. **Make your modifications**.

3. **Run unit tests, check that the code style and documentation are still correct:**

.. code:: bash

   tox

**NOTE:** this may take some time (between 5-10 minutes), do this step when
your code works.

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!
