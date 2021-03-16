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
<https://github.com/twisted/towncrier>`__ package to manage our changelog.
``towncrier`` uses independent files (called *news fragments*) for each pull
request. On release, those news fragments are compiled into our
``CHANGELOG.rst`` file.

Each file should be named like ``<ISSUE>.<TYPE>.rst``, where
``<ISSUE>`` is an issue number, and ``<TYPE>`` is one of:

* ``bugfix``: fixes a bug;
* ``doc``: documentation improvement, like rewording an entire session or
  adding missing docs;
* ``feature``: new user facing features, like a new walking strategy.

So for example: ``123.bugfix.rst``, ``456.feature.rst``.

Two scenarios exist for your pull request:

1. **if it fixes an issue**, use the issue number in the file name;
2. **otherwise**, use the pull request number.

If you are not sure what issue type to use, don't hesitate to ask in your pull
request.

``towncrier`` preserves multiple paragraphs and formatting (e.g., code
blocks, lists), but for entries other than ``feature`` it is usually better to
stick to a single paragraph to keep it concise.

If you would like to get a preview of how your change will look in
the final release notes, you can display the news fragments of the
``CHANGELOG`` file:

.. code:: bash

   tox -e changelog

From then on, you don't need to install ``towncrier`` yourself, you just have to abide by a
few simple rules:

- Wrap symbols like modules, functions, or classes into double backticks so
  they are rendered in a ``monospace font``.
- Wrap arguments into asterisks like in docstrings: *these* or *attributes*.
- If you mention functions or other callables, add parentheses at the end of
  their names: ``foo.func()`` or ``Foo.method()``.
  This makes the changelog a lot more readable.
- Prefer simple past tense or constructions with "now".
- If you would like to reference multiple issues, copy the news fragment to
  another filename. ``towncrier`` will merge all news fragments with identical
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

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that `poetry
   <https://python-poetry.org/>`__ is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec``:

.. code:: bash

   poetry install

2. **Create your embedder** (e.g., ``foo.py``) in ``pyrdf2vec/embedders``.
3. **Import your embedder** in the ``pyrdf2vec/embedders/__init__.py`` file and
   in the ``__all__`` list:

.. code:: python

   from .embedder import Embedder
   from .foo import FooEmbedder
   from .word2vec import Word2Vec

   __all__ = [
      "Embedder",
      "FooEmbedder",
      "Word2Vec",
  ]

4. in your embedder's class, **extend the** `Embedder
   <https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/embedders/embedder.py>`__
   **class** and implement at least the ``fit`` and ``transform`` functions:

.. code:: python3

   from typing import List

   import rdflib

   from pyrdf2vec.embedders import Embedder

   class FooEmbedder(Embedder):
    """Defines Foo embedding technique."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __init__(self):
        pass

    def fit(self, corpus: List[List[str]]):
        """Fits the Foo model based on provided corpus.

        Args:
            corpus: The corpus.

        Returns:
            The fitted model according to an embedding technique.

        """
        # TODO: to be implemented

    def transform(self, entities: List[rdflib.URIRef]) -> List[str]:
        """Constructs a features vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        # TODO: to be implemented

**NOTE:** don't forget to update the docstring of your embedder.

5. **Check that the code style and documentation are still correct:**

.. code:: bash

   tox -e lint,docs

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!


Add a new walking strategy
--------------------------

To add your own sampling strategy, the steps are almost similar to those for
adding an embedding technique:

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that `poetry
   <https://python-poetry.org/>`__ is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec``:

.. code:: bash

   poetry install

2. **Create your walker** (e.g., ``foo.py``) in ``pyrdf2vec/walkers``.
3. **Import your walker** in the ``pyrdf2vec/walkers/__init__.py`` file and in
   the ``__all__`` list:

.. code:: python

   from .anonymous import AnonymousWalker
   # ...
   from .walklets import WalkletWalker
   from .weisfeiler_lehman import WLWalker
   from .foo import FooWalker

   __all__ = [
       "AnonymousWalker",
       # ...
       "WalkletWalker",
       "WLWalker",
       "FooWalker",
  ]

4. in your walker's class, **extend the** `Walker
   <https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/walkers/walker.py>`__
   **class** and implement at least the ``extract`` function:

.. code:: python3

   from typing import Any, List, Set, Tuple

   import rdflib

   from pyrdf2vec.graph import KG
   from pyrdf2vec.samplers import Sampler, UniformSampler
   from pyrdf2vec.walkers import Walker

   class FooWalker(Walker):
    """Defines the foo walking strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Default to UniformSampler().

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph,
        sampler: Sampler = UniformSampler(),
    ):
        super().__init__(depth, walks_per_graph, sampler)

    def _extract(
        self, kg: KG, entities: List[rdflib.URIRef]
    ) -> Set[Tuple[Any, ...]]:
        """Extracts the walks and processes them for the embedding model.

        Args:
            kg: The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided entities.
            entities: The entities to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        # TODO: to be implemented

**NOTE:** don't forget to update the docstring of your walker.

5. **Run unit tests, check that the code style and documentation are still correct:**

.. code:: bash

   pytest tests/test_walkers.py
   tox -e lint,docs

In case you had to modify classes from ``pyRDF2Vec``, it will be necessary to
make sure that all tests still work:

.. code:: bash

   tox -e tests

**NOTE:** this may take some time (between 10-15 minutes), do this step only necessary.

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Add a new sampling strategy
---------------------------

To add your own sampling strategy, the steps are almost similar to those for
adding a walking strategy:

1. **Install the dependencies:** before you can install the dependencies of
   ``pyRDF2Vec``, you must first make sure that `poetry
   <https://python-poetry.org/>`__ is installed:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec``:

.. code:: bash

   poetry install

2. **Create your sampler** (e.g., ``Foo.py``) in ``pyrdf2vec/samplers``.
3. **Import your sampler** in the ``pyrdf2vec/samplers/__init__.py`` file and
   in the ``__all__`` list:

.. code:: python

   from .sampler import Sampler
   # ...
   from .foo import FooSampler
   from .frequency import ObjFreqSampler, ObjPredFreqSampler, PredFreqSampler
   from .pagerank import PageRankSampler

   __all__ = [
       "FooSampler",
       # ...
       "ObjFreqSampler",
       "ObjPredFreqSampler",
       "PageRankSampler",
       "PredFreqSampler",
       "Sampler",
  ]

4. in your sampler's class, **extend the** `Sampler
   <https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/samplers/sampler.py>`__
   **class** and implement at least the ``fit`` and ``get_weights`` functions:

.. code:: python3

   from pyrdf2vec.graph import KG
   from pyrdf2vec.samplers import Sampler

   class FooSampler(Sampler):
    """Defines the Foo sampling strategy."""

    def __init__(self):
        super().__init__()

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop):
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        # TODO: to be implemented

**NOTE:** don't forget to update the docstring of your sampler.

5. **Run unit tests, check that the code style and documentation are still correct:**

.. code:: bash

   pytest tests/test_samplers.py
   tox -e lint,docs

In case you had to modify classes from ``pyRDF2Vec``, it will be necessary to
make sure that all tests still work:

.. code:: bash

   tox -e tests

**NOTE:** this may take some time (between 10-15 minutes), do this step only necessary.

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Improve the online documentation
--------------------------------

The `online documentation of
pyRDF2Vec <https://pyrdf2vec.readthedocs.io/en/latest/>`__ is hosted on
`Read the Docs <https://readthedocs.org/>`__. To generate this online
documentation, we use:

- `Sphinx <https://www.sphinx-doc.org/en/master/>`__ as a Python documentation generator ;
-  `Google Style
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

**NOTE:** this may take some time (between 10-15 minutes), do this step when
your code works.

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!
