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

1. `Add a new embedding technique <#add-a-new-embedding-technique>`__: offer a
   new possibility to have potentially better results when training a corpus of
   walks.
2. `Add a new walking strategy <#add-a-new-walking-strategy>`__: offer a
   new possibility to extract walks in a Knowledge Graph.
3. `Add a new sampling strategy <#add-a-new-sampling-strategy>`__: offer a new
   possibility of assigning weights to links in a Knowledge Graph.
4. `Add a new connector <#add-a-new-connector>`__: offer a new possibility to
   extend the use of ``pyRDF2Vec`` to other syntax and file formats in RDF.
5. `Improve the online documentation <#improve-the-online-documentation>`__:
   correcting spelling mistakes and using better sentence structures may help
   to better understand the use of ``pyRDF2Vec``.
6. `Submit your bugs and suggestions <#submit-your-bugs-and-suggestions>`__:
   reproduce the bug you encountered with ``pyRDF2Vec`` and/or suggest your
   ideas for the development of ``pyRDF2Vec``, would help us to improve this
   library.
7. `Improve the code <#improve-the-code>`__: whether by refactoring or
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
-  **It's up to you how you handle updates to the main branch:** since we
   squash on merge, whether you prefer to rebase on ``main`` or merge
   ``main`` into your branch, do whatever is more comfortable for you.


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

Install the Dependencies
------------------------

Before contributing, it is important that you can install the pyRDF2Vec
dependencies in your environment. Whether you are using a Notebook or directly
your text editor, here is the preferred method:

Install `poetry <https://python-poetry.org/>`__:

.. code:: bash

   pip install poetry

With ``poetry`` installed, you can now install the dependencies related
to ``pyRDF2Vec`` in a virtual environment:

.. code:: bash

   poetry install

Now all you have to do is spawn a terminal to your virtual environment:

.. code:: bash

   poetry shell

If you prefer, you could contribute directly to ``pyRDF2Vec`` with the Docker
image and avoid installing dependencies on your machine:

.. code:: bash

   docker-compose up --build -d

Now you only have to go to `localhost:9000 <http://localhost:9000>`__.

Add a New Embedding Technique
-----------------------------

Adding a new embedding technique offers the possibility to have potentially
better results for some uses-cases when training a corpus of walks.

To achieve this, there are 5 points there are 5 points to follow:

1. **Create your embedder** (e.g., ``Foo``) in ``pyrdf2vec/embedders``.
2. **Import your embedder** in the ``pyrdf2vec/embedders/__init__.py`` file and
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

3. **Extend the** `Embedder
   <https://github.com/IBCNServices/pyRDF2Vec/blob/main/pyrdf2vec/embedders/embedder.py>`__
   **class** in your embedder's class and implement at least the ``fit`` and
   ``transform`` functions:

.. code:: python

    from typing import List

    import attr

    from pyrdf2vec.embedders import Embedder
    from pyrdf2vec.typings import Embeddings, Entities

    @attr.s
    class Foo(Embedder):
        """Defines Foo embedding technique."""

        def fit(self,  walks: List[List[SWalk]], is_updated: bool = False) -> Embedder:
            """Fits the model based on provided walks.

            Args:
                walks: The walks to create the corpus to to fit the model.
                is_update: True if the new walks should be added to old model's
                    walks, False otherwise.
                    Defaults to False.

            Returns:
                The fitted Foo model.

            """
            # TODO: to be implemented

        def transform(self, entities: Entities) -> Embeddings:
            """The features vector of the provided entities.

            Args:
                entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

            Returns:
                The features vector of the provided entities.

            """
            # TODO: to be implemented

4. **Create unit tests of your embedding technique:**

Create a ``tests/embedders/foo.py`` file and see `how the tests are done for
Word2Vec
<https://github.com/IBCNServices/pyRDF2Vec/blob/main/tests/embedders/test_word2vec.py>`__
as an example.

Once this is done, run your tests:

.. code:: bash

   pytest tests/samplers/foo.py

5. **Check that the code style and the documentation are still correct:**

.. code:: bash

   tox -e lint,docs


Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!


Add a New Walking Strategy
--------------------------

Adding a new walking strategy allows to perform on a new possibility to
extract walks for a given Knowledge Graph.

To add your own walking strategy, the steps are almost similar to those for
adding an embedding technique:

1. **Create your walker** (e.g., ``FooWalker``) in ``pyrdf2vec/walkers``.
2. **Import your walker** in the ``pyrdf2vec/walkers/__init__.py`` file and in
   the ``__all__`` list:

.. code:: python

    from .walker import Walker

    # ...
    from .foo import FooWalker
    # ...
    from .walklet import WalkletWalker
    from .weisfeiler_lehman import WLWalker

    __all__ = [
        # ...
        "FooWalker",
        # ...
        "Walker",
        "WalkletWalker",
        "WLWalker",
    ]

3. **Extend the** `Walker
   <https://github.com/IBCNServices/pyRDF2Vec/blob/main/pyrdf2vec/walkers/walker.py>`__
   **class** in your walker's class and implement at least the ``_extract``
   function:

.. code:: python

    from hashlib import md5
    from typing import List, Set

    import attr

    from pyrdf2vec.graphs import KG, Vertex
    from pyrdf2vec.typings import EntityWalks, SWalk, Walk
    from pyrdf2vec.walkers import Walker

    @attr.s
    class FooWalker(Walker):
        """Defines the foo walking strategy.

        Args:
            depth: The maximum depth of one walk.
            max_walks: The maximum number of walks per entity.
            sampler: The sampling strategy.
                Defaults to pyrdf2vec.samplers.UniformSampler().
            n_jobs: The number of process to use for multiprocessing.
                Defaults to 1.
            with_reverse: extracts children's and parents' walks from the root,
                creating (max_walks * max_walks) more walks of 2 * depth.
                Defaults to False.
            random_state: The random state to use to ensure random determinism to
                generate the same walks for entities.
                Defaults to None.

        """

        def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
            """Extracts walks rooted at the provided entities which are then
            each transformed into a numerical representation.

            Args:
                kg: The Knowledge Graph.
                instance: The instance to be extracted from the Knowledge Graph.

            Returns:
                The 2D matrix with its number of rows equal to the number of
                provided entities; number of column equal to the embedding size.

            """
            canonical_walks: Set[SWalk] = set()
            # TODO: to be implemented
            return {instance.name: list(canonical_walks)}

4. **Create unit tests of your walking strategy:**

Create a ``tests/walkers/foo.py`` file and see `how the tests are done for
RandomWalker
<https://github.com/IBCNServices/pyRDF2Vec/blob/main/tests/walkers/test_random.py>`__
as an example.

Once this is done, run your tests:

.. code:: bash

   pytest tests/walkers/foo.py

5. **Check that the code style and the documentation are still correct:**

.. code:: bash

   tox -e lint,docs

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Add a New Sampling Strategy
---------------------------

Adding a new sampling strategy performs a new way of assigning weights to links
in a Knowledge Graph.

To add your own sampling strategy, the steps are almost similar to those for
adding a walking strategy and a embedding technique:

1. **Create your sampler** (e.g., ``FooSampler``) in ``pyrdf2vec/samplers``.
2. **Import your sampler** in the ``pyrdf2vec/samplers/__init__.py`` file and
   in the ``__all__`` list:

.. code:: python

    from .sampler import Sampler

    # ...
    from .foo import FooSampler
    # ...
    from .uniform import UniformSampler

    __all__ = [
        # ...
        "FooSampler",
        # ...
        "Sampler",
        "UniformSampler",
    ]

3. **Extend the** `Sampler
   <https://github.com/IBCNServices/pyRDF2Vec/blob/main/pyrdf2vec/samplers/sampler.py>`__
   **class** in your sampler's class and implement at least the ``fit`` and
   ``get_weight`` functions:

.. code:: python

    import attr

    from pyrdf2vec.graph import KG
    from pyrdf2vec.samplers import Sampler
    from pyrdf2vec.typings import Hop

    @attr.s
    class FooSampler(Sampler):
        """Defines the Foo sampling strategy."""

        def fit(self, kg: KG) -> None:
            """Since the weights are uniform, this function does nothing.

            Args:
                kg: The Knowledge Graph.

            """
            # TODO: to be implemented

        def get_weight(self, hop: Hop) -> int:
            """Gets the weight of a hop in the Knowledge Graph.

            Args:
                hop: The hop (pred, obj) to get the weight.

            Returns:
                The weight for a given hop.

            """
            # TODO: to be implemented

4. **Create unit tests of your sampling technique:**

Create a ``tests/samplers/foo.py`` file and see `how the tests are done for
UniformSampler
<https://github.com/IBCNServices/pyRDF2Vec/blob/main/tests/samplers/uniform.py>`__
as an example.

Once this is done, run your tests:

.. code:: bash

   pytest tests/samplers/foo.py

5. **Check that the code style and the documentation are still correct:**

.. code:: bash

   tox -e lint,docs

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Add a New Connector
-------------------

Add a new connector to extend the use of ``pyRDF2Vec`` to other syntax and file
formats in RDF (e.g., Turtle syntax)

To add your own connector, the steps are almost similar to those previously
illustrated:

1. **Create your connector** (e.g., ``FooConnector``) in ``pyrdf2vec/connectors``.
2. **Import your connector** in the ``pyrdf2vec/connector/__init__.py`` file.
3. **Extend the** `Connector
   <https://github.com/IBCNServices/pyRDF2Vec/blob/main/pyrdf2vec/connectors.py>`__
   **class** in your connector's class and implement at least the ``fetch``
   function:

.. code:: python

    import attr
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util import Retry

    from pyrdf2vec.connectors import Connector

    @attr.s
    class FooConnector(Connector):
        """Represents a Foo connector."""

        def __attrs_post_init__(self):
            adapter = HTTPAdapter(
                Retry(
                    total=3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    method_whitelist=["HEAD", "GET", "OPTIONS"],
                )
            )
            self._session.mount("http", adapter)
            self._session.mount("https", adapter)

        def fetch(self, query: str) -> None:
            """Fetchs the result of a query.

               Args:
                   query: The query to fetch the result.

               Returns:
                   The generated dictionary from the ['results']['bindings']
                   json.

            """
            # TODO: to be implemented

4. **Create unit tests of your connector:**

Create a ``tests/connectors/foo.py`` file and see `how the tests are done for
SPARQLConnector
<https://github.com/IBCNServices/pyRDF2Vec/blob/main/pyrdf2vec/connectors.py>`__
as an example.

Once this is done, run your tests:

.. code:: bash

   pytest tests/connectors/foo.py

5. **Check that the code style and the documentation are still correct:**

.. code:: bash

   tox -e lint,docs

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!


Improve the Online Documentation
--------------------------------

The `online documentation of
pyRDF2Vec <https://pyrdf2vec.readthedocs.io/en/latest/>`__ is hosted on
`Read the Docs <https://readthedocs.org/>`__. To generate this online
documentation, we use:

- `Sphinx <https://www.sphinx-doc.org/en/main/>`__ as a Python documentation generator ;
-  `Google Style
   docstrings <https://www.sphinx-doc.org/en/main/usage/extensions/example_google.html>`__:
   as a docstring writing convention.
- ``mypy``: as a optional static typing for Python.

To update the documentation, 4 steps are essential:

1. **Modify what needed to be modified in the documentation**: available in the
   ``docs`` folder.

2. **Generate this documentation locally**:

.. code:: bash

   tox -e docs

3. **Check that the changes made are correct with your web browser:**

.. code:: bash

   $BROWSER _build/html/index.html

4. **Check that the code style of the documentation is still correct:**

.. code:: bash

   tox -e lint

Everything ok? Make a `pull request
<https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!

Submit your Bugs and Suggestions
--------------------------------

Feel free to `open an issue
<https://github.com/IBCNServices/pyRDF2Vec/issues/new/choose>`__ in case something is
not working as expected, or if you have any questions/suggestions.

Improve the Code
----------------

The refactoring and optimization of code complexity is an art that must
be necessary to facilitate future contributions of ``pyRDF2Vec``.

To improve the code, 2 steps are essential:

1. **Make your modifications**.
2. **Run unit tests, check that the code style and documentation are still correct:**

.. code:: bash

   tox

Everything ok? Make a `pull
request <https://github.com/IBCNServices/pyRDF2Vec/pulls>`__!
