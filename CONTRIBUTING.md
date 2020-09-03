# Contributing

Thank you for wanting to bring your part to the `pyRDF2Vec` building. The project
being new, there are many opportunities for contributions:
1. **Add a new embedding technique**: the field of natural language processing
   is advancing rapidly. As such, many of the techniques used in the original
   implementation of RDF2Vec were outdated.
2. **Add a new walking strategy**: as an extension of the original RDF2Vec
   algorithm, nothing prevents you from adding your own walk strategy.
3. **Add a new sampling strategy**: sampling strategies was currently not yet
   implemented in `pyRDF2Vec`. As a result, those proposed by Cochez et al have
   been implemented, as well as other strategies.
4. **Improve the online documentation**: correcting spelling mistakes and using
   better sentence structures may help to better understand the use of
   `pyRDF2Vec`.
5. **Submit your bugs and suggestions**: reproduce the bug you encountered with
   `pyRDF2Vec` and/or suggest your ideas for the development of `pyRDF2Vec`,
   would help us to improve this library.
6. **Improve the code**: whether by refactoring or optimizing the complexity of
   certain functions, if an implementation seems adequate to you, do not
   hesitate to modify it by opening a pull request, or to inform us by opening
   an issue.

Each of the following sub-sections will give you more information on the
opportunities listed below.

---

## Add a new embedding technique

Currently, `pyRDF2Vec` does not have a generic implementation. Only
[Word2Vec](https://en.wikipedia.org/wiki/Word2vec) is implemented, and other
more powerful embedding techniques are available (*e.g.,* [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), [fasttext](https://fasttext.cc/), [GloVe](https://nlp.stanford.edu/projects/glove/)).

Before adding a new embedding technique, it is important to implement an
adequate architecture that must be discussed in an issue.

A good architecture should allow a user to easily choose a embedding technique
(*e.g.,* BERT) with a walking strategy (*e.g.,* Weisfeler-Lehman) and a sampling strategy.

## Add a new walking strategy

To add your own walking strategy, five steps are essential:
1. create your walker (*e.g.,* `foo.py`) in `pyrdf2vec/walkers`;
2. import your walker at the end of the `pyrdf2vec/walkers/__init__.py` file:

```bash
from .walker import *
...
from .wildcard import *
from .foo import *
```

3. in your walker, extend the
  [Walker](https://github.com/IBCNServices/pyRDF2Vec/blob/master/pyrdf2vec/walkers/walker.py)
  class and implement at least the `extract(self, graph, instances)` function:

```python3
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
```

**NOTE:** don't forget to update the docstring of your walker.

4. create the unit tests (*e.g.,* in the `test_foo.py` file) related to your
   walker in `tests/walkers/`:

```python3
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
```

5. install dependencies, run unit tests and check that the style of code is still correct:

```bash
pip install poetry
poetry install
tox -e style
tox -e tests
```

Everything ok? Make a [pull request](https://github.com/IBCNServices/pyRDF2Vec/pulls)!

## Add a new sampling strategy

The use of `pyRDF2Vec` implicitly asks to load the whole knowledge graph in
memory. This was perfect for the smalller benchmarks datasets that IDLab had
been using until now. However, this use is not ideal. It was therefore necessary
to make it possible to extract the walks by only loading only parts of the graph
into memory and/or by querying a certain end point (`SPARQL`) on a server.

## Improve the online documentation

The [online documentation of
pyRDF2Vec](https://pyrdf2vec.readthedocs.io/en/latest/) is hosted on [Read the
Docs](https://readthedocs.org/). To generate this online documentation, we use:
- [Sphinx](https://www.sphinx-doc.org/en/master/) as a Python documentation generator;
- [Google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html):
  as a docstring writing convention.

Before you can modify the `pyRDF2Vec` documentation with, you must first make sure you have `poetry` to install :
```bash
pip install poetry
```

With `poetry` installed, you can now install the dependencies related to the documentation:
```bash
poetry install -E docs
```

Once you have modified what needed to be modified in the documentation
(available in the `docs` folder), it is important to generate this documentation
locally with `tox`, to ensure that your modification has been taken into
account:
```bash
tox -e docs
```

As the documentation is updated, check that the changes made are correct with
your web browser:
```bash
$BROWSER _build/html/index.html
```

Everything is well rendered? Make a [pull request](https://github.com/IBCNServices/pyRDF2Vec/pulls)!

## Submit your bugs and suggestions

Feel free to [open an issue](https://github.com/IBCNServices/pyRDF2Vec/issues/new) in case something
is not working as expected, or if you have any questions/suggestions.

In order to help you out as good as possible:
- **describe the question/problem as precise as possible**;
- **inform your operating system**;
- **provide an example of minimal work with sample data (if possible) to reproduce
  the bug**.

## Improve the code

The refactoring and optimization of code complexity is an art that must be
necessary to facilitate future contributions of `pyRDF2Vec`.

Before you can modify the `pyRDF2Vec` code, you must first make sure you have `poetry` to install :
```bash
pip install poetry
```

With `poetry` installed, you can now install the dependencies:
```bash
poetry install
```

Once you have made your modifications, check that the style of the code is
still respected:
```bash
tox -e style
```

Then, launch the unit tests which can take several minutes:
```bash
tox -e tests
```

Everything ok? Make a [pull request](https://github.com/IBCNServices/pyRDF2Vec/pulls)!