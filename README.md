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
</p>
<p align="center">Python implementation and extension of <a href="http://rdf2vec.org/">RDF2Vec</a> <b>to create a 2D feature matrix from a knowledge graph</b> for downstream ML tasks.<p>

---

## What is RDF2Vec?

RDF2Vec is an unsupervised technique that builds further on
[Word2Vec](https://en.wikipedia.org/wiki/Word2vec), where an embedding is
learned per word, in two ways:
1. **the word based on its context**: Continuous Bag-of-Words (CBOW);
2. **the context based on a word**: Skip-Gram (SG).

To create this embedding, RDF2Vec first creates "sentences" which can be fed to
Word2Vec by extracting walks of a certain depth from a knowledge graph.

This repository contains an implementation of the algorithm in "RDF2Vec: RDF
Graph Embeddings and Their Applications" by Petar Ristoski, Jessica Rosati,
Tommaso Di Noia, Renato De Leone, Heiko Paulheim
([[paper]](http://semantic-web-journal.net/content/rdf2vec-rdf-graph-embeddings-and-their-applications-0)
[[original code]](http://data.dws.informatik.uni-mannheim.de/rdf2vec/)).

## Installation

`pyRDF2Vec` can be installed in two ways:

1. from [PyPI](https://pypi.org/project/pyrdf2vec) using `pip`:

```bash
pip install pyRDF2vec
```

2. from any compatible Python dependency manager (e.g., `poetry`):

```bash
poetry add pyRDF2vec
```

## Getting Started

To create embeddings for a list of entities, there are two steps to do beforehand:
1. **create a Knowledge Graph object**;
2. **define a walking strategy**.

For a more elaborate example, check at the
[example.py](https://github.com/IBCNServices/pyRDF2Vec/blob/master/example.py)
file:

```bash
PYTHONHASHSEED=42 python3 example.py
```

**NOTE:** the `PYTHONHASHSEED` (*e.g.,* 42) is to ensure determinism.

### Create a Knowledge Graph object

To create a Knowledge Graph object, you can initialize it in several ways:
```python
from pyrdf2vec.converters import endpoint_to_kg, rdflib_to_kg

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
label_predicates = [
    "http://dl-learner.org/carcinogenesis#isMutagenic"
]

# Create a Knowledge Graph from rdflib
kg = rdflib_to_kg("samples/mutag.owl", label_predicates=label_predicates)

# Create a Knowledge Graph from a SPARQL endpoint
kg = endpoint_to_kg("http://localhost:5820/db/query?query=", label_predicates=label_predicates)
```

### Define a walking strategy

To define a walking strategy, start by choosing one available on the [Wiki page](https://github.com/IBCNServices/pyRDF2Vec/wiki/Walking-Strategies).

For example, the definition of the Random walking strategy with a depth of 4 is
implemented as follows:

```python
from pyrdf2vec.walkers import RandomWalker

random_walker = RandomWalker(4, float("inf"))
```

### Create embeddings

Finally, the creation of embeddings for a list of entities simply goes like this:

```python
from pyrdf2vec import RDF2VecTransformer

transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
# Entities should be a list of URIs that can be found in the Knowledge Graph
embeddings = transformer.fit_transform(kg, entities)
```

## Documentation

For more information on how to use `pyRDF2Vec`, [visit our online
documentation](https://pyrdf2vec.readthedocs.io/en/latest/) which is
automatically updated with the latest version of the `master` branch.

From then on, you will be able to learn more about the use of the modules as
well as their functions available to you.

## Contributions

To add a new walking strategy and/or new features that are related to
[RDF2Vec](http://rdf2vec.org/), take a look at the
[CONTRIBUTING](https://github.com/IBCNServices/pyRDF2Vec/blob/master/CONTRIBUTING.md)
file. Also, feel free to submit your bugs and suggestions by opening an issue in
the issue tracker, it would help us a lot.

## Referencing

If you use `pyRDF2Vec` in a scholarly article, we would appreciate a citation:

```bibtex
@misc{pyrdf2vec,
      title={pyRDF2Vec: A python library for RDF2Vec},
      author={Gilles Vandewiele and Bram Steenwinckel and Michael Weyns
      		  and Pieter Bonte and Femke Ongenae and Filip De Turck},
      year={2020},
      note={\url{https://github.com/IBCNServices/pyRDF2Vec}}
}
```
