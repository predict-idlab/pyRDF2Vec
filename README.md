<div align="center">
	<img src="assets/embeddings.png"></div>
</div>

<p align="center">
    <a href="https://badge.fury.io/py/pyRDF2Vec">
        <img alt="PyPI" src="https://badge.fury.io/py/pyRDF2Vec.svg">
    </a>
    <a href="https://pepy.tech/project/pyrdf2vec">
        <img alt="Downloads" src="https://pepy.tech/badge/pyrdf2vec">
    </a>
</p>

<p align="center">Python implementation and extension of <a href="http://rdf2vec.org/">RDF2Vec</a> to create a 2D feature matrix from a knowledge graph for downstream ML tasks.<p>

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

It's as simple as that:

    pip install pyRDF2vec

## Getting Started

To create embeddings for a list of entities, there are two steps to do beforehand:
1. **create a Knowledge Graph object**;
2. **define a walking strategy**.

For a more elaborate example, check at the
[example.py](https://github.com/IBCNServices/pyRDF2Vec/blob/master/example.py)
file:

    PYTHONHASHSEED=42 python3 rdf2vec/example.py

**NOTE:** the `PYTHONHASHSEED` (*e.g.,* 42) is to ensure determinism.

### Create a Knowledge Graph object

To create a Knowledge Graph object, you can initialize it in several ways:
```python
from rdf2vec.converters import endpoint_to_kg, rdflib_to_kg

label_predicates = [
    "http://dl-learner.org/carcinogenesis#isMutagenic"
]

# Create a Knowledge Graph from rdflib:
kg = rdflib_to_kg("samples/mutag.owl", label_predicates=label_predicates)

# Create a Knowledge Graph from a SPARQL endpoint:
kg = endpoint_to_kg("http://localhost:5820/db/query?query=", label_predicates=label_predicates)
```

### Define a walking strategy

To define a walking strategy, start by choosing one available on the [Wiki page](https://github.com/IBCNServices/pyRDF2Vec/wiki/Walking-Strategies).

For example, the definition of the Random walking strategy with a depth of 4 is
implemented as follows:

```python
from rdf2vec.walkers import RandomWalker

random_walker = RandomWalker(4, float("inf"))
```

### Create embeddings

Finally, the creation of embeddings for a list of entities simply goes like this:

```python
from rdf2vec import RDF2VecTransformer

transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
# Entities should be a list of URIs that can be found in the Knowledge Graph
embeddings = transformer.fit_transform(kg, entities)
```

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

<div align="center">
	<img src="assets/ID_Lab_Logo.svg"></div>
</div>
