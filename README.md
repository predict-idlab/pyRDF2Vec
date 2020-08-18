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

<hr>

| Section | Description |
|-|-|
| [RDF2Vec](#what-is-rdf2vec) | What is RDF2Vec? |
| [Installing](#installation) | Installing pyRDF2Vec |
| [Getting started](#getting-started) | A quick introduction |
| [Documentation](#documentation) | A link to our documentation |
| [Citation](#referencing) | Citing pyRDF2Vec in scholarly articles |

# What is RDF2Vec?

This repository contains an implementation of the algorithm in "RDF2Vec: RDF Graph Embeddings and Their Applications" by Petar Ristoski, Jessica Rosati, Tommaso Di Noia, Renato De Leone, Heiko Paulheim ([[paper]](http://semantic-web-journal.net/content/rdf2vec-rdf-graph-embeddings-and-their-applications-0) [[original code]](http://data.dws.informatik.uni-mannheim.de/rdf2vec/)).

RDF2Vec is an unsupervised technique that builds further on Word2Vec, where an embedding is learned per word by either predicting the word based on its context (Continuous Bag-of-Words (CBOW)) or predicting the context based on a word (Skip-Gram (SG)). To do this, RDF2Vec first creates "sentences" which can be fed to Word2Vec by extracting walks of a certain depth from the Knowledge Graph.

# Installation

Few options:
* `(python -m) pip install pyRDF2Vec`
* Clone the repository & run `python setup.py install`
* `(python -m) pip install git+git://github.com/IBCNServices/pyRDF2Vec.git`

# Getting Started

### Create a Knowledge Graph object

First, you will need to create a Knowledge Graph object (defined in `graph.py`). We offer several conversion options (such as converting from rdflib or from an endpoint), which can be found in `converters.py`.

```python3
from rdf2vec.converters import rdflib_to_kg

# We want to filter out all triples with certain predicates
label_predicates = [
    'http://dl-learner.org/carcinogenesis#isMutagenic'
]

kg = rdflib_to_kg('sample/mutag.owl', label_predicates=label_predicates)
```

### Define a walking strategy

pyRDF2Vec offers several walking strategies, which can be found in the `walkers/` module.

```python3
from rdf2vec.walkers import RandomWalker

# We specify the depth and maximum number of walks per entity
random_walker = RandomWalker(4, float('inf'))
```

### Create embeddings

Then, we can create embeddings for a list of entities:

```python3
from rdf2vec import RDF2VecTransformer

transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
# Entities should be a list of URIs that can be found in the KG
embeddings = transformer.fit_transform(kg, entities)
```

For a more elaborate example, check `example.py`. You can run it as follows: `PYTHONHASHSEED=42 python3 rdf2vec/example.py`. The `PYTHONHASHSEED` is to ensure determinism.

# Documentation


# Referencing

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