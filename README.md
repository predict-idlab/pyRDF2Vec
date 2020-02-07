# pyRDF2Vec [![PyPI version](https://badge.fury.io/py/pyRDF2Vec.svg)](https://badge.fury.io/py/pyRDF2Vec) [![Downloads](https://pepy.tech/badge/pyrdf2vec)](https://pepy.tech/project/pyrdf2vec)

![Generated Embeddings](embedding.png "Generated Embeddings")

This repository contains an implementation of the algorithm in "RDF2Vec: RDF Graph Embeddings and Their Applications" by Petar Ristoski, Jessica Rosati, Tommaso Di Noia, Renato De Leone, Heiko Paulheim ([[paper]](http://semantic-web-journal.net/content/rdf2vec-rdf-graph-embeddings-and-their-applications-0) [[original code (Java + python)]](http://data.dws.informatik.uni-mannheim.de/rdf2vec/)).

## How does it work?

RDF2Vec is an unsupervised technique that builds further on Word2Vec, where an embedding is learned per word by either predicting the word based on its context (Continuous Bag-of-Words (CBOW)) or predicting the context based on a word (Skip-Gram (SG)). To do this, RDF2Vec first creates "sentences" which can be fed to Word2Vec by extracting random walks of a certain depth from the Knowledge Graph. To create a random walk, we initialize its first hop to be one of the specified training entities in our KG. Then, we can iteratively extend our random walk by sampling out of the neighbors from the last hop of our walk.

Optionally, the algorithm can be extended by applying a Weisfeiler-Lehman transformation first, where each node is remapped on a label that is a hash of the subtree of a certain depth, rooted at that node.

## Creating your own embeddings

We provide an example script in `rdf2vec/example.py`. In a nutshell:
* Load in your Knowledge Graph using [rdflib](https://github.com/RDFLib/rdflib)
```python3
# Load the data with rdflib
g = rdflib.Graph()
g.parse('../data/mutag.owl')
print('OK')
```
* Load the train/test entities that we want to embed and corresponding labels
```python3
# Load our train & test instances and labels
test_data = pd.read_csv('../data/MUTAG_test.tsv', sep='\t')
train_data = pd.read_csv('../data/MUTAG_train.tsv', sep='\t')

train_people = [rdflib.URIRef(x) for x in train_data['bond']]
train_labels = train_data['label_mutagenic']

test_people = [rdflib.URIRef(x) for x in test_data['bond']]
test_labels = test_data['label_mutagenic']

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
label_predicates = [
    rdflib.term.URIRef('http://dl-learner.org/carcinogenesis#isMutagenic')
]
```
* Convert the `rdflib.Graph` to our `KnowledgeGraph` object using `rdflib_to_kg` and provide it, together with a list of entities to the `RDF2VecTransformer`
```python3
# Convert the rdflib to our KnowledgeGraph object
kg = rdflib_to_kg(g, label_predicates=label_predicates)

# Create embeddings with random walks
transformer = RDF2VecTransformer(wl=False, max_path_depth=4)
walk_embeddings = transformer.fit_transform(kg, train_people + test_people)
```
* We can use then use the generated embeddings for a downstream tasks, such as classification.
```python3
# Fit model on the walk embeddings
train_embeddings = walk_embeddings[:len(train_people)]
test_embeddings = walk_embeddings[len(train_people):]

rf =  RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(train_embeddings, train_labels)

print('Random Forest:')
print(accuracy_score(test_labels, rf.predict(test_embeddings)))
print(confusion_matrix(test_labels, rf.predict(test_embeddings)))
```

## Determinism

In order to have deterministic results, the `PYTHONHASHSEED` environment variable has to be set: `PYTHONHASHSEED=42 python3 example.py`
