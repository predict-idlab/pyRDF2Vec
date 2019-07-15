# pyRDF2Vec

This repository contains an implementation of the algorithm in "RDF2Vec: RDF Graph Embeddings and Their Applications" by Petar Ristoski, Jessica Rosati, Tommaso Di Noia, Renato De Leone, Heiko Paulheim ([[paper]](http://semantic-web-journal.net/content/rdf2vec-rdf-graph-embeddings-and-their-applications-0) [[original code (Java + python)]](http://data.dws.informatik.uni-mannheim.de/rdf2vec/)).

## How does it work?

RDF2Vec is an unsupervised technique that builds further on Word2Vec, where an embedding is learned per word by either predicting the word based on its context (Continuous Bag-of-Words (CBOW)). To do this, RDF2Vec first creates "sentences" which can be fed to Word2Vec by extracting random walks of a certain depth from the Knowledge Graph. To create a random walk, we initialize its first hop to be one of the specified training entities in our KG. Then, we can iteratively extend our random walk by sampling out of the neighbors from the last hop of our walk.

Optionally, the algorithm can be extended by applying a Weisfeiler-Lehman transformation first, where each node is remapped on a label that is a hash of the subtree of a certain depth, rooted at that node.

## Creating your own embeddings

We provide an example script in `src/example.py`. In a nutshell:
* Load in your Knowledge Graph using [rdflib](https://github.com/RDFLib/rdflib) and convert it using `graph.rdflib_to_kg`
```python3
g = rdflib.Graph()
g.parse('../data/aifb.n3', format='n3')
kg = rdflib_to_kg(g, label_predicates=label_predicates)
```
* Create a list of entities and extract their neighborhoods using `graph.extract_instance`
```python3
test_data = pd.read_csv('../data/AIFB_test.tsv', sep='\t')
train_data = pd.read_csv('../data/AIFB_train.tsv', sep='\t')

train_people = [rdflib.URIRef(x) for x in train_data['person']]
train_labels = train_data['label_affiliation']

test_people = [rdflib.URIRef(x) for x in test_data['person']]
test_labels = test_data['label_affiliation']

train_graphs = [extract_instance(kg, person) for person in train_people]
test_graphs = [extract_instance(kg, person) for person in test_people]
```
* Provide the extracted neighborhoods to RDF2VecTransformer and call `fit`
* Afterwards, for each of the provided neighborhoods, you can retrieve its embedding (which is the embedding of the root of that neighborhood) by calling the `transform` method. Or you can use `fit_transform` at once.
```python3
transformer = RDF2VecTransformer(_type='walk', walks_per_graph=500)
embeddings = transformer.fit_transform(train_graphs + test_graphs)
```
* We can use then use the generated embeddings for a downstream tasks, such as classification.
```python3
train_embeddings = embeddings[:len(train_graphs)]
test_embeddings = embeddings[len(train_graphs):]

rf =  RandomForestClassifier(n_estimators=100)
rf.fit(train_embeddings, train_labels)

print(confusion_matrix(test_labels, rf.predict(test_embeddings)))
```
