import rdflib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from collections import defaultdict, Counter

from graph import *
from rdf2vec import RDF2VecTransformer


print(end='Loading data... ', flush=True)
g = rdflib.Graph()
g.parse('../data/aifb.n3', format='n3')
print('OK')

test_data = pd.read_csv('../data/AIFB_test.tsv', sep='\t')
train_data = pd.read_csv('../data/AIFB_train.tsv', sep='\t')

train_people = [rdflib.URIRef(x) for x in train_data['person']]
train_labels = train_data['label_affiliation']

test_people = [rdflib.URIRef(x) for x in test_data['person']]
test_labels = test_data['label_affiliation']

label_predicates = [
    rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
]

# Extract the train and test graphs

kg = rdflib_to_kg(g, label_predicates=label_predicates)

train_graphs = [extract_instance(kg, person) for person in train_people]
test_graphs = [extract_instance(kg, person) for person in test_people]

transformer = RDF2VecTransformer(_type='walk', walks_per_graph=500)
embeddings = transformer.fit_transform(train_graphs + test_graphs)

train_embeddings = embeddings[:len(train_graphs)]
test_embeddings = embeddings[len(train_graphs):]

rf =  RandomForestClassifier(n_estimators=100)
rf.fit(train_embeddings, train_labels)

print(confusion_matrix(test_labels, rf.predict(test_embeddings)))