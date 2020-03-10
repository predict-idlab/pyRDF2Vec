import random
import os
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

import rdflib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

from converters import rdflib_to_kg
from rdf2vec import RDF2VecTransformer

from walkers import (RandomWalker, WeisfeilerLehmanWalker, 
					 AnonymousWalker, WalkletWalker, NGramWalker,
					 CommunityWalker, HalkWalker)

import warnings
warnings.filterwarnings('ignore')

# Load our train & test instances and labels
test_data = pd.read_csv('../data/MUTAG_test.tsv', sep='\t')
train_data = pd.read_csv('../data/MUTAG_train.tsv', sep='\t')

train_people = [rdflib.URIRef(x) for x in train_data['bond']]
train_labels = train_data['label_mutagenic']

test_people = [rdflib.URIRef(x) for x in test_data['bond']]
test_labels = test_data['label_mutagenic']

all_labels = list(train_labels) + list(test_labels)

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
label_predicates = [
    'http://dl-learner.org/carcinogenesis#isMutagenic'
]

# Convert the rdflib to our KnowledgeGraph object
kg = rdflib_to_kg('../data/mutag.owl', label_predicates=label_predicates)

random_walker = RandomWalker(2, float('inf'))
ano_walker = AnonymousWalker(2, float('inf'))
walklet_walker = WalkletWalker(2, float('inf'))
ngram_walker = NGramWalker(2, float('inf'), n_wildcards=1)
wl_walker = WeisfeilerLehmanWalker(2, float('inf'))
com_walker = CommunityWalker(2, float('inf'))
halk_walker = HalkWalker(2, float('inf'), ub_freq_threshold=0.1)

# Create embeddings with random walks
transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
walk_embeddings = transformer.fit_transform(kg, train_people + test_people)

# Create embeddings using Weisfeiler-Lehman
transformer = RDF2VecTransformer(walkers=[halk_walker], sg=1)
wl_embeddings = transformer.fit_transform(kg, train_people + test_people)

# Fit model on the walk embeddings
train_embeddings = walk_embeddings[:len(train_people)]
test_embeddings = walk_embeddings[len(train_people):]

rf =  RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(train_embeddings, train_labels)

print('Random Forest:')
print(accuracy_score(test_labels, rf.predict(test_embeddings)))
print(confusion_matrix(test_labels, rf.predict(test_embeddings)))

clf =  GridSearchCV(SVC(random_state=42), {'kernel': ['linear', 'poly', 'rbf'], 'C': [10**i for i in range(-3, 4)]})
clf.fit(train_embeddings, train_labels)

print('Support Vector Machine:')
print(accuracy_score(test_labels, clf.predict(test_embeddings)))
print(confusion_matrix(test_labels, clf.predict(test_embeddings)))

# Fit model on the Weisfeiler-Lehman embeddings
train_embeddings = np.hstack((wl_embeddings[:len(train_people)], walk_embeddings[:len(train_people)]))
test_embeddings = np.hstack((wl_embeddings[len(train_people):], walk_embeddings[len(train_people):]))
# train_embeddings = wl_embeddings[:len(train_people)]
# test_embeddings = wl_embeddings[len(train_people):]

rf =  RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(train_embeddings, train_labels)

print('Random Forest:')
print(accuracy_score(test_labels, rf.predict(test_embeddings)))
print(confusion_matrix(test_labels, rf.predict(test_embeddings)))

clf =  GridSearchCV(SVC(random_state=42), {'kernel': ['linear', 'poly', 'rbf'], 'C': [10**i for i in range(-3, 4)]})
clf.fit(train_embeddings, train_labels)

print('Support Vector Machine:')
print(accuracy_score(test_labels, clf.predict(test_embeddings)))
print(confusion_matrix(test_labels, clf.predict(test_embeddings)))

# Create TSNE plots of our embeddings
colors = ['r', 'g', 'b', 'y']
color_map = {}
for i, label in enumerate(set(all_labels)):
	color_map[label] = colors[i]

walk_tsne = TSNE(random_state=42)
X_walk_tsne = walk_tsne.fit_transform(walk_embeddings)
wl_tsne = TSNE(random_state=42)
X_wl_tsne = wl_tsne.fit_transform(wl_embeddings)

f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X_walk_tsne[:, 0], X_walk_tsne[:, 1], c=[color_map[i] for i in all_labels])
ax[1].scatter(X_wl_tsne[:, 0], X_wl_tsne[:, 1], c=[color_map[i] for i in all_labels])
ax[0].set_title('Walk Embeddings')
ax[1].set_title('Weisfeiler-Lehman Embeddings')
plt.show()