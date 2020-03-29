import random
import os
import numpy as np
import rdflib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

from converters import rdflib_to_kg
from rdf2vec import RDF2VecTransformer
from walkers import RandomWalker

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

#########################################################################
#			      			   DATA LOADING                             #
#########################################################################

# Load our train & test instances and labels
test_data = pd.read_csv('sample/MUTAG_test.tsv', sep='\t')
train_data = pd.read_csv('sample/MUTAG_train.tsv', sep='\t')

train_entities = [rdflib.URIRef(x) for x in train_data['bond']]
train_labels = train_data['label_mutagenic']

test_entities = [rdflib.URIRef(x) for x in test_data['bond']]
test_labels = test_data['label_mutagenic']

all_entities = train_entities + test_entities
all_labels = list(train_labels) + list(test_labels)

# Define the label predicates, all triples with these predicates
# will be excluded from the graph
label_predicates = [
    'http://dl-learner.org/carcinogenesis#isMutagenic'
]

# Convert the rdflib to our KnowledgeGraph object
kg = rdflib_to_kg('sample/mutag.owl', label_predicates=label_predicates)

#########################################################################
#			      		CREATING EMBEDDINGS                             #
#########################################################################

# We'll all possible walks of depth 2
random_walker = RandomWalker(4, float('inf'))

# Create embeddings with random walks
transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
walk_embeddings = transformer.fit_transform(kg, all_entities)

# Split into train and test embeddings
train_embeddings = walk_embeddings[:len(train_entities)]
test_embeddings = walk_embeddings[len(train_entities):]

#########################################################################
#			      		    FIT CLASSIFIER                              #
#########################################################################

# Fit a support vector machine on train embeddings and evaluate on test
clf = SVC(random_state=42)
clf.fit(train_embeddings, train_labels)

print(end='Support Vector Machine: Accuracy = ')
print(accuracy_score(test_labels, clf.predict(test_embeddings)))
print(confusion_matrix(test_labels, clf.predict(test_embeddings)))

#########################################################################
#			      		       T-SNE PLOT                               #
#########################################################################

# Create t-SNE embeddings from RDF2Vec embeddings (dimensionality reduction)
walk_tsne = TSNE(random_state=42)
X_walk_tsne = walk_tsne.fit_transform(walk_embeddings)

# Define a color map
colors = ['r', 'g']
color_map = {}
for i, label in enumerate(set(all_labels)):
    color_map[label] = colors[i]

# Plot the train embeddings
plt.figure(figsize=(10, 4))
plt.scatter(
	X_walk_tsne[:len(train_entities), 0],
    X_walk_tsne[:len(train_entities), 1],
    edgecolors=[color_map[i] for i in all_labels],
    facecolors=[color_map[i] for i in all_labels],
)

# Plot the test embeddings
plt.scatter(
	X_walk_tsne[len(train_entities):, 0],
    X_walk_tsne[len(train_entities):, 1],
    edgecolors=[color_map[i] for i in all_labels],
    facecolors='none'
)

# Annotate a few points
for i, ix in enumerate([25, 35]):
    plt.annotate(
    	all_entities[ix].split('/')[-1],
        xy=(X_walk_tsne[ix, 0], X_walk_tsne[ix, 1]), xycoords='data',
        xytext=(0.1 * i, 0.05 + 0.1 * i),
        fontsize=8, textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", facecolor='black')
    )

# Create a legend
plt.scatter([], [], edgecolors='r', facecolors='r', label='train -')
plt.scatter([], [], edgecolors='g', facecolors='g', label='train +')
plt.scatter([], [], edgecolors='r', facecolors='none', label='test -')
plt.scatter([], [], edgecolors='g', facecolors='none', label='test +')
plt.legend(loc='top right', ncol=2)

# Show & save the figure
plt.title('pyRDF2Vec', fontsize=32)
plt.axis('off')
plt.savefig('embeddings.png')
plt.show()
