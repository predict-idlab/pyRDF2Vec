import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number
# generator.
np.random.seed(42)
random.seed(42)

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [rdflib.URIRef(x) for x in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [rdflib.URIRef(x) for x in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

entities = train_entities + test_entities
labels = train_labels + test_labels

embeddings = RDF2VecTransformer(
    # To also ensure determinism for Word2Vec.
    Word2Vec(workers=1),
    # Extract all possible walks of depth 2.
    walkers=[RandomWalker(2, None)],
).fit_transform(
    KG(
        "samples/mutag/mutag.owl",
        label_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
    ),
    entities,
    verbose=True,
)

train_embeddings = embeddings[: len(train_entities)]
test_embeddings = embeddings[len(train_entities) :]

# Fit a Support Vector Machine on train embeddings.
clf = SVC(random_state=42)
clf.fit(train_embeddings, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
print("Support Vector Machine:")
print(
    f"Accuracy = {accuracy_score(test_labels, clf.predict(test_embeddings))}"
)
print(confusion_matrix(test_labels, clf.predict(test_embeddings)))

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=42).fit_transform(
    train_embeddings + test_embeddings
)

# Define the color map
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

# Set the graph with a certain size.
plt.figure(figsize=(10, 4))

# Plot the train embeddings
plt.scatter(
    X_tsne[: len(train_entities), 0],
    X_tsne[: len(train_entities), 1],
    edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
    facecolors=[color_map[i] for i in labels[: len(train_entities)]],
)

# Plot the test embeddings.
plt.scatter(
    X_tsne[len(train_entities) :, 0],
    X_tsne[len(train_entities) :, 1],
    edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
    facecolors="none",
)

# Annotate few points.
plt.annotate(
    entities[25].split("/")[-1],
    xy=(X_tsne[25, 0], X_tsne[25, 1]),
    xycoords="data",
    xytext=(0.01, 0.0),
    fontsize=8,
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", facecolor="black"),
)
plt.annotate(
    entities[35].split("/")[-1],
    xy=(X_tsne[35, 0], X_tsne[35, 1]),
    xycoords="data",
    xytext=(0.4, 0.0),
    fontsize=8,
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", facecolor="black"),
)

# Create a legend
plt.scatter([], [], edgecolors="r", facecolors="r", label="train -")
plt.scatter([], [], edgecolors="g", facecolors="g", label="train +")
plt.scatter([], [], edgecolors="r", facecolors="none", label="test -")
plt.scatter([], [], edgecolors="g", facecolors="none", label="test +")
plt.legend(loc="upper right", ncol=2)

# Display the graph with a title, removing the axes for
# better readability.
plt.title("pyRDF2Vec", fontsize=32)
plt.axis("off")
plt.show()
