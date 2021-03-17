import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 10

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [entity for entity in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [entity for entity in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

entities = train_entities + test_entities
labels = train_labels + test_labels

embeddings, literals = RDF2VecTransformer(
    # Ensure random determinism for Word2Vec.
    # Must be used with PYTHONHASHSEED.
    Word2Vec(workers=1),
    # Extract all walks with a maximum depth of 2 for each entity using two
    # processes and use a random state to ensure that the same walks are
    # generated for the entities.
    walkers=[RandomWalker(2, None, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
).fit_transform(
    KG(
        "samples/mutag/mutag.owl",
        skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
        literals=[
            [
                "http://dl-learner.org/carcinogenesis#hasAtom",
                "http://dl-learner.org/carcinogenesis#charge",
            ],
        ],
    ),
    entities,
)

train_embeddings = embeddings[: len(train_entities)]
test_embeddings = embeddings[len(train_entities) :]

print("\nWithout using literals:")
# Fit a Support Vector Machine on train embeddings and pick the best
# C-parameters (regularization strength).
clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10 ** i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

print("\nUsing literals:")
features = []
for charges in literals:
    charges = list(map(float, charges))
    features.append(
        [
            np.max(charges),
            np.min(charges),
            np.mean(charges),
            np.std(charges),
            len(charges),
            np.sum(charges),
        ]
    )
features = np.array(features)

train_features = features[: len(train_entities)]
test_features = features[len(train_entities) :]

train_embeddings = np.hstack((train_embeddings, train_features))
test_embeddings = np.hstack((test_embeddings, test_features))

# Fit a Support Vector Machine on train embeddings and pick the best
# C-parameters (regularization strength).
clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10 ** i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
    np.vstack((train_embeddings, test_embeddings))
)

# Define the color map.
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

# Set the graph with a certain size.
plt.figure(figsize=(10, 4))

# Plot the train embeddings.
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
