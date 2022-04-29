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
from pyrdf2vec.samplers import WideSampler
from pyrdf2vec.walkers import HALKWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

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
    Word2Vec(workers=1, epochs=10),
    # Extract all walks with a maximum depth of 2 for each entity using two
    # processes and use a random state to ensure that the same walks are
    # generated for the entities without hashing as MUTAG is a short KG.
    walkers=[
        HALKWalker(
            2,
            None,
            n_jobs=2,
            sampler=WideSampler(),
            random_state=RANDOM_STATE,
            md5_bytes=None,
        )
    ],
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
            ["http://dl-learner.org/carcinogenesis#salmonella"],
            ["http://dl-learner.org/carcinogenesis#cytogen_sce"],
            ["http://dl-learner.org/carcinogenesis#cytogen_ca"],
            ["http://dl-learner.org/carcinogenesis#mouse_lymph"],
            ["http://dl-learner.org/carcinogenesis#amesTestPositive"],
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
    SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

print("\nUsing literals:")
features = []

for literal in literals:
    charges, salmonella, sce, ca, lymph, pos_test = literal

    charges = list(charges)  # type: ignore

    salmonella_feat = int(salmonella == "true")
    salmonella_missing = int(salmonella == np.NaN)

    sce_feat = int(sce == "true")
    sce_missing = int(sce == np.NaN)

    ca_feat = int(ca == "true")
    ca_missing = int(ca == np.NaN)

    lymph_feat = int(lymph == "true")
    lymph_missing = int(lymph == np.NaN)

    pos_test_feat = int(pos_test == "true")
    pos_test_missing = int(pos_test == np.NaN)

    features.append(
        [
            np.max(charges),
            np.min(charges),
            np.mean(charges),  # type: ignore
            np.std(charges),  # type: ignore
            len(charges),  # type: ignore
            np.sum(charges),  # type: ignore
            salmonella_feat,
            salmonella_missing,
            sce_feat,
            sce_missing,
            ca_feat,
            ca_missing,
            lymph_feat,
            lymph_missing,
            pos_test_feat,
            pos_test_missing,
        ]
    )
features = np.array(features)  # type: ignore

train_embeddings2 = np.hstack(
    (train_embeddings, features[: len(train_entities)])  # type: ignore
)
test_embeddings2 = np.hstack(
    (test_embeddings, features[len(train_entities) :])  # type: ignore
)

train_features = features[: len(train_entities)]
test_features = features[len(train_entities) :]

# fit a Support Vector Machine on train embeddings.
clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
)
clf.fit(train_embeddings2, train_labels)

# Evaluate the Support Vector Machine on test embeddings.
predictions2 = clf.predict(test_embeddings2)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions2) * 100 :.4f}%"
)
print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions2))

f, ax = plt.subplots(1, 2, figsize=(15, 6))

# Define the color map.
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

ax[0].set_title(
    f"Without Literals ({accuracy_score(test_labels, predictions) * 100:.2f}%)"
)

# Reduce the dimensions of entity embeddings without literals to represent them
# in a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
    np.vstack((train_embeddings, test_embeddings))
)

# Plot the train embeddings without literals.
ax[0].scatter(
    X_tsne[: len(train_entities), 0],
    X_tsne[: len(train_entities), 1],
    edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
    facecolors=[color_map[i] for i in labels[: len(train_entities)]],
)

# Plot the test embeddings without literals.
ax[0].scatter(
    X_tsne[len(train_entities) :, 0],
    X_tsne[len(train_entities) :, 1],
    edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
    facecolors="none",
)

# Create a legend.
ax[0].scatter([], [], edgecolors="r", facecolors="r", label="train -")
ax[0].scatter([], [], edgecolors="g", facecolors="g", label="train +")
ax[0].scatter([], [], edgecolors="r", facecolors="none", label="test -")
ax[0].scatter([], [], edgecolors="g", facecolors="none", label="test +")
ax[0].legend(loc="upper right", ncol=2)

ax[1].set_title(
    f"With Literals ({accuracy_score(test_labels, predictions2) * 100 :.2f}%)"
)

# Reduce the dimensions of entity embeddings with literals to represent them in
# a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
    np.vstack((train_embeddings2, test_embeddings2))
)

# Plot the train embeddings with literals.
ax[1].scatter(
    X_tsne[: len(train_entities), 0],
    X_tsne[: len(train_entities), 1],
    edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
    facecolors=[color_map[i] for i in labels[: len(train_entities)]],
)

# Plot the test embeddings with literals.
ax[1].scatter(
    X_tsne[len(train_entities) :, 0],
    X_tsne[len(train_entities) :, 1],
    edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
    facecolors="none",
)

# Create a legend.
ax[1].scatter([], [], edgecolors="r", facecolors="r", label="train -")
ax[1].scatter([], [], edgecolors="g", facecolors="g", label="train +")
ax[1].scatter([], [], edgecolors="r", facecolors="none", label="test -")
ax[1].scatter([], [], edgecolors="g", facecolors="none", label="test +")
ax[1].legend(loc="upper right", ncol=2)

plt.show()
