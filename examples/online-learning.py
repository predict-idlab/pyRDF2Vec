import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

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

# Defines the MUTAG KG with the predicates to be skipped.
kg = KG(
    "samples/mutag/mutag.owl",
    skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
)

transformer = RDF2VecTransformer(
    # Ensure random determinism for Word2Vec.
    # Must be used with PYTHONHASHSEED.
    Word2Vec(workers=1),
    # Extract all walks with a maximum depth of 2 for each entity by using two
    # processes and a random state to ensure that the same walks are generated
    # for the entities.
    walkers=[RandomWalker(2, None, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
)
embeddings, _ = transformer.fit_transform(kg, entities)

transformer.save("mutag")

train_embeddings = embeddings[: len(train_entities)]
test_embeddings = embeddings[len(train_entities) :]

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

print("\nAdding 20 mores entities.")

new_data = pd.read_csv("samples/mutag/online-training.tsv", sep="\t")
new_entities = [entity for entity in new_data["bond"]]
new_labels = list(new_data["label_mutagenic"])

transformer = RDF2VecTransformer(
    Word2Vec(workers=1),
    walkers=[RandomWalker(2, None, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
).load("mutag")
transformer.fit_transform(
    kg,
    new_entities,
    is_update=True,
)

# Get all the old and the new embeddings.
embeddings = transformer._embeddings

train_embeddings = embeddings[: len(train_entities)]
new_embeddings = embeddings[-len(new_entities) :]
test_embeddings = embeddings[len(train_entities) :][: -len(new_entities)]

clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
)
clf.fit(train_embeddings + new_embeddings, train_labels + new_labels)

predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

print("\nTrain all the entities.")

transformer = RDF2VecTransformer(
    Word2Vec(workers=1),
    walkers=[RandomWalker(2, None, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
)
embeddings, _ = transformer.fit_transform(
    kg, train_entities + new_entities + test_entities
)

train_embeddings = embeddings[: len(train_entities) + len(new_entities)]
test_embeddings = embeddings[len(train_entities) + len(new_entities) :]

clf = GridSearchCV(
    SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_labels + new_labels)

predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_entities)} entities with an accuracy of "
    + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
)
print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

os.remove("mutag")
