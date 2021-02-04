import random
import warnings

import numpy as np
import pandas as pd
import rdflib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
)

np.random.seed(42)
random.seed(42)

warnings.filterwarnings("ignore")

LABEL_PREDICATES = {"http://dl-learner.org/carcinogenesis#isMutagenic"}

# Load our train & test instances and labels
test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [rdflib.URIRef(x) for x in train_data["bond"]]
train_labels = train_data["label_mutagenic"]

test_entities = [rdflib.URIRef(x) for x in test_data["bond"]]
test_labels = test_data["label_mutagenic"]

entities = train_entities + test_entities

# Convert the rdflib to our KnowledgeGraph object
kg = KG("samples/mutag/mutag.owl", label_predicates=LABEL_PREDICATES)

samplers = [
    ("Uniform", UniformSampler()),
    ("Object Frequency", ObjFreqSampler()),
    ("Inverse Object Frequency", ObjFreqSampler(inverse=True)),
    (
        "Inverse Object Frequency Split",
        ObjFreqSampler(inverse=True, split=True),
    ),
    ("Predicate Frequency", PredFreqSampler()),
    ("Inverse Predicate Frequency", PredFreqSampler(inverse=True)),
    ("Predicate + Object Frequency", ObjPredFreqSampler()),
    ("Inverse Predicate + Object Frequency", ObjPredFreqSampler(inverse=True)),
    ("PageRank", PageRankSampler()),
    ("Inverse PageRank", PageRankSampler(inverse=True)),
    ("PageRank Split", PageRankSampler(split=True)),
    ("Inverse PageRank Split", PageRankSampler(inverse=True, split=True)),
]

for name, sampler in samplers:
    # Create embeddings with random walks
    transformer = RDF2VecTransformer(walkers=[RandomWalker(4, 100, sampler)])
    walk_embeddings = transformer.fit_transform(kg, entities, verbose=True)

    # Split into train and test embeddings
    train_embeddings = walk_embeddings[: len(train_entities)]
    test_embeddings = walk_embeddings[len(train_entities) :]

    # Fit a support vector machine on train embeddings and evaluate on test
    clf = SVC(random_state=42)
    clf.fit(train_embeddings, train_labels)

    print(end=f"[{name}] Support Vector Machine: Accuracy = ")
    print(accuracy_score(test_labels, clf.predict(test_embeddings)))
