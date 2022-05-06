import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
)

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [entity for entity in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [entity for entity in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

entities = train_entities + test_entities

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

print(f"Prediction of {len(test_entities)} entities:")

for _, sampler in samplers:
    embeddings, _ = RDF2VecTransformer(  # type:ignore
        # Use one worker threads for Word2Vec to ensure random determinism.
        # Must be used with PYTHONHASHSEED.
        Word2Vec(workers=1),
        # Extract a maximum of 100 walks of a maximum depth of 4 for each
        # entity using two processes and use a random state to ensure that the
        # same walks are generated for the entities.
        walkers=[
            RandomWalker(4, 100, sampler, n_jobs=2, random_state=RANDOM_STATE)
        ],
    ).fit_transform(
        KG(
            "samples/mutag/mutag.owl",
            skip_predicates={
                "http://dl-learner.org/carcinogenesis#isMutagenic"
            },
        ),
        entities,
    )

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
        f"{sampler}\naccuracy="
        + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
    )
