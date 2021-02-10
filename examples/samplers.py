import pandas as pd
from sklearn.metrics import accuracy_score
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
SEED = 42

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [entity for entity in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [entity for entity in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

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
    embeddings = RDF2VecTransformer(
        # Use one worker threads for Word2Vec to ensure random determinism.
        # Must be used with PYTHONHASHSEED.
        Word2Vec(workers=1),
        # Extract a maximum of 100 walks per entity of depth 4 and use a seed
        # to ensure that the same walks are generated for the entities.
        walkers=[RandomWalker(4, 100, sampler, seed=SEED)],
    ).fit_transform(
        KG(
            "samples/mutag/mutag.owl",
            label_predicates={
                "http://dl-learner.org/carcinogenesis#isMutagenic"
            },
        ),
        train_entities + test_entities,
    )

    train_embeddings = embeddings[: len(train_entities)]
    test_embeddings = embeddings[len(train_entities) :]

    # Fit a Support Vector Machine on train embeddings.
    clf = SVC(random_state=SEED)
    clf.fit(train_embeddings, train_labels)

    # Evaluate the Support Vector Machine on test embeddings.
    print(
        end=f"[{name},accuracy="
        + f"{accuracy_score(test_labels, clf.predict(test_embeddings))}]"
    )
