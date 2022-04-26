from pyrdf2vec.connectors import Connector
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.typings import Literal, Response
from cachetools import cachedmethod
from cachetools.keys import hashkey
import operator
from functools import partial
from typing import List, Optional
from rdflib import URIRef, Literal
import rdflib_hdt
import attr
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

pos_entities = ["http://dl-learner.org/benchmark/dataset/animals#dog01",
            "http://dl-learner.org/benchmark/dataset/animals#dolphin01",
            "http://dl-learner.org/benchmark/dataset/animals#platypus01",
            "http://dl-learner.org/benchmark/dataset/animals#bat01"]

neg_entities = ["http://dl-learner.org/benchmark/dataset/animals#trout01",
            "http://dl-learner.org/benchmark/dataset/animals#herring01",
            "http://dl-learner.org/benchmark/dataset/animals#shark01",
            "http://dl-learner.org/benchmark/dataset/animals#lizard01",
            "http://dl-learner.org/benchmark/dataset/animals#croco01",
            "http://dl-learner.org/benchmark/dataset/animals#trex01",
            "http://dl-learner.org/benchmark/dataset/animals#turtle01",
            "http://dl-learner.org/benchmark/dataset/animals#eagle01",
            "http://dl-learner.org/benchmark/dataset/animals#ostrich01",
            "http://dl-learner.org/benchmark/dataset/animals#penguin01"]

train_entities = [entity for entity in pos_entities[0:2]+neg_entities[0:5]]
train_labels = list([1]*2+[0]*5)

test_entities = [entity for entity in pos_entities[2:]+neg_entities[5:]]
test_labels = list([1]*2+[0]*5)

entities = train_entities + test_entities
labels = train_labels + test_labels


@attr.s
class HDTConnector(Connector):
    def __attrs_post_init__(self):
        self.store = None

    @cachedmethod(operator.attrgetter("cache"), key=partial(hashkey, "fetch"))
    def fetch(self, query: str) -> Response:
        if self.store is None:
            self.store = rdflib_hdt.HDTStore(self.endpoint)
        try:
            res = self.store.hdt_document.search((query, None, None))[0]
            val = [{"p": {"value": r[1].toPython()},
                    "o": {"value": r[2].n3().split('"')[1]}} if isinstance(
                r[2], Literal) else {"p": {"value": r[1].toPython()},
                                     "o": {"value": r[2].toPython()}} for r
                   in res]

            return {"results": {"bindings": val}}
        except Exception as e:
            return {"results": {"bindings": []}}


    def get_query(self, entity: str, preds: Optional[List[str]] = None) -> str:
        return URIRef(entity)

import matplotlib.pyplot as plt
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

test_data = pd.read_csv("/Users/bramsteenwinckel/Documents/repos/pyRDF2Vec/samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("/Users/bramsteenwinckel/Documents/repos/pyRDF2Vec/samples/mutag/train.tsv", sep="\t")

train_entities = [entity for entity in train_data["bond"]]
train_labels = list(train_data["label_mutagenic"])

test_entities = [entity for entity in test_data["bond"]]
test_labels = list(test_data["label_mutagenic"])

entities = train_entities + test_entities
labels = train_labels + test_labels

if __name__ == '__main__':

    connector = HDTConnector("/Users/bramsteenwinckel/Documents/repos/pyRDF2Vec/samples/mutag/mutag.hdt")

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
                random_state=RANDOM_STATE,
                md5_bytes=None,
            )
        ],
        verbose=1,
    ).fit_transform(
        KG(connector=connector, is_remote=True, skip_verify=True,
            skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
        ),
        entities,
    )

    train_embeddings = embeddings[: len(train_entities)]
    test_embeddings = embeddings[len(train_entities) :]

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
        train_embeddings + test_embeddings
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

    # Create a legend.
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

