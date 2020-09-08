import warnings
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import rdflib
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.converters import rdflib_to_kg
from pyrdf2vec.graph import KnowledgeGraph
from pyrdf2vec.walkers import RandomWalker, Walker

DATASET = {
    "test": ["samples/mutag-test.tsv", "bond", "label_mutagenic"],
    "train": ["samples/mutag-train.tsv", "bond", "label_mutagenic"],
}
LABEL_PREDICATES = ["http://dl-learner.org/carcinogenesis#isMutagenic"]
OUTPUT = "samples/mutag.owl"
WALKERS = [RandomWalker(4, float("inf"))]

PLOT_SAVE = "embeddings.png"
PLOT_TITLE = "pyRDF2Vec"

warnings.filterwarnings("ignore")


def create_embeddings(
    kg: KnowledgeGraph,
    entities: List[str],
    split: int,
    walkers: Sequence[Walker],
    sg: int = 1,
) -> Tuple[List[str], List[str]]:
    """Creates embeddings for a list of entities according to a knowledge graphs and a walking strategy.

    Args:
        kg: The knowledge graph.
            The graph from which the neighborhoods are extracted for the
            provided instances.
        entities (array-like): The train and test instances to create the
            embedding.
        split: Split value for train and test embeddings.
        walker: The list of walkers strategies.
        sg: The training algorithm. 1 for skip-gram; otherwise CBOW.
            Defaults to 1.

    Returns:
        The embeddings of the provided instances.

    """
    transformer = RDF2VecTransformer(walkers=walkers, sg=sg)
    walk_embeddings = transformer.fit_transform(kg, entities)
    return walk_embeddings[:split], walk_embeddings[split:]


def load_data(
    file_name: str, col_entity: str, col_label: str, sep: str = "\t"
) -> Tuple[List[rdflib.URIRef], List[str]]:
    """Loads entities and labels from a file.

    Args:
        file_name: The file name.
        col_entity: The name of the column header related to the entities.
        col_label: The name of the column header related to the labels.
        sep: The delimiter to use.
            Defaults to "\t".

    Returns:
        The URIs of the entities with their labels.

    """
    data = pd.read_csv(file_name, sep=sep, header=0)
    return [rdflib.URIRef(x) for x in data[col_entity]], list(data[col_label])


test_entities, test_labels = load_data(
    DATASET["test"][0], DATASET["test"][1], DATASET["test"][2]
)
train_entities, train_labels = load_data(
    DATASET["train"][0], DATASET["train"][1], DATASET["train"][2]
)

entities = train_entities + test_entities
labels = train_labels + test_labels

kg = rdflib_to_kg(OUTPUT, label_predicates=LABEL_PREDICATES)
train_embeddings, test_embeddings = create_embeddings(
    kg, entities, len(train_entities), WALKERS
)

# Fit a support vector machine on train embeddings and evaluate on test
clf = SVC(random_state=42)
clf.fit(train_embeddings, train_labels)
print("Support Vector Machine:")
print(
    f"Accuracy = {accuracy_score(test_labels, clf.predict(test_embeddings))}"
)
print(confusion_matrix(test_labels, clf.predict(test_embeddings)))

# Create t-SNE embeddings from RDF2Vec embeddings (dimensionality reduction)
walk_tsne = TSNE(random_state=42)
X_walk_tsne = walk_tsne.fit_transform(train_embeddings + test_embeddings)

# Define the color map
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

plt.figure(figsize=(10, 4))

# Plot the train embeddings
plt.scatter(
    X_walk_tsne[: len(train_entities), 0],
    X_walk_tsne[: len(train_entities), 1],
    edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
    facecolors=[color_map[i] for i in labels[: len(train_entities)]],
)

# Plot the test embeddings
plt.scatter(
    X_walk_tsne[len(train_entities) :, 0],
    X_walk_tsne[len(train_entities) :, 1],
    edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
    facecolors="none",
)

# Annotate a few points
for i, ix in enumerate([25, 35]):
    plt.annotate(
        entities[ix].split("/")[-1],
        xy=(X_walk_tsne[ix, 0], X_walk_tsne[ix, 1]),
        xycoords="data",
        xytext=(0.1 * i, 0.05 + 0.1 * i),
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

# Show & save the figure
plt.title(PLOT_TITLE, fontsize=32)
plt.axis("off")
plt.savefig(PLOT_SAVE)
plt.show()
