import random
import warnings
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker, Walker

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

FILE = "samples/countries-cities/entities.tsv"
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
LABEL_PREDICATES = {"www.w3.org/1999/02/22-rdf-syntax-ns#type"}

# We'll extract all possible walks of depth 4 (with 25 hops)
WALKERS = [RandomWalker(4, 25)]
# We'll extract all possible walks of depth 4 (with 25 hops) with
# multi-processing. Using multi-processing improves the speed of
# extraction of walks, but this may conflict with the policy of the SPARQL
# endpoint server.
# WALKERS = [RandomWalker(4, 25, n_jobs=2)]

PLOT_TITLE = "pyRDF2Vec"


def create_embeddings(
    kg: KG,
    entities: List[rdflib.URIRef],
    walkers: Sequence[Walker],
    sg: int = 1,
) -> List[str]:
    """Creates embeddings for a list of entities according to a knowledge
    graphs and a walking strategy.

    Args:
        kg: The knowledge graph.
            The graph from which the neighborhoods are extracted for the
            provided instances.
        entities: The train and test instances to create the embedding.
        walker: The list of walkers strategies.
        sg: The training algorithm. 1 for skip-gram; otherwise CBOW.
            Defaults to 1.

    Returns:
        The embeddings of the provided instances.

    """
    transformer = RDF2VecTransformer(Word2Vec(sg=sg), walkers=walkers)
    return transformer.fit_transform(kg, entities, verbose=True)


# Load our train & test instances and labels
data = pd.read_csv(FILE, sep="\t")

entities = [rdflib.URIRef(x) for x in data["location"]]
labels = data["label"]

kg = KG(
    SPARQL_ENDPOINT,
    label_predicates=LABEL_PREDICATES,
    is_remote=True,
)

# Create t-SNE embeddings from RDF2Vec embeddings (dimensionality reduction)
X_walk_tsne = TSNE(random_state=42).fit_transform(
    create_embeddings(kg, entities, WALKERS)
)

# Define a color map
colors = ["r", "g"]
color_map = {}
for i, label in enumerate(set(labels)):
    color_map[label] = colors[i]

plt.figure(figsize=(10, 4))

# Plot the train embeddings
plt.scatter(
    X_walk_tsne[:, 0],
    X_walk_tsne[:, 1],
    edgecolors=[color_map[i] for i in labels],
    facecolors=[color_map[i] for i in labels],
)

for x, y, t in zip(X_walk_tsne[:, 0], X_walk_tsne[:, 1], entities):
    plt.annotate(t.split("/")[-1], (x, y))

# Show & save the figure
plt.title(PLOT_TITLE, fontsize=32)
plt.axis("off")
plt.show()
