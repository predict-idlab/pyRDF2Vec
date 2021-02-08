import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number
# generator.
np.random.seed(42)
random.seed(42)

data = pd.read_csv("samples/countries-cities/entities.tsv", sep="\t")

# Defined that the Knowledge Graph (KG) is remotely located, as well as a set
# of predicates to exclude from this KG.
kg = KG(
    "https://dbpedia.org/sparql",
    label_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
    is_remote=True,
)

# Train and save the Word2Vec model according to the KG, the entities, and a
# walking strategy.
transformer = RDF2VecTransformer(walkers=[RandomWalker(4, 25, n_jobs=4)])
transformer.fit_transform(
    kg,
    [rdflib.URIRef(x) for x in data["location"]],
    verbose=True,
)
transformer.save("countries")

# Add new entities in the existing KG and update the previously trained model
# with the embeddings of these new entities.
data = pd.DataFrame(
    {
        "location": "http://dbpedia.org/resource/Afghanistan",
        "id": 23.0,
        "label": 0.0,
    },
    index=[0],
)
transformer = RDF2VecTransformer().load("countries")
embeddings = transformer.fit_transform(
    kg,
    [rdflib.URIRef(x) for x in data["location"]],
    is_update=True,
    verbose=True,
)

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=42).fit_transform(embeddings)

# Plot the embeddings of entities in a 2D plane, annotating them.
plt.figure(figsize=(10, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for x, y, t in zip(X_tsne[:, 0], X_tsne[:, 1], transformer.entities_):
    plt.annotate(t.split("/")[-1], (x, y))

# Display the graph with a title, removing the axes for better readability.
plt.title("Countries (Online Training)", fontsize=16)
plt.axis("off")
plt.show()
