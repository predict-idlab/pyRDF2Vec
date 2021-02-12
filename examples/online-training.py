import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 42

data = pd.read_csv("samples/countries-cities/entities.tsv", sep="\t")

# Defined that the Knowledge Graph (KG) is remotely located, as well as a set
# of predicates to exclude from this KG.
kg = KG(
    "https://dbpedia.org/sparql",
    skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
    is_remote=True,
)

# Train and save the Word2Vec model according to the KG, the entities, a
# walking strategy, and use a random state to ensure to generate the same walks
# for entities.
transformer = RDF2VecTransformer(
    # Ensure random determinism for Word2Vec.
    # Must be used with PYTHONHASHSEED.
    Word2Vec(workers=1),
    # Extract a maximum of 25 walks per entity of depth 4 and use a random
    # state to ensure that the same walks are generated for the entities.
    walkers=[RandomWalker(4, 25, random_state=RANDOM_STATE)],
    verbose=1,
)
transformer.fit_transform(
    kg,
    [entity for entity in data["location"]],
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
transformer = RDF2VecTransformer(verbose=1).load("countries")
embeddings = transformer.fit_transform(
    kg,
    [entity for entity in data["location"]],
    is_update=True,
)

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(embeddings)

# Plot the embeddings of entities in a 2D plane, annotating them.
plt.figure(figsize=(10, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for x, y, t in zip(X_tsne[:, 0], X_tsne[:, 1], transformer._entities):
    plt.annotate(t.split("/")[-1], (x, y))

# Display the graph with a title, removing the axes for better readability.
plt.title("Countries (Online Training)", fontsize=16)
plt.axis("off")
plt.show()

# Remove the binary file related to the RDF2VecTransformer.
os.remove("countries")
