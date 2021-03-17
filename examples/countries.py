import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22

data = pd.read_csv("samples/countries-cities/entities.tsv", sep="\t")

transformer = RDF2VecTransformer(
    # Use one worker threads for Word2Vec to ensure random determinism.
    # Must be used with PYTHONHASHSEED.
    Word2Vec(workers=1),
    # Extract a maximum of 10 walks of a maximum depth of 4 for each entity
    # using two processes and use a random state to ensure that the same walks
    # are generated for the entities.
    walkers=[RandomWalker(4, 10, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
)

# Train and save the Word2Vec model according to the KG, the entities, and
# a walking strategy.
embeddings, _ = transformer.fit_transform(
    # Defined the DBpedia endpoint server, as well as a set of predicates to
    # exclude from this KG.
    KG(
        "https://dbpedia.org/sparql",
        skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
        literals=[
            [
                "http://dbpedia.org/ontology/wikiPageWikiLink",
                "http://www.w3.org/2004/02/skos/core#prefLabel",
            ],
            ["http://dbpedia.org/ontology/humanDevelopmentIndex"],
        ],
    ),
    [entity for entity in data["location"]],
)

# Reduce the dimensions of entity embeddings to represent them in a 2D plane.
X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(embeddings)

# Ploy the embeddings of entities in a 2D plane, annotating them.
plt.figure(figsize=(10, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for x, y, t in zip(X_tsne[:, 0], X_tsne[:, 1], transformer._entities):
    plt.annotate(t.split("/")[-1], (x, y))

# Display the graph with a title, removing the axes for better readability.
plt.title("pyRDF2Vec", fontsize=32)
plt.axis("off")
plt.show()
