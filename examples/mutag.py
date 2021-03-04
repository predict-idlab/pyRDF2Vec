import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 42

# data = pd.read_csv("samples/countries-cities/entities.tsv", sep="\t")

# transformer = RDF2VecTransformer(
#     # Use one worker threads for Word2Vec to ensure random determinism.
#     # Must be used with PYTHONHASHSEED.
#     Word2Vec(workers=1),
#     # Extract a maximum of 25 walks of depth 4 for each entity using two
#     # processes and use a random state to ensure that the same walks are
#     # generated for the entities.
#     walkers=[RandomWalker(2, 2, n_jobs=2, random_state=RANDOM_STATE)],
#     verbose=1,
# )

kg = KG(
    "http://10.2.35.70:5820/dbpedia",
    skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
)
from pyrdf2vec.graphs import Vertex

print(kg.get_neighbors(Vertex("http://dbpedia.org/resource/Belgium")))

# # Reduce the dimensions of entity embeddings to represent them in a 2D plane.
# X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
#     # Train and save the Word2Vec model according to the KG, the entities, and
#     # a walking strategy.
#     transformer.fit_transform(
#         # Defined that the KG is remotely located, as well as a set of
#         # predicates to exclude from this KG.
#         KG(
#             "http://10.2.35.70:5820/dbpedia",
#             skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
#         ),
#         [entity for entity in data["location"]],
#     )
# )

# # Ploy the embeddings of entities in a 2D plane, annotating them.
# plt.figure(figsize=(10, 4))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
# for x, y, t in zip(X_tsne[:, 0], X_tsne[:, 1], transformer._entities):
#     plt.annotate(t.split("/")[-1], (x, y))

# # Display the graph with a title, removing the axes for better readability.
# plt.title("pyRDF2Vec", fontsize=32)
# plt.axis("off")
# plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.manifold import TSNE
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.svm import SVC

# from pyrdf2vec import RDF2VecTransformer
# from pyrdf2vec.embedders import Word2Vec
# from pyrdf2vec.graphs import KG
# from pyrdf2vec.walkers import RandomWalker

# # Ensure the determinism of this script by initializing a pseudo-random number.
# RANDOM_STATE = 42

# test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
# train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

# train_entities = [entity for entity in train_data["bond"]]
# train_labels = list(train_data["label_mutagenic"])

# test_entities = [entity for entity in test_data["bond"]]
# test_labels = list(test_data["label_mutagenic"])

# entities = train_entities + test_entities
# labels = train_labels + test_labels

# embeddings = RDF2VecTransformer(
#     # Ensure random determinism for Word2Vec.
#     # Must be used with PYTHONHASHSEED.
#     Word2Vec(workers=1),
#     # Extract all walks of depth 2 for each entity using two processes
#     # and use a random state to ensure that the same walks are generated for
#     # the entities.
#     walkers=[RandomWalker(2, 2, n_jobs=2, random_state=RANDOM_STATE)],
#     verbose=1,
# ).fit_transform(
#     KG(
#         "http://10.2.35.70:5820/dbpedia",
#     ),
#     entities,
# )

# # train_embeddings = embeddings[: len(train_entities)]
# # test_embeddings = embeddings[len(train_entities) :]

# # # Fit a Support Vector Machine on train embeddings.
# # clf = SVC(random_state=RANDOM_STATE)
# # clf.fit(train_embeddings, train_labels)

# # # Evaluate the Support Vector Machine on test embeddings.
# # predictions = clf.predict(test_embeddings)
# # print(
# #     f"Predicted {len(test_entities)} entities with an accuracy of "
# #     + f"{round(accuracy_score(test_labels, predictions), 2) * 100} %"
# # )
# # print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
# # print(confusion_matrix(test_labels, predictions))

# # # Reduce the dimensions of entity embeddings to represent them in a 2D plane.
# # X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
# #     train_embeddings + test_embeddings
# # )

# # # Define the color map
# # colors = ["r", "g"]
# # color_map = {}
# # for i, label in enumerate(set(labels)):
# #     color_map[label] = colors[i]

# # # Set the graph with a certain size.
# # plt.figure(figsize=(10, 4))

# # # Plot the train embeddings
# # plt.scatter(
# #     X_tsne[: len(train_entities), 0],
# #     X_tsne[: len(train_entities), 1],
# #     edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
# #     facecolors=[color_map[i] for i in labels[: len(train_entities)]],
# # )

# # # Plot the test embeddings.
# # plt.scatter(
# #     X_tsne[len(train_entities) :, 0],
# #     X_tsne[len(train_entities) :, 1],
# #     edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
# #     facecolors="none",
# # )

# # # Annotate few points.
# # plt.annotate(
# #     entities[25].split("/")[-1],
# #     xy=(X_tsne[25, 0], X_tsne[25, 1]),
# #     xycoords="data",
# #     xytext=(0.01, 0.0),
# #     fontsize=8,
# #     textcoords="axes fraction",
# #     arrowprops=dict(arrowstyle="->", facecolor="black"),
# # )
# # plt.annotate(
# #     entities[35].split("/")[-1],
# #     xy=(X_tsne[35, 0], X_tsne[35, 1]),
# #     xycoords="data",
# #     xytext=(0.4, 0.0),
# #     fontsize=8,
# #     textcoords="axes fraction",
# #     arrowprops=dict(arrowstyle="->", facecolor="black"),
# # )

# # # Create a legend
# # plt.scatter([], [], edgecolors="r", facecolors="r", label="train -")
# # plt.scatter([], [], edgecolors="g", facecolors="g", label="train +")
# # plt.scatter([], [], edgecolors="r", facecolors="none", label="test -")
# # plt.scatter([], [], edgecolors="g", facecolors="none", label="test +")
# # plt.legend(loc="upper right", ncol=2)

# # # Display the graph with a title, removing the axes for
# # # better readability.
# # plt.title("pyRDF2Vec", fontsize=32)
# # plt.axis("off")
# # plt.show()
