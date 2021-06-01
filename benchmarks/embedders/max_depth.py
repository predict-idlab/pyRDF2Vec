import asyncio
import itertools
import os
import sys

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname("../pyrdf2vec"))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import FastText, Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler

dcemd_to_avg_stdev = {}
RANDOM_STATE = 22

for db, embedder_name, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [
        "Word2Vec",
        "FastText",
    ],
    [1, 2, 4],
    [250],
):
    test_data = pd.read_csv(f"res/{db}/test.tsv", sep="\t")
    train_data = pd.read_csv(f"res/{db}/train.tsv", sep="\t")

    label = "bond"
    if db == "am":
        label = "proxy"
        train_labels = list(train_data["label_cateogory"])
        test_labels = list(test_data["label_category"])
    elif db == "dbpedia":
        label = "DBpedia_URL"
        train_labels = list(train_data["label"])
        test_labels = list(test_data["label"])
    else:
        train_labels = list(train_data["label_mutagenic"])
        test_labels = list(test_data["label_mutagenic"])

    train_entities = [entity for entity in train_data[label]]
    test_entities = [entity for entity in test_data[label]]
    entities = train_entities + test_entities

    embedder = Word2Vec(workers=1)
    if embedder_name == "FastText":
        embedder = FastText(workers=1)

    skip_preds = {}
    if db == "mutag":
        skip_preds = {"http://dl-learner.org/carcinogenesis#isMutagenic"}
    elif db == "am":
        skip_preds = {
            "http://purl.org/collections/nl/am/objectCategory",
            "http://purl.org/collections/nl/am/material",
        }

    requests.get("http://10.2.35.70:5000/restart_stardog")
    accuracies = []

    for _ in tqdm(range(10)):
        embeddings, _ = RDF2VecTransformer(
            embedder,
            walkers=[
                RandomWalker(max_depth, max_walks, UniformSampler(), n_jobs=4)
            ],
        ).fit_transform(
            KG(
                f"http://10.2.35.70:5820/{db}",
                skip_predicates=skip_preds,
                skip_verify=True,
                mul_req=True,
            ),
            entities,
        )

        train_embeddings = embeddings[: len(train_entities)]
        test_embeddings = embeddings[len(train_entities) :]

        clf = GridSearchCV(
            SVC(random_state=RANDOM_STATE),
            {"C": [10 ** i for i in range(-3, 4)]},
        )
        clf.fit(train_embeddings, train_labels)
        predictions = clf.predict(test_embeddings)

        score = accuracy_score(test_labels, predictions)
        avg_stdev = [
            np.round(np.mean(score), 4),
            np.round(np.std(score), 4),
        ]

        accuracies.append(avg_stdev)

        print(
            f"{walker.name},accuracy={avg_stdev[0] * 100 :.2f} % +/- {avg_stdev[1]}"
        )
        dcemd_to_avg_stdev[
            (db, embedder_name, max_depth, max_walks)
        ] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},embedder={k[1]},"
        + f"max_depth={k[2]},max_walks={k[3]}) = "
        + f"{v[0] * 100 :.4f}"
    )
