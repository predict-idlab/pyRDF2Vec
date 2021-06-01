import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname("../pyrdf2vec"))

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    WideSampler,
)

dcemd_to_avg_stdev = {}  # type: ignore

for db, entities, max_depth, max_walks, sampler in itertools.product(
    ["mutag", "am", "dbpedia"],
    [
        ObjFreqSampler(),
        ObjPredFreqSampler(),
        PageRankSampler(),
        PredFreqSampler(),
        UniformSampler(),
        WideSampler(),
    ],
    [25, 50, 100],
    [4],
    [500],
):
    label = "bond"
    skip_predicates = {"http://dl-learner.org/carcinogenesis#isMutagenic"}
    if db == "am":
        label = "proxy"
        skip_predicates = {
            "http://purl.org/collections/nl/am/objectCategory",
            "http://purl.org/collections/nl/am/material",
        }

    elif db == "dbpedia":
        label = "DBpedia_URL"
        skip_predicates = {"www.w3.org/1999/02/22-rdf-syntax-ns#type"}

    e = [e for e in pd.read_csv(f"res/{db}.tsv", sep="\t")[label]][:entities:]
    times = []

    for _ in tqdm(range(10)):
        embeddings, _ = RDF2VecTransformer(
            # Ensure random determinism for Word2Vec.
            # Must be used with PYTHONHASHSEED.
            Word2Vec(workers=1, epochs=10),
            # Extract all walks with a maximum depth of 2 for each entity using two
            # processes and use a random state to ensure that the same walks are
            # generated for the entities without hashing as MUTAG is a short KG.
            walkers=[
                RandomWalker(max_depth, max_walks, n_jobs=4, sampler=sampler)
            ],
            verbose=1,
        ).fit_transform(
            KG(
                "samples/mutag/mutag.owl",
                skip_predicates={
                    "http://dl-learner.org/carcinogenesis#isMutagenic"
                },
                skip_verify=True,
            ),
            entities,
        )

    avg_stdev = [
        np.round(np.mean(times), 2),
        np.round(np.std(times), 2),
    ]

    num_walks = sum([len(e_walk) for e_walk in entity_walks])
    print(
        f"(db={db},mul_req={mul_req},entities={len(e)},"
        + f"max_depth={max_depth},max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} > {num_walks} walks"
    )
    dcemd_to_avg_stdev[
        (db, mul_req, entities, max_depth, max_walks, num_walks)
    ] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},mul_req={k[1]},entities={k[2]},"
        + f"max_depth={k[3]},max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]} > {k[5]} walks"
    )
