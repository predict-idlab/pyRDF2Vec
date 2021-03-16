import itertools
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

dcemd_to_avg_stdev = {}

for db, is_cache, entities, depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [False, True],
    [10, 25, 50],
    [25],
    [25],
):
    if not is_cache:
        kg = KG(
            f"http://10.2.35.70:5820/{db}",
            mul_req=False,
            cache=None,
        )
    else:
        kg = KG(f"http://10.2.35.70:5820/{db}", mul_req=False)

    label = "bond"
    if db == "am":
        label = "proxy"
    elif db == "dbpedia":
        label = "DBpedia_URL"

    e = [
        entity
        for entity in pd.read_csv(
            f"benchmarks/{db}/{db}-{entities}.tsv", sep="\t"
        )[label]
    ]

    times = []
    print(
        f"(db={db}, is_cache={is_cache}, entities={entities}, "
        + f"depth={depth}, max_walks={max_walks})"
    )
    for _ in tqdm(range(10)):
        if is_cache:
            tic = time.perf_counter()
            transformer = RDF2VecTransformer(
                Word2Vec(workers=1),
                [RandomWalker(depth, max_walks, random_state=RANDOM_STATE)],
            ).fit(KG(f"http://10.2.35.70:5820/{db}", mul_req=False), e)
            toc = time.perf_counter()
            times.append(toc - tic)
        else:
            tic = time.perf_counter()
            RDF2VecTransformer(
                Word2Vec(workers=1),
                [RandomWalker(depth, max_walks, random_state=RANDOM_STATE)],
            ).fit(
                KG(
                    f"http://10.2.35.70:5820/{db}",
                    mul_req=False,
                    cache=None,
                ),
                e,
            )
            toc = time.perf_counter()
        times.append(toc - tic)
    avg_stdev = [
        round(np.mean(times), 2),  # type:ignore
        round(np.std(times), 2),  # type:ignore
    ]

    print(
        f"(db={db}, is_cache={is_cache}, entities={entities}, "
        + f"depth={depth}, max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} ({transformer._walks} walks "
        + f"extracted per test"
    )
    dcemd_to_avg_stdev[(db, is_cache, entities, max_walks, depth)] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]}, is_cache={k[1]}, entities={k[2]}, "
        + f"depth={k[3]}, max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]}"
    )
