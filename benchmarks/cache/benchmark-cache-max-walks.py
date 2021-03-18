import itertools
import random
import time

import numpy as np
import pandas as pd
from cachetools import TTLCache
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from tqdm import tqdm

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

dcemd_to_avg_stdev = {}  # type: ignore

for db, is_cache, entities, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [False, True],
    [25],
    [25],
    [10, 25, 50],
):
    label = "bond"
    if db == "am":
        label = "proxy"
    elif db == "dbpedia":
        label = "DBpedia_URL"

    e = [
        entity
        for entity in pd.read_csv(f"res/{db}/{db}-{entities}.tsv", sep="\t")[
            label
        ]
    ]
    times = []

    for _ in tqdm(range(10)):
        cache = TTLCache(maxsize=1024, ttl=1200) if is_cache else None
        tic = time.perf_counter()
        walks = RandomWalker(
            max_depth, max_walks, random_state=RANDOM_STATE
        ).extract(
            KG(f"http://10.2.35.70:5820/{db}", mul_req=False, cache=cache), e
        )
        toc = time.perf_counter()
        times.append(toc - tic)

        with open(f"{db}-e{entities}-d{max_depth}-w{max_walks}", "w") as f:
            for walk in walks:
                f.write("%s\n" % walk)

    avg_stdev = [
        np.round(np.mean(times), 2),
        np.round(np.std(times), 2),
    ]

    print(
        f"(db={db},is_cache={is_cache},entities={entities},"
        + f"max_depth={max_depth},max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} (for {len(walks)} walks)"
    )
    dcemd_to_avg_stdev[
        (db, is_cache, entities, max_depth, max_walks)
    ] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},is_cache={k[1]},entities={k[2]},"
        + f"max_depth={k[3]},max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]}"
    )
