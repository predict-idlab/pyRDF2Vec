import itertools
import time

import numpy as np
import pandas as pd
from cachetools import TTLCache
from tqdm import tqdm

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

dcemd_to_avg_stdev = {}  # type: ignore


for db, is_cache, entities, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [False, True],
    [50],
    [4],
    [100, 500, 1000],
):
    label = "bond"
    if db == "am":
        label = "proxy"
    elif db == "dbpedia":
        label = "DBpedia_URL"

    e = [e for e in pd.read_csv(f"res/{db}.tsv", sep="\t")[label]][:entities:]
    times = []

    for _ in tqdm(range(10)):
        cache = TTLCache(maxsize=1024, ttl=1200) if is_cache else None
        tic = time.perf_counter()
        entity_walks = RandomWalker(max_depth, max_walks).extract(
            KG(f"http://10.2.35.70:5820/{db}", cache=cache), e
        )
        toc = time.perf_counter()
        times.append(toc - tic)

    avg_stdev = [
        np.round(np.mean(times), 2),
        np.round(np.std(times), 2),
    ]

    num_walks = sum([len(e_walk) for e_walk in entity_walks])
    print(
        f"(db={db},is_cache={is_cache},entities={len(e)},"
        + f"max_depth={max_depth},max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} > {num_walks} walks"
    )
    dcemd_to_avg_stdev[
        (db, is_cache, entities, max_depth, max_walks, num_walks)
    ] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},is_cache={k[1]},entities={len(k[2])},"
        + f"max_depth={k[3]},max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]} > {k[5]} walks"
    )
