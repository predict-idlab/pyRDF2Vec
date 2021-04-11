import itertools
import time

import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname("../pyrdf2vec"))

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

dcemd_to_avg_stdev = {}  # type: ignore


for db, mul_req, entities, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [False, True],
    [25, 50, 100],
    [4],
    [500],
):
    label = "bond"
    if db == "am":
        label = "proxy"
    elif db == "dbpedia":
        label = "DBpedia_URL"

    e = [e for e in pd.read_csv(f"res/{db}.tsv", sep="\t")[label]][:entities:]
    times = []

    for _ in tqdm(range(10)):
        kg = KG(f"http://10.2.35.70:5820/{db}", mul_req=mul_req, cache=None)

        tic = time.perf_counter()
        entity_walks = RandomWalker(max_depth, max_walks).extract(kg, e)
        toc = time.perf_counter()
        times.append(toc - tic)

        kg.connector.close()

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
