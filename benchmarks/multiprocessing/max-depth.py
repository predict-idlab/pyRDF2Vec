import asyncio
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname("../pyrdf2vec"))

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

dcemd_to_avg_stdev = {}  # type: ignore


for db, n_jobs, entities, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
    [1, 2, 4],
    [50],
    [1, 2, 4],
    [500],
):
    label = "bond"
    if db == "am":
        label = "proxy"
    elif db == "dbpedia":
        label = "DBpedia_URL"

    e = [e for e in pd.read_csv(f"res/{db}.tsv", sep="\t")[label]][:entities:]

    requests.get("http://10.2.35.70:5000/restart_stardog")

    times = []

    for _ in tqdm(range(10)):
        kg = KG(
            f"http://10.2.35.70:5820/{db}",
            mul_req=False,
            cache=None,
            skip_verify=True,
        )

        tic = time.perf_counter()
        entity_walks = RandomWalker(
            max_depth, max_walks, n_jobs=n_jobs
        ).extract(kg, e)
        toc = time.perf_counter()
        times.append(toc - tic)

    avg_stdev = [
        np.round(np.mean(times), 2),
        np.round(np.std(times), 2),
    ]

    num_walks = sum([len(e_walk) for e_walk in entity_walks])
    print(
        f"(db={db},n_jobs={n_jobs},entities={len(e)},"
        + f"max_depth={max_depth},max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} > {num_walks} walks"
    )
    dcemd_to_avg_stdev[
        (db, n_jobs, entities, max_depth, max_walks, num_walks)
    ] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},n_jobs={k[1]},entities={k[2]},"
        + f"max_depth={k[3]},max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]} > {k[5]} walks"
    )
