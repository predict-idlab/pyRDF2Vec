import itertools
import sys
import time

import numpy as np
import pandas as pd
from cachetools import TTLCache
from tqdm import tqdm

from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

dcemd_to_avg_stdev = {}  # type: ignore


for db, entities, max_depth, max_walks in itertools.product(
    ["mutag", "am", "dbpedia"],
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
    sizes = []

    for _ in tqdm(range(10)):
        kg = KG(f"http://10.2.35.70:5820/{db}", mul_req=False)
        walks = RandomWalker(max_depth, max_walks).extract(kg, e)
        sizes.append(sys.getsizeof(kg._transition_matrix))

    avg_stdev = [
        np.round(np.mean(sizes), 2),
        np.round(np.std(sizes), 2),
    ]

    print(
        f"(db={db},entities={e},"
        + f"max_depth={max_depth},max_walks={max_walks}) = "
        + f"{avg_stdev[0]} +/- {avg_stdev[1]} > {len(walks)} walks"
    )
    dcemd_to_avg_stdev[(db, entities, max_depth, max_walks)] = avg_stdev

for k, v in dcemd_to_avg_stdev.items():
    print(
        f"(db={k[0]},entities={k[2]},max_depth={k[3]},max_walks={k[4]}) = "
        + f"{v[0]} +/- {v[1]}"
    )
