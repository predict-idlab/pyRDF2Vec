import itertools
import multiprocessing
import random
import time

import attr
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker, Walker

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# A large number of entities but with small number of walks per entity is
# interesting as probably cache will not be used very often here. Small number of
# entities but large number of walks is interesting in a similar way as well.  Of
# course, keep the entities you extract walk from fixed (i.e. pick 10, 100 and
# 1000 entities but always use the same ones as some might be faster than
# others)


@attr.s
class Benchmark:

    kg = attr.ib(validator=attr.validators.instance_of(KG))

    entities = attr.ib(
        factory=list, validator=attr.validators.instance_of(list)
    )

    embedder: Embedder = attr.ib(
        factory=lambda: Word2Vec(workers=1),
        validator=attr.validators.instance_of(Embedder),
    )

    walker = attr.ib(
        factory=lambda: [RandomWalker(2, random_state=RANDOM_STATE)],
        validator=attr.validators.instance_of(list),
    )

    tests_itr = attr.ib(default=10, validator=attr.validators.instance_of(int))

    def evaluate(
        self,
        verbose=0,
    ):
        times = []
        for _ in tqdm(
            range(self.tests_itr),
            disable=True if verbose == 0 else False,
        ):
            tic = time.perf_counter()
            RDF2VecTransformer(Word2Vec(workers=1), self.walker).fit_transform(
                self.kg, self.entities
            )
            toc = time.perf_counter()
            times.append(toc - tic)
        return [
            round(np.mean(times), 2),
            round(np.stdev(times), 2),
            round(np.stdev(times, ddof=1), 2),
        ]


if __name__ == "__main__":
    dcemd_to_avg_stdev = {}

    for db, is_cache, entities, depth, max_walks in itertools.product(
        ["mutag", "am", "dbpedia"],
        [False, True],
        [25],
        [25],
        [10, 25, 50],
    ):
        if not is_cache:
            kg = KG(f"http://10.2.35.70:5820/{db}", mul_req=False, cache=None)
        else:
            kg = KG(f"http://10.2.35.70:5820/{db}", mul_req=False)

        label = "bond"
        if db == "am":
            label = "proxy"
        elif db == "dbpedia":
            label = "DBpedia_URL"

        avg_stdev = Benchmark(
            kg,
            [
                entity
                for entity in pd.read_csv(
                    f"benchmarks/{db}/{db}-{entities}.tsv", sep="\t"
                )[label]
            ],
            walker=[
                RandomWalker(
                    depth, max_walks=max_walks, random_state=RANDOM_STATE
                )
            ],
        ).evaluate()

        print(
            f"(db={db}, is_cache={is_cache}, entities={entities},"
            + f"depth={depth}, max_walks={max_walks}) = "
            + f"{avg_stdev[0]} +/- {avg_stdev[1]}"
        )
        dcemd_to_avg_stdev[
            (db, is_cache, entities, max_walks, depth)
        ] = avg_stdev

    for k, v in dcemd_to_avg_stdev.items():
        print(
            f"(db={k[0]}, is_cache={k[1]}, entities={k[2]},"
            + f"depth={k[3]}, max_walks={k[4]}) = "
            + f"{v[0]} +/- {v[1]} ({v[2]})"
        )
