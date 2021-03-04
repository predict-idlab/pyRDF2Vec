import itertools
import multiprocessing
import random
import statistics
import time

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cachetools import TTLCache
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
        verbose=1,
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
            round(statistics.fmean(times), 2),
            round(statistics.stdev(times), 2),
        ]

    @staticmethod
    def display(
        samples1,
        samples2,
        xlabel=None,
        ylabel=None,
        title=None,
        label1=None,
        label2=None,
        xticks=None,
        yerr1=None,
        yerr2=None,
        c1="r",
        c2="g",
        autolabel=True,
    ):
        _, ax = plt.subplots()
        index = np.arange(len(samples1))
        bar_width = 0.35
        opacity = 0.8

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(index + bar_width, xticks)

        res1 = plt.bar(
            index,
            samples1,
            bar_width,
            alpha=opacity,
            color=c1,
            label=label1,
            yerr=yerr1,
        )

        res2 = plt.bar(
            index + bar_width,
            samples2,
            bar_width,
            alpha=opacity,
            color=c2,
            label=label2,
            yerr=yerr2,
        )

        if autolabel:
            _autolabel(res1, ax)
            _autolabel(res2, ax)

        plt.tight_layout()
        plt.show()
        plt.legend()


def _autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


if __name__ == "__main__":
    dcemd_to_avg_stdev = {}

    for db, is_cache, entities, depth, max_walks in itertools.product(
        ["mutag", "am", "dbpedia"],
        [True, False],
        [10, 25, 50],
        [10, 25, 50],
        [10, 25, 50],
    ):
        if not is_cache:
            kg = KG(
                f"http://10.2.35.70:5820/{db}", is_mul_req=False, cache=None
            )
        else:
            kg = KG(f"http://10.2.35.70:5820/{db}", is_mul_req=False)

        label = "bond"
        if db == "am":
            label = "proxy"
        else:
            label = "DBpedia_URL"

        avg_stdev = Benchmark(
            kg,
            [
                entity
                for entity in pd.read_csv(
                    f"benchmarks/mutag-{entities}.tsv", sep="\t"
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

    for db, is_cache, entities, depth, max_walks in dcemd_to_avg_stdev.items():
        print(
            f"(db={db}, is_cache={is_cache}, entities={entities},"
            + f"depth={depth}, max_walks={max_walks}) = "
            + f"{avg_stdev[0]} +/- {avg_stdev[1]}"
        )
