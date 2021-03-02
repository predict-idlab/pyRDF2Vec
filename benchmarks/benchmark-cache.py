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
    without_cache = [
        37.44,
        94.74,
        191.61,
        195.55,
        478.28,
        940.38,
        370.08,
        938.84,
        1850.04,
    ]
    with_cache = [
        3.71,
        4.25,
        4.63,
        18.97,
        20.45,
        23,
        41.67,
        44.47,
        51.3,
    ]
    v1 = [1.3, 5.81, 7.06, 19.58, 9.26, 23.51, 10.04, 11.99, 25.72]
    v2 = [0.09, 0.21, 0.06, 0.15, 0.75, 0.36, 4.27, 2.47, 5.69]

    xticks = (
        "10,10",
        "10,25",
        "10,50",
        "50,10",
        "50,25",
        "50,50",
        "100,10",
        "100,25",
        "100,50",
    )

    Benchmark.display(
        without_cache,
        with_cache,
        xlabel="entities,max_walks",
        ylabel="Time (s)",
        label1="without cache",
        label2="with cache",
        title="pyRDF2Vec with/without cache (MUTAG)",
        xticks=xticks,
    )

    # for is_cache in [False, True]:
    #     if not is_cache:
    #         kg = KG(
    #             "http://10.2.35.70:5820/mutag", is_mul_req=False, cache=None
    #         )
    #     else:
    #         kg = KG("http://10.2.35.70:5820/mutag", is_mul_req=False)

    #     emc_to_avg_stdev = {}
    #     for e, max_walks in itertools.product([10, 50, 100], [10, 25, 50]):
    #         emc_to_avg_stdev[(e, max_walks, cache)] = Benchmark(
    #             kg,
    #             [
    #                 entity
    #                 for entity in pd.read_csv(
    #                     f"benchmarks/mutag-{e}.tsv", sep="\t"
    #                 )["bond"]
    #             ],
    #             walker=[
    #                 RandomWalker(
    #                     2, max_walks=max_walks, random_state=RANDOM_STATE
    #                 )
    #             ],
    #         ).evaluate()

    # for emc, avg_stdev in emc_to_avg_stdev.items():
    #     print(
    #         f"(entities={emc[0]},max_walks={emc[1]},is_cache={emc[2]}):"
    #         + f"{avg_stdev[0]} +/- {avg_stdev[1]}"
    #     )
