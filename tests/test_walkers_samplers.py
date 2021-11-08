import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.rdf2vec import RDF2VecTransformer

from pyrdf2vec.walkers import (  # isort: skip
    AnonymousWalker,
    HALKWalker,
    NGramWalker,
    RandomWalker,
    WalkletWalker,
    WLWalker,
)
from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    WideSampler,
)


LOOP = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Dean"],
    ["Dean", "loves", "Alice"],
]
LONG_CHAIN = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Mathilde"],
    ["Mathilde", "knows", "Alfy"],
    ["Alfy", "knows", "Stephane"],
    ["Stephane", "knows", "Alfred"],
    ["Alfred", "knows", "Emma"],
    ["Emma", "knows", "Julio"],
]
URL = "http://pyRDF2Vec"

KG_LOOP = KG()
KG_CHAIN = KG()

IS_INVERSE = [False, True]
IS_SPLIT = [False, True]
KGS = [KG_LOOP, KG_CHAIN]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]
SAMPLERS = [
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    WideSampler,
]
WALKERS = [
    AnonymousWalker,
    HALKWalker,
    NGramWalker,
    RandomWalker,
    WalkletWalker,
    WLWalker,
]


class TestWalkerSampler:
    @pytest.fixture(scope="session")
    def setup(self):
        for i, graph in enumerate([LOOP, LONG_CHAIN]):
            for row in graph:
                subj = Vertex(f"{URL}#{row[0]}")
                obj = Vertex((f"{URL}#{row[2]}"))
                pred = Vertex(
                    (f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj
                )
                if i == 0:
                    KG_LOOP.add_walk(subj, pred, obj)
                else:
                    KG_CHAIN.add_walk(subj, pred, obj)

    @pytest.mark.parametrize(
        "kg, walker, sampler, is_inverse, is_split",
        list(
            itertools.product(
                KGS,
                WALKERS,
                SAMPLERS,
                IS_INVERSE,
                IS_SPLIT,
            )
        ),
    )
    def test_fit_transform(
        self, setup, kg, walker, sampler, is_inverse, is_split
    ):
        if "UniformSampler" in str(sampler):
            sampler = sampler()
        else:
            sampler = sampler(is_inverse, is_split)

        assert RDF2VecTransformer(
            walkers=[walker(2, 5, sampler, random_state=42)]
        ).fit_transform(
            kg, [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        )
