import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PredFreqSampler,
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


FREQ_SAMPLERS = [ObjFreqSampler, ObjPredFreqSampler, PredFreqSampler]
IS_INVERSE = [False, True]
IS_REVERSE = [False, True]
IS_SPLIT = [False, True]
KGS = [KG_LOOP, KG_CHAIN]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]


class TestFreqSampler:
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
        "sampler, is_inverse, is_split",
        list(
            itertools.product(
                FREQ_SAMPLERS,
                IS_INVERSE,
                IS_SPLIT,
            )
        ),
    )
    def test_invalid_weight(self, sampler, is_inverse, is_split):
        with pytest.raises(ValueError):
            sampler(is_inverse, is_split).get_weight(None)

    @pytest.mark.parametrize(
        "kg, sampler, is_inverse, is_split",
        list(
            itertools.product(
                KGS,
                FREQ_SAMPLERS,
                IS_INVERSE,
                IS_SPLIT,
            )
        ),
    )
    def test_fit(self, setup, kg, sampler, is_inverse, is_split):
        sampler = sampler(is_inverse, is_split)
        assert len(sampler._counts) == 0
        sampler.fit(kg)
        if isinstance(sampler, ObjFreqSampler):
            if kg == KG_LOOP:
                assert len(sampler._counts) == 3
            else:
                assert len(sampler._counts) == 9
        elif isinstance(sampler, ObjPredFreqSampler):
            if kg == KG_LOOP:
                assert len(sampler._counts) == 3
            else:
                assert len(sampler._counts) == 8
        else:
            if kg == KG_LOOP:
                assert len(sampler._counts) == 2
            else:
                assert len(sampler._counts) == 1

    @pytest.mark.parametrize(
        "kg, root, is_reverse, sampler, is_inverse, is_split",
        list(
            itertools.product(
                KGS,
                ROOTS_WITHOUT_URL,
                IS_REVERSE,
                FREQ_SAMPLERS,
                IS_INVERSE,
                IS_SPLIT,
            )
        ),
    )
    def test_weight(
        self, setup, kg, root, is_reverse, sampler, is_inverse, is_split
    ):
        sampler = sampler(is_inverse, is_split)
        sampler.fit(kg)
        for hop in kg.get_hops(Vertex(f"{URL}#{root}"), is_reverse=is_reverse):
            if isinstance(sampler, ObjFreqSampler):
                assert sampler.get_weight(hop) <= 4
