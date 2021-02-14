import functools
import itertools

import pandas as pd
import pytest

from pyrdf2vec.graphs import KG
from pyrdf2vec.rdf2vec import RDF2VecTransformer

from pyrdf2vec.walkers import (  # isort: skip
    AnonymousWalker,
    CommunityWalker,
    HalkWalker,
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
)

ENTITIES = [
    entity
    for entity in pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)[
        "bond"
    ]
]


SAMPLER_CLASSES = {
    ObjFreqSampler: "Object Frequency",
    ObjPredFreqSampler: "Predicate-Object Frequency",
    PageRankSampler: "PageRank",
    PredFreqSampler: "Predicate Frequency",
    UniformSampler: "Uniform",
}

SAMPLERS = {
    **SAMPLER_CLASSES,
}

SAMPLERS.update(
    {
        functools.partial(sample, inverse=True): (  # type: ignore
            "Inverse %s" % desc
        )
        for sample, desc in SAMPLERS.items()
        if sample is not UniformSampler
    }
)


WALKER_CLASSES = {
    AnonymousWalker: "Anonymous",
    CommunityWalker: "Community",
    HalkWalker: "HALK",
    NGramWalker: "NGram",
    RandomWalker: "Random",
    WalkletWalker: "Walklet",
    WLWalker: "Weisfeiler-Lehman",
}


class TestRDF2Vec:
    @pytest.mark.parametrize(
        "walker, sampler", itertools.product(WALKER_CLASSES, SAMPLERS)
    )
    def test_fit_transform(self, walker, sampler):
        assert RDF2VecTransformer(
            walkers=[walker(2, 5, sampler(), random_state=42)]
        ).fit_transform(
            KG(
                "samples/mutag/mutag.owl",
                skip_predicates={
                    "http://dl-learner.org/carcinogenesis#isMutagenic"
                },
            ),
            ENTITIES[:5],
        )
