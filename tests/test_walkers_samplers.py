import functools
import itertools

import pandas as pd
import pytest
import rdflib

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

KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl",
    skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
)
TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)
ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]


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
        functools.partial(samp, inverse=True): (  # type: ignore
            "Inverse %s" % desc
        )
        for samp, desc in SAMPLERS.items()
        if samp is not UniformSampler
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
            walkers=[walker(2, 5, sampler(), seed=42)]
        ).fit_transform(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)
