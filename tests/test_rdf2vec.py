import functools
import itertools
import random

import numpy as np
import pandas as pd
import pytest
import rdflib

from pyrdf2vec._rdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KnowledgeGraph
from pyrdf2vec.walkers import RandomWalker

from pyrdf2vec.samplers import (  # isort: skip
    ObjPredFreqSampler,
    PredFreqSampler,
    UniformSampler,
)

# To fix: ObjFreqSampler, PageRankSampler

# To fix: AnonymousWalker, CommunityWalker, HalkWalker, NGramWalker,
#         WeisfeilerLehmanWalker; WalkletWalker,


np.random.seed(42)
random.seed(42)

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [rdflib.URIRef(x) for x in train_data["bond"]]
test_entities = [rdflib.URIRef(x) for x in test_data["bond"]]
entities = train_entities + test_entities

LABEL_PREDICATES = ["http://dl-learner.org/carcinogenesis#isMutagenic"]
KG = KnowledgeGraph(
    "samples/mutag/mutag.owl", label_predicates=LABEL_PREDICATES
)

SAMPLER_CLASSES = {
    # ObjFreqSampler: "Object Frequency",
    ObjPredFreqSampler: "Predicate-Object Frequency",
    # PageRankSampler: "PageRank",
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
    # AnonymousWalker: "Anonymous",
    # CommunityWalker: "Community",
    # HalkWalker: "HALK",
    # NGramWalker: "NGram",
    RandomWalker: "Random",
    # WalkletWalker: "Walklet",
    # WeisfeilerLehmanWalker: "Weisfeiler-Lehman",
}


class TestRDF2Vec:
    @pytest.mark.parametrize(
        "walker, sampler", itertools.product(WALKER_CLASSES, SAMPLERS)
    )
    def test_fit_transform_with_cbow(self, walker, sampler):
        transformer = RDF2VecTransformer(
            walkers=[walker(2, 5, sampler())], sg=0
        )
        assert transformer.fit_transform(KG, entities)

    @pytest.mark.parametrize(
        "walker, sampler", itertools.product(WALKER_CLASSES, SAMPLERS)
    )
    def test_fit_transform_with_skip_gram(self, walker, sampler):
        transformer = RDF2VecTransformer(
            walkers=[walker(2, 5, sampler())], sg=1
        )
        assert transformer.fit_transform(KG, entities)
