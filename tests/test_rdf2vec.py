import functools
import itertools
import random
import pytest

import numpy as np
import pandas as pd
import pytest
import rdflib
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.converters import rdflib_to_kg
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from pyrdf2vec.samplers import (  # isort: skip
    ObjPredFreqSampler,
    PredFreqSampler,
    UniformSampler,
)

# To fix: ObjFreqSampler, PageRankSampler

# To fix: AnonymousWalker, CommunityWalker, HalkWalker, NGramWalker,
#         WeisfeilerLehmanWalker; WalkletWalker,

# TODO: Can we use pytest.fixtures to create a transformer automatically?

np.random.seed(42)
random.seed(42)


LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
kg = KG("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])
LEAKY_KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[])
train_df = pd.read_csv("samples/mutag-train.tsv", sep='\t', header=0)
entities = [rdflib.URIRef(x) for x in train_df['bond']]
entities_subset = entities[:5]

test_data = pd.read_csv("samples/mutag/test.tsv", sep="\t")
train_data = pd.read_csv("samples/mutag/train.tsv", sep="\t")

train_entities = [rdflib.URIRef(x) for x in train_data["bond"]]
test_entities = [rdflib.URIRef(x) for x in test_data["bond"]]
entities = train_entities + test_entities

# TODO: This should be moved to sampler/walker tests
SAMPLER_CLASSES = {
    # ObjFreqSampler: "Object Frequency",
    ObjPredFreqSampler: "Predicate-Object Frequency",
    # PageRankSampler: "PageRank",
    PredFreqSampler: "Predicate Frequency",
    UniformSampler: "Uniform",
}
SAMPLER_CLASSES.update(
    {
        functools.partial(samp, inverse=True): (  # type: ignore
            "Inverse %s" % desc
        )
        for samp, desc in SAMPLER_CLASSES.items()
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
class TestRDF2VecTransformer:
    def test_fit(self):
        transformer = RDF2VecTransformer()

        # The provided entities to fit() should be in the KG
        with pytest.raises(ValueError):
            non_existing_entities = ['does', 'not', 'exist']
            transformer.fit(KG, non_existing_entities)

        # Check if the fit doesn't crash.
        transformer.fit(KG, entities_subset)
        assert True

    def test_fit_transform(self):
        transformer = RDF2VecTransformer()

        # Check if result of fit_transform() is the same as fit().transform()
        walk_embeddings_1 = transformer.fit_transform(KG, entities_subset)
        walk_embeddings_2 = (transformer.fit(KG, entities_subset)
        	                            .transform(entities_subset))
        np.testing.assert_array_equal(walk_embeddings_1, walk_embeddings_2)

    def test_transform(self):
        transformer = RDF2VecTransformer()

        # fit() should be called first before calling transform()
        with pytest.raises(NotFittedError):
            _ = transformer.transform(entities_subset)

        # Check if doesn't crash.
        transformer.fit(KG, entities_subset)
        features_vectors = transformer.transform(entities_subset)

        # Should return a list
        assert type(features_vectors) == list

    # TODO: This should be moved to sampler/walker tests
    @pytest.mark.parametrize(
        "walker, sampler", itertools.product(WALKER_CLASSES, SAMPLERS)
    )
    def test_fit_transform_with_cbow(self, walker, sampler):
        transformer = RDF2VecTransformer(
            walkers=[walker(2, 5, sampler())], sg=0
        )
        assert transformer.fit_transform(KNOWLEDGE_GRAPH, entities)

    # TODO: This should be moved to sampler/walker tests
    @pytest.mark.parametrize(
        "walker, sampler", itertools.product(WALKER_CLASSES, SAMPLERS)
    )
    def test_fit_transform_with_skip_gram(self, walker, sampler):
        transformer = RDF2VecTransformer(
            walkers=[walker(2, 5, sampler())], sg=1
        )
        assert transformer.fit_transform(KNOWLEDGE_GRAPH, entities)
