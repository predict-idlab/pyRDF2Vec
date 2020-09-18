import random
import pytest

import rdflib
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.converters import rdflib_to_kg


# TODO: Can we use pytest.fixtures to create a transformer automatically?


LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])
LEAKY_KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[])
train_df = pd.read_csv("samples/mutag-train.tsv", sep='\t', header=0)
entities = [rdflib.URIRef(x) for x in train_df['bond']]
entities_subset = entities[:5]


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