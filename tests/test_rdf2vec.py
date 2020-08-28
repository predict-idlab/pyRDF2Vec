import random

import pandas as pd
import rdflib
from sklearn.utils.validation import check_is_fitted

from rdf2vec._rdf2vec import RDF2VecTransformer
from rdf2vec.converters import rdflib_to_kg
from rdf2vec.walkers import RandomWalker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestRDF2VecTransformer:
    def test_fit(self):
        RDF2VecTransformer().fit(KG, generate_entities())
        assert True

    def test_fit_transform(self):
        walk_embeddings = RDF2VecTransformer().fit_transform(
            KG, generate_entities()
        )
        assert type(walk_embeddings) == list

    def test_transform(self):
        entities = generate_entities()

        transformer = RDF2VecTransformer()
        transformer.fit(KG, entities)
        features_vectors = transformer.transform(KG, entities)
        assert type(features_vectors) == list
