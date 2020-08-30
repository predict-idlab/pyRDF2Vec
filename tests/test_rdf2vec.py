import random

import rdflib

from rdf2vec._rdf2vec import RDF2VecTransformer
from rdf2vec.converters import rdflib_to_kg

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


entities = generate_entities()
transformer = RDF2VecTransformer()


class TestRDF2VecTransformer:
    def test_fit(self):
        transformer.fit(KG, entities)
        assert True

    def test_fit_transform(self):
        walk_embeddings = transformer.fit_transform(KG, entities)
        assert type(walk_embeddings) == list

    def test_transform(self):
        features_vectors = transformer.transform(KG, entities)
        assert type(features_vectors) == list
