import random

import rdflib

from rdf2vec.converters import rdflib_to_kg
from rdf2vec.walkers import AnonymousWalker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestAnonymousWalker:
    def test_extract(self):
        canonical_walks = AnonymousWalker(4, float("inf")).extract(
            KG, generate_entities()
        )
        assert type(canonical_walks) == set
