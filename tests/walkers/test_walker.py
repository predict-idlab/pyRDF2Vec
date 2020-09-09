import random

import pytest
import rdflib

from pyrdf2vec.converters import rdflib_to_kg
from pyrdf2vec.walkers import Walker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestWalker:
    def test_extract_not_implemented(self):
        walker = Walker(4, float("inf"))
        with pytest.raises(NotImplementedError):
            walker.extract(KG, generate_entities())
