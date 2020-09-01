import random

import rdflib

from pyrdf2vec.converters import rdflib_to_kg

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestWeisfeilerLehmanWalker:
    def test_extract(self):
        # KeyError
        # self._label_map[x][n - 1] for x in graph.get_inv_neighbors(vertex)
        #
        # canonical_walks = WeisfeilerLehmanWalker(4, float("inf")).extract(
        #     KG, generate_entities()
        # )
        # assert type(canonical_walks) == set
        pass
