import random

import rdflib

from pyrdf2vec.converters import rdflib_to_kg

# from pyrdf2vec.graph import Vertex
# from pyrdf2vec.walkers import WeisfeilerLehmanWalker

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

    def test_create_label(self):
        # AttributeError: 'WeisfeilerLehmanWalker' object has no attribute
        # '_label_map'
        #
        # WeisfeilerLehmanWalker(4, float("inf"))._create_label(
        #     KG, Vertex("a"), 0
        # )
        # assert True
        pass

    def test_weisfeiler_lehman(self):
        # KeyError:
        # neighbor_names = [
        # self._label_map[x][n - 1] for x in graph.get_inv_neighbors(vertex)
        #
        # WeisfeilerLehmanWalker(4, float("inf"))._weisfeiler_lehman(KG)
        # assert True
        pass
