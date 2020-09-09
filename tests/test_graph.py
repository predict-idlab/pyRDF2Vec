from pyrdf2vec.converters import rdflib_to_kg
from pyrdf2vec.graph import Vertex

a = Vertex("a")
b = Vertex("b")
c = Vertex("c", predicate=True, vprev=a, vnext=b)


class TestVertex:
    def test_eq(self):
        assert a == a

    def test_eq_with_none(self):
        assert a is not None

    def test_id_incremental(self):
        assert b.id == 1

    def test_id_init(self):
        assert a.id == 0

    def test_neq(self):
        assert a != b


LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = rdflib_to_kg("samples/mutag.owl", label_predicates=[LABEL_PREDICATE])


class TestKnowledgeGraph:
    def test_visualise(self):
        KG.visualise()
        assert True

    def test_add_edge(self):
        KG.add_edge(a, c)
        assert True

    def test_get_neighbors(self):
        KG.get_neighbors()
        assert True

    def test_inv_get_neighbors(self):
        KG.get_inv_neighbors()
        assert True

    def test_remove_edge(self):
        KG.remove_edge(a, c)
        assert True
