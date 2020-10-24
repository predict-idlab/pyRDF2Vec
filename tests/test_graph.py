from pyrdf2vec.graphs import KG, Vertex

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


KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl",
    label_predicates=["http://dl-learner.org/carcinogenesis#isMutagenic"],
)


class TestKG:
    def test_visualise(self):
        KNOWLEDGE_GRAPH.visualise()
        assert True

    def test_add_edge(self):
        KNOWLEDGE_GRAPH.add_edge(a, c)
        assert True

    def test_get_neighbors(self):
        KNOWLEDGE_GRAPH.get_neighbors(a)
        assert True

    def test_inv_get_neighbors(self):
        KNOWLEDGE_GRAPH.get_inv_neighbors(a)
        assert True

    def test_remove_edge(self):
        KNOWLEDGE_GRAPH.remove_edge(a, c)
        assert True
