from rdf2vec.graph import Vertex

a = Vertex("a")
b = Vertex("b")
c = Vertex("c", predicate=True, _from=a, _to=b)


class TestVertex:
    def test_eq(self):
        assert a == a

    def test_hash_with_predicate(self):
        assert hash(c) == hash((c.id, a, b, "c"))

    def test_hash_without_predicate(self):
        assert hash(a) == hash("a")

    def test_id_incremental(self):
        assert b.id == 1

    def test_id_init(self):
        assert a.id == 0

    def test_lt(self):
        assert a < b

    def test_neq(self):
        assert a != b
