from rdf2vec.graph import Vertex

s_v = Vertex("s")
o_v = Vertex("o")
p_v = Vertex("p", predicate=True, _from=s_v, _to=o_v)


class TestVertex:
    def test_eq(self):
        assert s_v == s_v

    def test_hash_without_predicate(self):
        assert hash(s_v) == hash("s")

    def test_hash_with_predicate(self):
        assert hash(p_v) == hash((p_v.id, s_v, o_v, "p"))

    def test_id_init(self):
        assert s_v.id == 0

    def test_id_incremental(self):
        assert o_v.id == 1

    def test_lt(self):
        assert o_v < s_v

    def test_neq(self):
        assert s_v != o_v
