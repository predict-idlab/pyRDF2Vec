from rdf2vec.graph import Vertex


def test_eq_vertex():
    assert Vertex("s") == Vertex("s")
