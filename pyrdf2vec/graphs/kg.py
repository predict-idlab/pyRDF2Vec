import abc
from typing import List, Tuple


class Vertex(object):
    vertex_counter = 0

    def __init__(self, name, predicate=False, vprev=None, vnext=None):
        self.name = name
        self.predicate = predicate
        self.vprev = vprev
        self.vnext = vnext

        self.id = Vertex.vertex_counter
        Vertex.vertex_counter += 1

    def __eq__(self, other):
        if other is None:
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if self.predicate:
            return hash((self.id, self.vprev, self.vnext, self.name))
        else:
            return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


class KG(metaclass=abc.ABCMeta):
    """Represents a Knowledge Graph."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_hops(self, vertex: str) -> List[Tuple[str, str]]:
        """Returns a hop (vertex -> predicate -> object)

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        raise NotImplementedError("This has to be implemented")
