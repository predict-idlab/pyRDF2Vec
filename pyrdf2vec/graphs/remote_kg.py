from typing import List, Tuple

import rdflib
from SPARQLWrapper import JSON, SPARQLWrapper


class RemoteKnowledgeGraph:
    """Represents a Knowledge Graph from a SPARQL endpoint."""

    def __init__(self, location):
        self.location = location
        self.endpoint = SPARQLWrapper(self.location)

    def get_hops(self, vertex: str) -> List[Tuple[str, str]]:
        """Returns a hop (vertex -> predicate -> object)

        Args:
            vertex: The name of the vertex to get the hops.

        Returns:
            The hops of a vertex in a (predicate, object) form.

        """
        if not vertex.startswith("http://"):
            return []

        query = (
            """
            SELECT ?p ?o WHERE {
                <"""
            + vertex
            + """> ?p ?o .
            }
        """
        )
        self.endpoint.setQuery(query)
        self.endpoint.setReturnFormat(JSON)
        results = self.endpoint.query().convert()
        hops = []
        for result in results["results"]["bindings"]:
            hops.append((result["p"]["value"], result["o"]["value"]))
        return hops
