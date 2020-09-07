from pyrdf2vec.graph import KnowledgeGraph


class Walker:
    """Base class for the walking strategies.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth: int, walks_per_graph: float):
        self.depth = depth
        self.walks_per_graph = walks_per_graph

    def extract(self, graph: KnowledgeGraph, instances: list) -> set:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        raise NotImplementedError("This must be implemented!")

    def print_walks(
        self, graph: KnowledgeGraph, instances: list, file_name: str
    ) -> None:
        """Prints the walks of a knowledge graph.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.
            file_name: The filename that contains the rdflib.Graph

        """
        walks = self.extract(graph, instances)
        walk_strs = []
        for _, walk in enumerate(walks):
            s = ""
            for i in range(len(walk)):
                s += f"{walk[i]} "
                if i < len(walk) - 1:
                    s += "--> "
            walk_strs.append(s)

        with open(file_name, "w+") as f:
            for s in walk_strs:
                f.write(s)
                f.write("\n\n")
