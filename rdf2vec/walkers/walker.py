class Walker:
    """Base class for the walking strategies.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth, walks_per_graph):
        self.depth = depth
        self.walks_per_graph = walks_per_graph

    def extract(self, graph, instances):
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (array-like): The instances to extract the knowledge graph.

        Returns:
            list: The 2D matrix with its:
                number of rows equal to the number of provided instances;
                number of column equal to the embedding size.

        """
        raise NotImplementedError("This must be implemented!")

    def print_walks(self, graph, instances, file_name):
        """Prints the walks of a knowledge graph.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.
            The graph from which the neighborhoods are extracted for the
            provided instances.
            instances (array-like): The instances to extract the knowledge
                graph.
            file_name (str): The filename that contains the rdflib.Graph

        """
        walks = self.extract(graph, instances)
        walk_strs = []
        for walk_nr, walk in enumerate(walks):
            s = ""
            for i in range(len(walk)):
                if i % 2:
                    s += "{} ".format(walk[i])
                else:
                    s += "{} ".format(walk[i])

                if i < len(walk) - 1:
                    s += "--> "
            walk_strs.append(s)

        with open(file_name, "w+") as myfile:
            for s in walk_strs:
                myfile.write(s)
                myfile.write("\n\n")
