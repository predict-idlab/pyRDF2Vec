from pyrdf2vec.graph import Vertex
from pyrdf2vec.walkers import RandomWalker


class AnonymousWalker(RandomWalker):
    """Defines the anonymous walking strategy.

    Attributes:
        depth (int): The depth per entity.
        walks_per_graph (float): The maximum number of walks per entity.

    """

    def __init__(self, depth, walks_per_graph):
        super().__init__(depth, walks_per_graph)

    def extract(self, graph, instances):
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph (graph.KnowledgeGraph): The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances (list): The instances to extract the knowledge graph.

        Returns:
            set: The 2D matrix with its:
                number of rows equal to the number of provided instances;
                number of column equal to the embedding size.

        """
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                canonical_walk = []
                str_walk = [x.name for x in walk]
                for i, hop in enumerate(walk):
                    if i == 0:
                        canonical_walk.append(hop.name)
                    else:
                        canonical_walk.append(str(str_walk.index(hop.name)))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
