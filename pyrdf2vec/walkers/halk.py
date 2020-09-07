from collections import defaultdict
from hashlib import md5
from typing import List

from pyrdf2vec.graph import KnowledgeGraph, Vertex
from pyrdf2vec.walkers import RandomWalker


class HalkWalker(RandomWalker):
    """Defines the Hierarchical Walking (HALK) strategy.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        freq_thresholds: List[float] = [0.001],
    ):
        super().__init__(depth, walks_per_graph)
        self.freq_thresholds = freq_thresholds
        # self.lb_freq_threshold = lb_freq_threshold

    def extract(self, graph: KnowledgeGraph, instances: list) -> set:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            graph: The knowledge graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to extract the knowledge graph.

        Returns:
            The 2D matrix with its:
              number of rows equal to the number of provided instances;
              number of column equal to the embedding size.

        """
        canonical_walks = set()
        all_walks = []
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            all_walks.extend(walks)

        freq = defaultdict(set)
        for i in range(len(all_walks)):
            for hop in all_walks[i]:
                freq[hop.name].add(i)

        for freq_threshold in self.freq_thresholds:
            uniformative_hops = set()
            for hop in freq:
                # if len(freq[hop])/len(all_walks) > self.ub_freq_threshold:
                #     uniformative_hops.add(hop)
                if len(freq[hop]) / len(all_walks) < freq_threshold:
                    uniformative_hops.add(hop)

            for walk in all_walks:
                canonical_walk = []
                for i, hop in enumerate(walk):
                    if i == 0:
                        canonical_walk.append(hop.name)
                    else:
                        if hop.name not in uniformative_hops:
                            digest = md5(hop.name.encode()).digest()[:8]
                            canonical_walk.append(str(digest))
                canonical_walks.add(tuple(canonical_walk))
        return canonical_walks
