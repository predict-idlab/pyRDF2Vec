from collections import defaultdict
from hashlib import md5
from typing import DefaultDict, List, Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class HALKWalker(RandomWalker):
    """HALK walking strategy which removes rare vertices from randomly
    extracted walks, increasing the quality of the generated embeddings while
    memory usage decreases.

    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        freq_thresholds: The minimum frequency thresholds of a (predicate,
            object) hop to be kept.
            Defaults to [0.01].
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        md5_bytes: The number of bytes to keep after hashing objects in
            MD5. Hasher allows to reduce the memory occupied by a long text. If
            md5_bytes is None, no hash is applied.
            Defaults to 8.
        random_state: The random state to use to keep random determinism with
            the walking strategy.
            Defaults to None.
        sampler: The sampling strategy.
            Defaults to UniformSampler.
        with_reverse: True to extracts children's and parents' walks from the
            root, creating (max_walks * max_walks) more walks of 2 * depth,
            False otherwise.
            Defaults to False.

    """

    freq_thresholds = attr.ib(
        kw_only=True,
        factory=lambda: [0.01],
        type=List[float],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(float),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    md5_bytes = attr.ib(kw_only=True, default=8, type=int, repr=False)

    def build_dictionary(
        self, walks: List[Walk]
    ) -> DefaultDict[Vertex, Set[int]]:
        """Builds a dictionary of vertices mapped to the extracted walk indices.

        Args:
            walks: The walks to build the dictionary.

        Returns:
            The dictionary of vertex.

        """
        vertex_to_windices: DefaultDict[Vertex, Set[int]] = defaultdict(set)
        for i in range(len(walks)):
            for vertex in walks[i]:
                vertex_to_windices[vertex].add(i)
        return vertex_to_windices

    def get_rare_vertices(
        self,
        vertex_to_windices: DefaultDict[Vertex, Set[int]],
        walks: List[Walk],
        freq_threshold: float,
    ) -> Set[Vertex]:
        """Gets vertices which doesn't reach a certain threshold of frequency
        of occurrence.

        Args:
            vertex_to_windices: The dictionary of vertices mapped to the
                extracted walk indices.
            walks: The walks.
            freq_threshold: The threshold frequency of occurrence.

        Returns:
            the infrequent vertices.

        """
        rare_vertices = set()
        for vertex in vertex_to_windices:
            if len(vertex_to_windices[vertex]) / len(walks) < freq_threshold:
                rare_vertices.add(vertex)
        return rare_vertices

    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts random walks for an entity based on a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            A dictionary having the entity as key and a list of tuples as value
            corresponding to the extracted walks.

        """
        walks = self.extract_walks(kg, entity)
        vertex_to_windices = self.build_dictionary(walks)
        canonical_walks: Set[SWalk] = set()

        for freq_threshold in self.freq_thresholds:
            rare_vertices = self.get_rare_vertices(
                vertex_to_windices, walks, freq_threshold
            )
            for walk in walks:
                canonical_walk = []
                for i, vertex in enumerate(walk):
                    if i == 0 or (
                        vertex not in rare_vertices and self.md5_bytes is None
                    ):
                        canonical_walk.append(vertex.name)
                    elif vertex not in rare_vertices:
                        canonical_walk.append(
                            str(
                                md5(vertex.name.encode()).digest()[
                                    : self.md5_bytes
                                ]
                            )
                        )
                canonical_walks.add(tuple(canonical_walk))
        return {entity.name: list(canonical_walks)}

    def _post_extract(self, res: List[EntityWalks]) -> List[List[SWalk]]:
        """Post processed walks.

        Args:
            res: the result of the walks extracted with multiprocessing.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        """
        conv_res = list(
            walks
            for entity_to_walks in res
            for walks in entity_to_walks.values()
        )
        walks: List[SWalk] = [
            walk for entity_walks in conv_res for walk in entity_walks
        ]

        predicates_dict = self.build_dictionary(walks)
        pred_thresholds = [
            self.get_rare_predicates(predicates_dict, walks, freq_threshold)
            for freq_threshold in self.freq_thresholds
        ]
        res_halk = []
        for rare_predicates in pred_thresholds:
            for entity_walks in conv_res:
                canonical_walks = []
                for walk in entity_walks:
                    canonical_walk = [walk[0]]
                    for i, vertex in enumerate(walk[1::2], 2):
                        if vertex not in rare_predicates:
                            obj = walk[i] if i % 2 == 0 else walk[i + 1]
                            canonical_walk += [vertex, obj]
                    if len(canonical_walk) > 1:
                        canonical_walks.append(tuple(canonical_walk))
                res_halk.append(canonical_walks)
        return res_halk
