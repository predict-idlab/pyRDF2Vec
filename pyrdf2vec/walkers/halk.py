import itertools
import math
from collections import defaultdict
from typing import DefaultDict, List, Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk
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
            object) hop to be kept. Beware that the accumulation of several
            freq_thresholds extracts more walks, which is not always desirable.
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
        with_reverse: True to extracts parents and children hops from an
            entity, creating (max_walks * max_walks) walks of 2 * depth,
            allowing also to centralize this entity in the walks. False
            otherwise.
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

    def build_dictionary(
        self, walks: List[SWalk]
    ) -> DefaultDict[str, Set[int]]:
        """Builds a dictionary of predicates mapped with the walk(s)
        identifiers to which it appears.

        Args:
            walks: The walks to build the dictionary.

        Returns:
            The dictionary of predicate names.

        """
        vertex_to_windices: DefaultDict[str, Set[int]] = defaultdict(set)
        for i in range(len(walks)):
            for vertex in itertools.islice(walks[i], 1, None, 2):
                vertex_to_windices[vertex].add(i)
        return vertex_to_windices

    def get_rare_predicates(
        self,
        vertex_to_windices: DefaultDict[str, Set[int]],
        walks: List[SWalk],
        freq_threshold: float,
    ) -> Set[str]:
        """Gets vertices which doesn't reach a certain threshold of frequency
        of occurrence.

        Args:
            vertex_to_windices: The dictionary of predicates mapped with the
                walk(s) identifiers to which it appears.
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
        return super()._extract(kg, entity)

    # flake8: noqa: C901
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
                if not self.with_reverse:
                    curr_entity = entity_walks[0][0]
                else:
                    curr_walk = list(entity_walks[0])
                    curr_entity = curr_walk[math.trunc(len(curr_walk) / 2)]
                for walk in entity_walks:
                    if not self.with_reverse:
                        canonical_walk = [curr_entity]
                    else:
                        canonical_walk = [walk[0]]
                    reverse = True
                    j = 0
                    for i, vertex in enumerate(walk[1::2], 2):
                        if vertex not in rare_predicates:
                            if self.with_reverse:
                                obj = walk[i + j]
                                j += 1
                            else:
                                obj = walk[i] if i % 2 == 0 else walk[i + 1]
                            if self.with_reverse and reverse:
                                if obj == curr_entity:
                                    reverse = False
                            canonical_walk += [vertex, obj]
                    if len(canonical_walk) >= 3:
                        canonical_walks.append(tuple(canonical_walk))
                if canonical_walks:
                    res_halk.append(canonical_walks)
                else:
                    res_halk.append([(curr_entity,)])
        return res_halk
