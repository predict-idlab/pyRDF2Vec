import re
from typing import Set

import attr

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, List, SWalk, Walk
from pyrdf2vec.walkers import RandomWalker


@attr.s
class SplitWalker(RandomWalker):
    """Splitting walking strategy which splits each vertex (except the root
    node) present in the randomly extracted walks.

    Attributes:
        _is_support_remote: True if the walking strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        kg: The global KG used later on for the worker process.
            Defaults to None.
        max_depth: The maximum depth of one walk.
        max_walks: The maximum number of walks per entity.
            Defaults to None.
        md5_bytes: The number of bytes to keep after hashing objects in
            MD5. Hasher allows to reduce the memory occupied by a long
            text. If md5_bytes is None, no hash is applied.
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
        func_split: The function to call for the splitting of vertices. In case
            of reimplementation, it is important to respect the signature
            imposed by `basic_split` function.
            Defaults to func_split.

    """

    func_split = attr.ib(kw_only=True, default=None, repr=False)

    def __attrs_post_init__(self):
        if self.func_split is None:
            self.func_split = self.basic_split

    # flake8: noqa: C901
    def basic_split(self, walks: List[Walk]) -> Set[SWalk]:
        """Splits vertices of random walks for an entity based. To achieve
        this, each vertex (except the root node) is split according to symbols
        and capitalization by removing any duplication.

        Some examples:
        ('http://dl-learner.org/carcinogenesis#d19'),
         'http://dl-learner.org/carcinogenesis#hasBond'),
         'http://dl-learner.org/carcinogenesis#bond3209')

        -> ('http://dl-learner.org/carcinogenesis#d19', 'has', 'bond', '3209')

        ('http://dl-learner.org/carcinogenesis#d19'),
         'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
         'http://dl-learner.org/carcinogenesis#Compound')

        -> ('http://dl-learner.org/carcinogenesis#d19', 'type', 'compound')

        Args:
            walks: The random extracted walks.

        Returns:
            The list of tuples that contains split walks.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in walks:
            tmp_vertices = []  # type: ignore
            canonical_walk = []
            if self.with_reverse:
                canonical_walk = [walk[0].name]
            for i, _ in enumerate(walk[1::], 1):
                vertices = []
                if "http" in walk[i].name:
                    vertices = " ".join(re.split("[#]", walk[i].name)).split()
                name = vertices[-1] if vertices else walk[i].name

                vertices = [
                    sub_name
                    for sub_name in re.split(r"([A-Z][a-z]*)", name)
                    if sub_name
                ]
                if i % 2 != 1:
                    try:
                        vertices = [str(float(name))]
                    except ValueError:
                        vertices = re.sub("[^A-Za-z0-9]+", " ", name).split()
                        if len(vertices) == 1:
                            match = re.match(
                                r"([a-z]+)([0-9]+)", vertices[0], re.I
                            )
                            if match:
                                vertices = list(match.groups())

                if self.with_reverse:
                    if tmp_vertices:
                        tmp_vertices.append(vertices)
                        tmp_vertices.reverse()
                        for v in tmp_vertices:
                            for vertex in v:
                                canonical_walk += [vertex.lower()]
                        tmp_vertices = []
                    else:
                        tmp_vertices.append(vertices)
                else:
                    for vertex in vertices:
                        canonical_walk += [vertex.lower()]
            if self.with_reverse:
                canonical_walk += [walk[0].name]
            canonical_walks.add(
                tuple(list(dict(zip(canonical_walk, canonical_walk))))
            )
        return canonical_walks

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
        return {entity.name: list(self.func_split(walks))}
