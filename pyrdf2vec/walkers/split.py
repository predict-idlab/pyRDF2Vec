import os
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
        func_split: The function to call for the splitting of vertices. In case
            of reimplementation, it is important to respect the signature
            imposed by `basic_split` function.
    """

    func_split = attr.ib(kw_only=True, default=None, repr=False)

    def __attrs_post_init__(self):
        if self.func_split is None:
            self.func_split = self.basic_split

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

        -> ('http://dl-learner.org/carcinogenesis#d19', 'type', 'compound', 'class')

        Args:
            walks: The random extracted walks.

        Returns:
            The list of tuples that contains split walks.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in walks:
            canonical_walk = [walk[0].name]
            for i, _ in enumerate(walk[1::], 1):
                vertices = []
                if "http" in walk[i].name:
                    vertices = " ".join(re.split("[#]", walk[i].name)).split()
                if i % 2 == 1:
                    name = vertices[1] if vertices else walk[i].name
                    preds = [
                        sub_name
                        for sub_name in re.split(r"([A-Z][a-z]*)", name)
                        if sub_name
                    ]
                    for pred in preds:
                        canonical_walk += [pred.lower()]
                else:
                    name = vertices[-1] if vertices else walk[i].name
                    objs = []
                    try:
                        objs = [str(float(name))]
                    except ValueError:
                        objs = re.sub("[^A-Za-z0-9]+", " ", name).split()
                        if len(objs) == 1:
                            match = re.match(
                                r"([a-z]+)([0-9]+)", objs[0], re.I
                            )
                            if match:
                                objs = list(match.groups())
                    for obj in objs:
                        canonical_walk += [obj.lower()]
            canonical_walk = list(dict(zip(canonical_walk, canonical_walk)))
            canonical_walks.add(tuple(canonical_walk))
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
