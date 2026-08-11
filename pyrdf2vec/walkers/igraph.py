from hashlib import md5
from typing import List, Optional, Set
import random

import attr

import numpy as np
import pandas as pd

from igraph import Graph

from itertools import groupby

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import EntityWalks, SWalk, Walk
from pyrdf2vec.walkers import Walker

@attr.s
class RandomWalker(Walker):
    """Random walking strategy which extracts walks from a root node using the
    Depth First Search (DFS) algorithm if a maximum number of walks is
    specified, otherwise the Breadth First Search (BFS) algorithm is used.

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

    """

    md5_bytes = attr.ib(
        kw_only=True,
        type=Optional[int],
        default=8,
        repr=False,
    )

    def transformKG(self, kg: KG):
        """
        Transform each pyRDF2Vec KG object into igraph graph

        Args:
            kg: Knowledge Graph of PYRDF2Vec object
        """
        # extract nodes and edges from KG storing them into tuples
        nodeTuple = tuple((vertex for vertex in kg._vertices if not vertex.predicate))
        predicateTuple = tuple((vertex for vertex in kg._vertices if vertex.predicate))
        # merge node and edge tuples into one
        tupleValue = nodeTuple + predicateTuple
        # transform tuple into graph and store into class variable
        self.graph = Graph.TupleList(tupleValue, directed=True, edge_attrs='description')

    def predicateGeneration(self, pathList):
        """Generate path sequence for a list of single paths based on graph object
        using shortest path algorithm
        
        Args:
            pathList: List of paths for one vertex ID
        
        Returns:
            List of path sequences in a tuple
        """
        graph = self.graph
        predValues = np.array([e.attributes()['description'] for e in graph.es(pathList)])
        nodeSequence = np.array([graph.vs().select(e.tuple).get_attribute_values('name') for e in graph.es(pathList)]).flatten()
        nodeSequence = np.array([key for key, _group in groupby(nodeSequence)])
        pathSequence = np.insert(predValues, np.arange(len(nodeSequence)), nodeSequence)
        pathSequence = tuple(pathSequence)
        return pathSequence


    def _bfs(self, kg: KG , idNumber: int, is_reverse:bool = False):
        """Extracts random walks for an entity based on Knowledge Graph using
        the Depth First Search (DFS) algorithm.
        
        Args:
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False
            idNumber: ID number of a node within a graph
        Returns: 
            nodeIndex: Index of node in graph
            dfsList: List of unique walks for the provided entitiy
        """
        graph = self.transformKG()
        # extract node index for vertices
        nodeIndex = graph.vs.find(idNumber).index
        # define orientation of graph
        orient = 'out' if is_reverse else'all'
        # perform breadth-first search extraction
        bfsList = self.graph.bfsiter(nodeIndex, orient, advanced=True)
        return nodeIndex, bfsList
    
    def _dfs(self, is_reverse:bool, idNumber):
        """Extracts random walks for an entity based on Knowledge Graph using
        the Depth First Search (DFS) algorithm.
        
        Args:
            is_reverse: True to get the parent neighbors instead of the child
                neighbors, False otherwise.
                Defaults to False
            idNumber: ID number of a node within a graph
        Returns:
            nodeIndex: Index of node in graph
            dfsList: List of unique walks for the provided entitiy
        """
        assert self.max_walks is not None
        nodeIndex = self.graph.vs.find(idNumber).index
        orient = 'out' if is_reverse else'all'
        dfsList = self.graph.dfsiter(nodeIndex, orient, advanced=True)
        return nodeIndex, dfsList
    
    def extract_walks(self, entity: Vertex) -> List[Walk]:
        """Extracts random walks for an entity based on Knowledge Graph using
        the Depth First Search (DFS) algorithm if a maximum number of walks is
        specified, otherwise the Breadth First Search (BFS) algorithm is used.

        Args:
            entity: The root node to extract walks.
        Returns:
            The list of unique walks for the provided entity.
        """
        fct_search = self._bfs if self.max_walks is None else self._dfs
        nodeIndex, fctList = fct_search(self.with_reverse, entity)
        distanceList = tuple((nodePath for nodePath in fctList if nodePath[1] <= self.distance))
        vertexList = tuple((vertexElement[0] for vertexElement in distanceList))
        # limit maximum walks to maximum length of walkSequence length
        maxWalks = len(vertexList) if len(vertexList) < self.max_walks else self.max_walks
        # random sample defined maximumWalk from vertexList list
        random.seed(15)
        vertexList = random.sample(vertexList, maxWalks)
        shortestPathList = self.graph.get_shortest_paths(v=nodeIndex, to=vertexList, output='epath')
        pathSequence = list(map(self.predicateGeneration, shortestPathList))
        return pathSequence
    

    def _map_vertex(self, entity: Vertex, pos: int) -> str:
        """Maps certain vertices to MD5 hashes to save memory. For entities of
        interest (provided by the user to the extract function) and predicates,
        the string representation is kept.

        Args:
            entity: The entity to be mapped.
            pos: The position of the entity in the walk.

        Returns:
            A hash (string) or original string representation.

        """
        if (
            entity.name in self._entities
            or pos % 2 == 1
            or self.md5_bytes is None
        ):
            return entity.name
        else:
            ent_hash = md5(entity.name.encode()).digest()
            return str(ent_hash[: self.md5_bytes])


    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts random walks for an entity based on a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.
            entity: The root node to extract walks.

        Returns:
            A dictionary having the entity as key and a list of tuples as value
            corresponding to the extracted walks.

        """
        canonical_walks: Set[SWalk] = set()
        for walk in self.extract_walks(kg, entity):
            canonical_walk: List[str] = [
                self._map_vertex(vertex, i) for i, vertex in enumerate(walk)
            ]
            canonical_walks.add(tuple(canonical_walk))
        return {entity.name: list(canonical_walks)}