import attr
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.typings import Hop
from multiprocessing import Manager, Process
from collections import defaultdict
from shapely import from_wkt, GeometryCollection, Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, Geometry
import math
import re
from typing import List, Set, Tuple
from pyrdf2vec.graphs.vertex import Vertex
from tqdm.auto import tqdm
import multiprocessing as mp
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import numpy as np

@attr.s
class GeoSampler(Sampler):
    """Uniform sampling strategy that assigns a uniform weight to each edge in
    a Knowledge Graph, in order to prioritizes walks with strongly connected
    entities.

    Attributes:
        _is_support_remote: True if the sampling strategy can be used with a
            remote Knowledge Graph, False Otherwise
            Defaults to True.
        _random_state: The random state to use to keep random determinism with
            the sampling strategy.
            Defaults to None.
        _vertices_deg: The degree of the vertices.
            Defaults to {}.
        _visited: Tags vertices that appear at the max depth or of which all
            their children are tagged.
            Defaults to set.
        inverse: True if the inverse algorithm must be used, False otherwise.
            Defaults to False.
        split: True if the split algorithm must be used, False otherwise.
            Defaults to False.

    """

    inverse = attr.ib(
        init=False,
        default=False,
        type=bool,
        validator=attr.validators.instance_of(bool),
    )

    split = attr.ib(
        init=False,
        default=False,
        type=bool,
        validator=attr.validators.instance_of(bool),
    )

    njobs = attr.ib(
        init=False,
        default=1,
        type=int,
        validator=attr.validators.instance_of(int),
    )

    spatial_weighting_strategy = attr.ib(
        init=False,
        default="min_max_normalization",
        type=str,
        validator=attr.validators.instance_of(str),
    )

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    def extract_geometry(self, node_label: str) -> tuple:
        """Method in order to extract geometries from Node labels which do not contain single geometries.
        It uses a regex pattern to extract from node labels the respective geometry. It assumes a correct WKT
        storage of the geometry as a node label

        Args:
            node_label (str): label of node of Knowledge Graph

        Returns:
            tuple: Tuple containing a boolean value and string value.
                If the regex pattern does not discover a WKT geometry, we assign to first position false and at second position empty string.
                If the regex pattern discovers a WKT geometry, we assign to first position true and at second position regex extracted string.
        """
        # regex for overall extraction of geometries if string is not a real WKT geometry
        node_label = str(node_label)
        extracted_geometry = re.search(
            r'(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)(\s?\(.+)(\))',
            node_label)
        # If overall extracted geometry is not an empty list, set to true. Otherwise, the method returns false
        if extracted_geometry:
            return (True, extracted_geometry[0])
        else:
            return (False, "")

    def neighborhood(self, kg: KG, vertices: List[Vertex], order: int = 1,
                     include_self: bool = True) -> List[Set[Vertex]]:
        """Mimics igraph.Graph.neighborhood.

        Args:
            kg: The Knowledge Graph.
            vertices: List of starting Vertex nodes.
            order: How many hops away to include (default 1).
            include_self: Whether to include the starting nodes themselves.

        Returns:
            A set of Vertex instances in the neighborhood.
        """
        results = []

        for root in vertices:
            visited = set([root] if include_self else [])
            frontier = set([root])

            for _ in range(order):
                next_frontier = set()
                for vertex in frontier:
                    hops = kg.get_hops(vertex, is_reverse=True)+kg.get_hops(vertex)
                    for _, neighbor in hops:
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
                visited.update(next_frontier)
                frontier = next_frontier

            results.append(visited)

        return results

    def process_chunk(self, geo_nodes_batch, graph, geometry_cache,
                      visited_node_graph, order, results):
        """
        Process a chunk of geo_nodes to search for neighbors.
        """
        local_geo_vertex_dict = defaultdict(list)
        local_current_neighbors = set()

        # Process multiple geo_nodes at once
        batch_neighbors = self.neighborhood(kg=graph, vertices=geo_nodes_batch, order=order)


        for geo_node, geo_neighbors in zip(geo_nodes_batch, batch_neighbors):
            _, geometry = geometry_cache[geo_node]
            geometry = from_wkt(geometry)
            for geo_neighbor in geo_neighbors:
                if geo_neighbor not in visited_node_graph:
                    bool_geo_neighbor, _ = geometry_cache[geo_neighbor]
                    if not bool_geo_neighbor:
                        local_geo_vertex_dict[geo_neighbor].append(geometry)
                    elif geo_neighbor == geo_node:
                        local_geo_vertex_dict[geo_neighbor].append(geometry)
                    local_current_neighbors.add(geo_neighbor)

        # Append results to shared list
        results.append((local_geo_vertex_dict, local_current_neighbors))

    def neighborhood_flood(self, geo_nodes, kg) -> defaultdict:
        """Flood the neighborhood starting from geographic nodes. Whenever a vertex already has been visited in the past, the geometry is not assigned.
        In case the visited vertex is not a geographic vertex and has not been visited before, it gets a geometry assigned. In the end, all nodes have a
        geography assigned to itself.

        Args:
            geo_nodes (list): list of indices which contain geographic geometries

        Returns:
            defaultdict: return of nodes with assigned geometries as keys and assigned geometries as elements in the list
        """
        # initialize variables to method variables
        graph = kg
        geometry_cache = self.geometry_cache
        graph_flood = True
        order = 1
        geo_vertex_dict = defaultdict(list)

        # Use multiprocessing Manager for shared state
        manager = Manager()
        visited_node_graph = set()  # Use a regular Python set for visited nodes
        previous_length_set = 0
        num_processors = 4

        while graph_flood:
            current_neighbors = set()  # Use a local set for this order

            # Dynamically calculate batch size to balance memory usage
            batch_size = max(1, math.ceil(len(geo_nodes) / (
                    num_processors * 4)))  # 4x tasks per processor
            batches = [geo_nodes[i:i + batch_size] for i in
                       range(0, len(geo_nodes), batch_size)]

            # Use multiprocessing to parallelize batch processing
            processes = []
            results = manager.list()
            for batch in batches:
                p = Process(
                    target=self.process_chunk,
                    args=(batch, graph, geometry_cache, visited_node_graph,
                          order, results)
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Aggregate results from all processes
            for local_geo_vertex_dict, local_current_neighbors in results:
                for key, value in local_geo_vertex_dict.items():
                    geo_vertex_dict[key].extend(value)
                current_neighbors.update(local_current_neighbors)

            # Update global visited nodes
            visited_node_graph.update(current_neighbors)


            # Check for convergence
            length_set = len(visited_node_graph)
            if previous_length_set == length_set:
                break
            previous_length_set = length_set
            order += 1

        return geo_vertex_dict

    def determine_optimal_chunksize(self, length_iterable: int) -> int:
        """Method to determine optimal chunksize for parallelism of unordered method

        Args:
            length_iterable (int): Size of iterable

        Returns:
            int: determined chunksize
        """
        chunksize, extra = divmod(length_iterable, self.cpu_count * 4)
        if extra:
            chunksize += 1
        return chunksize

    def geom_preparation(self, geometry_tuple: tuple) -> tuple:
        """_summary_

        Args:
            dict_key (int): _description_
            geom_dict (defaultdict): _description_

        Returns:
            list: _description_
        """
        vertex_id, geometry_list = geometry_tuple
        if len(geometry_list) > 1:
            centroid_geometry = [
                GeometryCollection(geoms=geometry_list).centroid]
        else:
            centroid_geometry = geometry_list
        return (vertex_id, centroid_geometry)

    def get_all_edges(self, kg: KG) -> Set[Tuple[Vertex, Vertex]]:
        """Returns all directed edges in the KG.

        Args:
            kg: The Knowledge Graph.

        Returns:
            A set of (from_vertex, to_vertex) tuples representing edges.
        """
        edges = set()
        for v1, neighbors in kg._transition_matrix.items():
            for v2 in neighbors:
                edges.add((v1, v2))
        return edges

    def extract_point_coordinates(self, geom: Geometry) -> list:
        """For non Point geometries we use the centroid of a geometry in order to calculate the geodesic distance between geometries.

        Args:
            geom (Geometry): Input geometry for calculation of centroid. In case of a point geometry, no point geometry is calculated

        Raises:
            ValueError: In case the geometry is not supported a Value Error is raised

        Returns:
            list: List of Longitude, Latitude tuple pair
        """
        if geom.is_empty:
            return []

        if isinstance(geom, Point):
            return [(geom.x, geom.y)]
        elif isinstance(geom,
                        (LineString, Polygon, MultiPoint, MultiLineString,
                         MultiPolygon, GeometryCollection)):
            geom_centroid = geom.centroid
            return [(geom_centroid.x, geom_centroid.y)]
        else:
            raise ValueError(
                "Given input represents an unsupported geometry type")

    def spatial_distance_calculation(self, geom1: list, geom2: list) -> float:
        """Method calculating the spatial weighting of 2 geometries. In order to calculate the geodesic distance,
        non point geometries are transformed by being represented of their

        Args:
            geom1 (list): List containing first geometry for distance calculation
            geom2 (list): List containing second geometry for distance calculation

        Returns:
            float: Spatial weight based on great circular distance
        """
        # check if both input parameters contains non empty list
        # exctract both geometries from lists
        # extracts centroid to calculate the spherical distance
        if geom1 and geom2:
            geom1 = geom1[0]
            geom2 = geom2[0]
            coords1 = self.extract_point_coordinates(geom1)
            coords2 = self.extract_point_coordinates(geom2)
            # calculate geodesic distance between centroids of geom1 and geom2 in kilometers
            try:
                distance = geodesic(coords1, coords2).kilometers
            except:
                distance = 0
        else:
            distance = 0
        return distance

    def spatial_weighting_calculation(self, geom1: list, geom2: list) -> float:
        """Method calculating the spatial weighting of 2 geometries. In order to calculate the geodesic distance,
        non point geometries are transformed by being represented of their

        Args:
            geom1 (list): List containing first geometry for distance calculation
            geom2 (list): List containing second geometry for distance calculation

        Returns:
            float: Spatial weight based on great circular distance
        """
        # check if both input parameters contains non empty list
        if geom1 and geom2:
            # exctract both geometries from lists
            geom1 = geom1[0]
            geom2 = geom2[0]
            # extracts centroid to calculate the spherical distance
            coords1 = self.extract_point_coordinates(geom1)
            coords2 = self.extract_point_coordinates(geom2)
            # calculate geodesic distance between centroids of geom1 and geom2 in kilometers
            distance = geodesic(coords1, coords2).kilometers
            # calculate spatial weighting by using exp function with negative distance
            spatial_weight = math.exp(-distance)
            return spatial_weight
        else:
            return 1.0

    def fit(self, kg: KG) -> None:
        super().fit(kg)
        self.weights = {}
        if self.njobs == -1:
            self.cpu_count = mp.cpu_count()-1
        else:
            self.cpu_count = self.njobs

        self.geometry_cache = {vertex: self.extract_geometry(vertex.name) for vertex in kg._entities}
        geo_nodes = [key for key, value in self.geometry_cache.items() if value[0]]
        if len(geo_nodes) == 0:
            raise AttributeError("No geographic entity found, therefore only normal RDF2Vec needed")
        geo_vertex_dict = self.neighborhood_flood(geo_nodes, kg)
        geo_tuple_list = tuple(geo_vertex_dict.items())

        chunksize = self.determine_optimal_chunksize(len(geo_tuple_list))

        with mp.Pool(self.cpu_count) as pool:
            geo_vertex_tuple = tuple(tqdm(pool.imap_unordered(self.geom_preparation, geo_tuple_list, chunksize=chunksize), desc="Geom centroid", total=len(geo_tuple_list)))

        geo_vertex_list = defaultdict(list, geo_vertex_tuple)

        spatial_weighting_strategy = self.spatial_weighting_strategy
        if spatial_weighting_strategy == "naive":
            # create weights for graph
            for edge in tqdm(kg.iter_triples(), desc="Graph weighting"):
                source_node, target_node = edge[0], edge[2]
                source_geom, target_geom = geo_vertex_list[source_node], geo_vertex_list[target_node]
                weighting = self.spatial_weighting_calculation(source_geom, target_geom)
                key = (source_node, edge[1], target_node)
                self.weights[key] = weighting
                #edge["weight"] = weighting
        elif spatial_weighting_strategy == "min_max_normalization":
            edges = []
            for edge in tqdm(kg.iter_triples(), desc="Graph edge distance"):
                source_node, target_node = edge[0], edge[2]
                source_geom, target_geom = geo_vertex_list[source_node], geo_vertex_list[target_node]
                distance = self.spatial_distance_calculation(source_geom, target_geom)
                #edge["distance"] = distance
                edges.append((edge[0], edge[2], edge[1], distance))

            import pandas as pd
            edge_distance_df = pd.DataFrame(edges, columns=["source", "target", "predicate", "distance"])

            #edge_distance_df = self.graph.get_edge_dataframe()

            for graph_node in tqdm(kg._entities, desc="Graph node weighting normalization"):
                node_neighbors = kg.get_hops(graph_node)+kg.get_hops(graph_node,is_reverse=True)
                if len(node_neighbors)>0:
                    node_neighbors = [x[1] for x in node_neighbors]
                    neighbor_edge_data = edge_distance_df[(((edge_distance_df["source"] == graph_node) & (edge_distance_df["target"].isin(node_neighbors))) | ((edge_distance_df["source"].isin(node_neighbors)) & (edge_distance_df["target"] == graph_node)))]
                    neighbor_distances = neighbor_edge_data["distance"].tolist()
                    neighbor_distances = np.array(neighbor_distances).reshape(-1,1)
                    min_max_scaler = MinMaxScaler()
                    normalized_distances = min_max_scaler.fit_transform(neighbor_distances)
                    normalized_distances = normalized_distances.flatten().tolist()
                    weights = [math.exp(-distance) for distance in normalized_distances]
                    edge_ids = neighbor_edge_data.index.tolist()
                    edge_spatial_weights = zip(edge_ids, weights)
                    for edge_id, weight in edge_spatial_weights:
                        row = edge_distance_df.iloc[edge_id]
                        key = (row['source'], row['predicate'], row['target'])
                        self.weights[key] = weight
                        #self.graph.es[edge_id]["weight"] = weight

    def get_weight(self, entity: Vertex, hop: Hop) -> float:
        """Gets the weight of a hop in the Knowledge Graph.

        Args:
            entity: The start vertex of the edge.
            hop: The hop of a vertex in a (predicate, object) form to get the
                weight.

        Returns:
            The weight of a given hop.

        """
        key = (entity, hop[0], hop[1])
        if key in self.weights:
            return self.weights[key]
        else:
            return 0.0
