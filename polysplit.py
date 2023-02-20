"""
This module contains the function polysplit, which splits a plygon into regions using k-medoids clustering
and a distance metric based on the shortest distance between two points within the polygon.
"""

from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import voronoi_diagram
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def polysplit(polygon: Polygon, k: int = 2, num_points: int = 1000, plot=False) -> list:
    """Split a polygon into k regions using k-medoids clustering and a distance metric based on the shortest distance
    between two points within the polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to split.
    k : int, optional
        The number of regions to split the polygon into, by default 2
    num_points : int, optional
        The number of points to generate within the polygon, by default 1000
    plot : bool, optional
        Whether to plot the results, by default False

    Returns
    -------
    list
        A list of shapely polygons, each of which is a region within the input polygon.
    """

    # Generate a set of points within the polygon
    points = generate_points_within_polygon(polygon, num_points)

    # convert points to a list of tuples
    points = [shapely_point_to_tuple(point) for point in points]

    # Calculate the distance matrix between each pair of points
    distances = pairwise_distances(points, metric=shortest_path_distance)

    # Cluster the points using K-medoids
    # TODO: dynamically consider only the points withing a certain radius around the centroid
    # TODO: then use the shortest path algo to prune far away points
    kmedoids = KMedoids(n_clusters=k, metric='precomputed').fit(distances)

    # Get the cluster labels for each point
    labels = kmedoids.labels_

    # Get the region polygons
    regions = get_regions(polygon, np.array(points), labels, k)

    # Plot the results
    if plot:
        plot_polygon_and_regions(polygon, regions)

    return regions


def generate_points_within_polygon(polygon: Polygon, num_points: int) -> np.ndarray:
    """Generate a set of points within a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to generate points within.
    num_points : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_points, 2) containing the points.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    # add a uniform grid of at most num_points/2 points to the polygon
    x = np.linspace(min_x, max_x, int(np.sqrt(num_points)))
    y = np.linspace(min_y, max_y, int(np.sqrt(num_points)))
    xv, yv = np.meshgrid(x, y)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            point = Point([xv[i, j], yv[i, j]])
            if polygon.contains(point):
                points.append(point)

    # add the remainder of the points at random
    while len(points) < num_points:
        point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if polygon.contains(point):
            points.append(point)

    return np.array(points)


def get_regions(polygon: Polygon, points: np.ndarray, labels: np.ndarray, k: int) -> list:
    """Get the regions from the points and labels using Voronoi diagram on centroids.

    Parameters
    ----------
    points : np.ndarray
        The points.
    labels : np.ndarray
        The labels.
    k : int
        The number of regions.

    Returns
    -------
    list
        A list of shapely.geometry.Polygon objects.
    """
    centroids = []
    for i in range(k):
        # Get the points in the region
        region_points = points[labels == i]
        # Get the centroid of the region
        # TODO: use polylabel instead
        centroid = np.mean(region_points, axis=0)
        centroids.append(centroid)

    # TODO: remove this when done testing
    print(centroids)
    plt.scatter(*zip(*centroids))
    plt.show()

    # Get the regions using the Voronoi diagram on the centroids
    # TODO: fix the representation of the regions
    regions = voronoi_diagram(MultiPoint(centroids), Polygon(centroids).convex_hull)

    # Convert regions from GeometryCollection to list of Polygons
    regions = [region for region in regions.geoms if isinstance(region, Polygon)]

    # Trim the regions to the input polygon
    regions = [region.intersection(polygon) for region in regions]

    return regions


def shortest_path_distance(point1: Point, point2: Point) -> float:
    """Calculate the shortest path distance between two points.

    Parameters
    ----------
    point1 : shapely.geometry.Point
        The first point.
    point2 : shapely.geometry.Point
        The second point.

    Returns
    -------
    float
        The shortest path distance between the points.
    """
    # Get the coordinates of the points
    x1, y1 = point1
    x2, y2 = point2
    # Calculate the distance
    # TODO: use the shortest path algorithm instead
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance


def plot_polygon_and_regions(polygon: Polygon, regions: list):
    """Plot the input polygon and the resulting regions.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon.
    regions : list
        A list of shapely.geometry.Polygon objects.
    """
    # Plot the polygon
    plt.plot(*polygon.exterior.xy)
    # Plot the regions
    for region in regions:
        plt.plot(*region.exterior.xy)

    plt.show()


def shapely_point_to_tuple(point: Point) -> tuple:
    """Convert a shapely point to a tuple.

    Parameters
    ----------
    point : shapely.geometry.Point
        The point to convert.

    Returns
    -------
    tuple
        The tuple representation of the point.
    """
    return tuple(point.coords[0])


def main():
    # Create a polygon
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    # Split the polygon
    # TODO: test on a large number of points and regions
    # TODO: test on a polygon with holes
    regions = polysplit(polygon, k=5, num_points=20, plot=True)
    print(regions)


if __name__ == '__main__':
    main()
