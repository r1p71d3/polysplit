import unittest
import polysplit as ps
import matplotlib as plt
from shapely.geometry import Polygon, Point


class TestPolySplit(unittest.TestCase):
    def setUp(self):
        # Define a square polygon for testing
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    def test_generate_points_within_polygon(self):
        # Test that generate_points_within_polygon returns the expected number of points
        num_points = 10
        points = ps.generate_points_within_polygon(self.polygon, num_points)
        self.assertEqual(points.shape[0], num_points)

        # Test that all points generated are within the polygon
        for point in points:
            self.assertTrue(self.polygon.contains(point))

    def test_shortest_path_distance(self):
        # Test that shortest_path_distance returns the expected result
        point1 = (0.0, 0.0)
        point2 = (1.0, 1.0)
        expected_result = 1.41421356  # square root of 2
        result = ps.euclidean_distance(point1, point2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_visibility(self):
        outer_coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
        hole_coords = [(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)]
        polygon = Polygon(outer_coords, [hole_coords])
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        expected_result = 1.44222051
        result = ps.visibility(polygon, p1, p2)[1]
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_polysplit(self):
        # Test that polysplit returns the expected number of regions
        num_regions = 5
        regions = ps.polysplit_main(self.polygon, k=num_regions, num_points=10, plot=False)
        self.assertEqual(len(regions), num_regions)

        # Test that all regions returned are within the original polygon
        for region in regions:
            self.assertTrue(self.polygon.contains(region))

    def test_point_to_tuple(self):
        # test point to tuple
        point = Point(1, 2)
        expected_tuple = (1.0, 2.0)
        self.assertEqual(ps.shapely_point_to_tuple(point), expected_tuple)

    def test_plot_polygon_and_regions(self):
        # Define test data
        regions = ps.polysplit_main(self.polygon, k=5, num_points=10, plot=False)
        # Call the function
        ps.plot_polygon_and_regions(self.polygon, regions)
        # Check that a plot was created
        self.assertTrue(plt.pyplot.gcf().get_size_inches().tolist() != [0.0, 0.0])
