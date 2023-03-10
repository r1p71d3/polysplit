import unittest
import polysplit as psp
import matplotlib as plt
from shapely.geometry import Polygon, Point


class TestPolySplit(unittest.TestCase):
    def setUp(self):
        # Define a square polygon for testing
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    def test_generate_points_within_polygon(self):
        # Test that generate_points_within_polygon returns the expected number of points
        num_points = 10
        points = psp.generate_points_within_polygon(self.polygon, num_points)
        self.assertEqual(points.shape[0], num_points)

        # Test that all points generated are within the polygon
        for point in points:
            self.assertTrue(self.polygon.contains(point))

    def test_shortest_path_distance(self):
        # Test that shortest_path_distance returns the expected result
        point1 = (0.0, 0.0)
        point2 = (1.0, 1.0)
        expected_result = 1.41421356  # square root of 2
        result = psp.shortest_path_distance(point1, point2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_polysplit(self):
        # Test that polysplit returns the expected number of regions
        num_regions = 5
        regions = psp.polysplit(self.polygon, k=num_regions, num_points=10, plot=False)
        self.assertEqual(len(regions), num_regions)

        # Test that all regions returned are within the original polygon
        for region in regions:
            self.assertTrue(self.polygon.contains(region))

    def test_point_to_tuple(self):
        point = Point(1, 2)
        expected_tuple = (1.0, 2.0)
        self.assertEqual(psp.shapely_point_to_tuple(point), expected_tuple)

    def test_plot_polygon_and_regions(self):
        # Define test data
        regions = psp.polysplit(self.polygon, k=5, num_points=10, plot=False)
        # Call the function
        psp.plot_polygon_and_regions(self.polygon, regions)
        # Check that a plot was created
        self.assertTrue(plt.pyplot.gcf().get_size_inches().tolist() != [0.0, 0.0])
